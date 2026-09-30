import importlib.util
import os
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

import aiohttp
import torch

from slime.utils.types import Sample


def _load_grade_answer_verl():
    math_utils_path = Path(__file__).resolve().parents[1] / "slime/rollout/rm_hub/math_utils.py"
    spec = importlib.util.spec_from_file_location("_opd_math_utils", math_utils_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.grade_answer_verl


try:
    grade_answer_verl = _load_grade_answer_verl()
except ModuleNotFoundError:

    def _normalize_answer(answer):
        answer = str(answer).strip()
        answer = answer.replace(",", "").replace(" ", "")
        try:
            return Decimal(answer)
        except InvalidOperation:
            return answer

    def grade_answer_verl(solution_str, ground_truth):
        matches = re.findall(r"\\boxed\{([^{}]+)\}", solution_str)
        prediction = matches[-1] if matches else solution_str.strip().splitlines()[-1]
        return _normalize_answer(prediction) == _normalize_answer(ground_truth)


async def reward_func(args, sample, **kwargs):
    if sample.metadata.get("opd_eval", False):
        return 1 if grade_answer_verl(sample.response, sample.label) else 0

    payload = {
        # "text": sample.prompt + sample.response,
        "input_ids": _teacher_scoring_input_ids(sample),
        "sampling_params": {
            "temperature": float(os.environ.get("TEACHER_TEMPERATURE", "0")),
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
        "top_logprobs_num": getattr(args, "teacher_top_logprobs_num", 20),
    }
    session_kwargs = {}
    async with aiohttp.ClientSession(**session_kwargs) as session:
        async with session.post(args.rm_url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


def _teacher_scoring_input_ids(sample: Sample) -> list[int]:
    input_ids = sample.tokens
    student_eos = os.environ.get("OPD_STUDENT_EOS_TOKEN_ID")
    teacher_eos = os.environ.get("OPD_TEACHER_EOS_TOKEN_ID")
    if not student_eos or not teacher_eos:
        return input_ids
    if sample.status != Sample.Status.COMPLETED or sample.response_length <= 0:
        return input_ids
    if not input_ids or input_ids[-1] != int(student_eos):
        return input_ids

    mapped_input_ids = input_ids.copy()
    mapped_input_ids[-1] = int(teacher_eos)
    return mapped_input_ids


def completed_only_filter(args, samples: list[Sample] | list[list[Sample]]) -> None:
    del args

    def iter_samples(items):
        for item in items:
            if isinstance(item, list):
                yield from iter_samples(item)
            else:
                yield item

    for sample in iter_samples(samples):
        sample.remove_sample = sample.status != Sample.Status.COMPLETED


def completed_only_dynamic_filter(args, sample: Sample) -> bool:
    del args
    return sample.status == Sample.Status.COMPLETED


def post_process_rewards(args, samples: list[Sample], **kwargs):
    rewards = [sample.get_reward_value(args) for sample in samples]
    response_lengths = [sample.response_length for sample in samples]
    teacher_log_probs = [
        torch.tensor([item[0] for item in reward["meta_info"]["input_token_logprobs"][1:]], dtype=torch.float32)
        for reward in rewards
    ]
    teacher_log_probs = [
        t_log_prob[-response_length:]
        for t_log_prob, response_length in zip(teacher_log_probs, response_lengths, strict=False)
    ]
    teacher_top_logprobs = None
    if getattr(args, "teacher_top_logprobs_num", 20) > 0:
        teacher_top_logprobs = [
            (reward["meta_info"].get("input_top_logprobs") or [])[1:] for reward in rewards
        ]
        teacher_top_logprobs = [
            top_logprobs[-response_length:]
            for top_logprobs, response_length in zip(teacher_top_logprobs, response_lengths, strict=False)
        ]

    for sample, t_log_probs in zip(samples, teacher_log_probs, strict=False):
        sample.teacher_log_probs = t_log_probs
        # Store correctness for rollout pass-rate logging
        if sample.label is not None:
            sample.metadata["raw_reward"] = 1 if grade_answer_verl(sample.response, sample.label) else 0

    if teacher_top_logprobs is not None:
        for sample, t_top_logprobs in zip(samples, teacher_top_logprobs, strict=False):
            sample.teacher_top_logprobs = t_top_logprobs

    return teacher_log_probs, teacher_log_probs
