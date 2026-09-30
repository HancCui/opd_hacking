# Adapt from https://github.com/OpenRLHF/OpenRLHF/blob/10c733694ed9fbb78a0a2ff6a05efc7401584d46/openrlhf/models/utils.py
# and https://github.com/OpenRLHF/OpenRLHF/blob/10c733694ed9fbb78a0a2ff6a05efc7401584d46/openrlhf/trainer/ppo_utils/experience_maker.py

import math
from argparse import Namespace

import torch
import torch.distributed as dist
import torch.nn.functional as F


@torch.compile(dynamic=True)
def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    kl_loss_type: str,
    importance_ratio: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
        kl_loss_type: Type of KL estimator (k1, k2, k3, low_var_kl).
        importance_ratio: Optional IS ratio (π_θ/π_old) for unbiased KL estimation.
    """
    log_ratio = log_probs.float() - log_probs_base.float()

    if kl_loss_type == "k1":
        kl = log_ratio
    elif kl_loss_type == "k2":
        kl = log_ratio**2 / 2.0
    elif kl_loss_type in ["k3", "low_var_kl"]:
        # The non negative kl approximation in
        # http://joschu.net/blog/kl-approx.html
        # Besides non negative, it is also unbiased and have lower variance.
        log_ratio = -log_ratio
        kl = log_ratio.exp() - 1 - log_ratio
    else:
        raise ValueError(f"Unknown kl_loss_type: {kl_loss_type}")

    # Apply IS ratio for unbiased KL estimation (DeepSeek-V3.2)
    if importance_ratio is not None:
        kl = importance_ratio * kl

    # Clamp only for low_var_kl for numerical stability
    if kl_loss_type == "low_var_kl":
        kl = torch.clamp(kl, min=-10, max=10)

    return kl


def compute_opsm_mask(
    args: Namespace,
    full_log_probs: list[torch.Tensor],
    full_old_log_probs: list[torch.Tensor],
    advantages: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Off-Policy Sequence Masking (OPSM) mask.

    Args:
        args: Configuration containing `opsm_delta` threshold.
        full_log_probs: Current policy log-probs per sample.
        full_old_log_probs: Old policy log-probs per sample.
        advantages: Advantage values per sample.
        loss_masks: Loss masks per sample.

    Returns:
        Tuple of `(opsm_mask, opsm_clipfrac)` where `opsm_mask` is a
        concatenated tensor of per-token masks and
        `opsm_clipfrac` is the count of masked sequences.
    """
    opsm_mask_list = []
    device = advantages[0].device
    opsm_clipfrac = torch.tensor(0.0, device=device)

    for full_log_prob, full_old_log_prob, advantage, loss_mask in zip(
        full_log_probs, full_old_log_probs, advantages, loss_masks, strict=False
    ):
        # Calculate sequence-level KL
        seq_kl = ((full_old_log_prob - full_log_prob) * loss_mask).sum() / torch.clamp_min(loss_mask.sum(), 1)

        # Create mask: 0 if (advantage < 0 and seq_kl > delta), else 1
        mask = ((advantage < 0) & (seq_kl > args.opsm_delta)).float()
        opsm_clipfrac += mask.sum() / torch.clamp_min(loss_mask.sum(), 1)

        opsm_mask_list.append(1 - mask)

    opsm_mask = torch.cat(opsm_mask_list, dim=0)
    return opsm_mask, opsm_clipfrac


def compute_gspo_kl(
    full_log_probs: list[torch.Tensor],
    full_old_log_probs: list[torch.Tensor],
    local_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
) -> torch.Tensor:
    """Compute GSPO-style per-sequence KL divergence.

    Args:
        full_log_probs: Current policy log-probs per sample (full or CP-local).
        full_old_log_probs: Old policy log-probs per sample (full or CP-local).
        local_log_probs: Local (CP-local) log-probs for expansion shape reference.
        loss_masks: Loss masks per sample.

    Returns:
        Concatenated tensor of per-token KL values where each token in a
        sequence has the same KL value (the sequence-level KL).
    """
    # Compute sequence-level KL and expand to per-token
    ppo_kl = [
        ((old_logprob - log_prob) * loss_mask).sum() / torch.clamp_min(loss_mask.sum(), 1)
        for log_prob, old_logprob, loss_mask in zip(full_log_probs, full_old_log_probs, loss_masks, strict=False)
    ]
    ppo_kl = [kl.expand_as(log_prob) for kl, log_prob in zip(ppo_kl, local_log_probs, strict=False)]
    ppo_kl = torch.cat(ppo_kl, dim=0)

    return ppo_kl


@torch.compile(dynamic=True)
def compute_policy_loss(
    ppo_kl: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    eps_clip_high: float,
    eps_clip_c: float | None = None,
):
    ratio = (-ppo_kl).exp()
    pg_losses1 = -ratio * advantages
    pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    clipfrac = torch.gt(pg_losses2, pg_losses1).float()

    if eps_clip_c is not None:
        assert (
            eps_clip_c > 1.0
        ), f"The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0, but get the value: {eps_clip_c}."
        pg_losses3 = -eps_clip_c * advantages
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    else:
        pg_losses = clip_pg_losses1

    return pg_losses, clipfrac


def _topk_entry_token_id(entry) -> int | None:
    if isinstance(entry, dict):
        value = entry.get("token_id", entry.get("id"))
    elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
        value = entry[1]
    else:
        value = None
    return int(value) if value is not None else None


def _topk_entry_logprob(entry) -> float | None:
    if isinstance(entry, dict):
        value = entry.get("logprob", entry.get("logp"))
    elif isinstance(entry, (list, tuple)) and entry:
        value = entry[0]
    else:
        value = None
    return float(value) if value is not None and math.isfinite(float(value)) else None


def _entropy_from_logprobs(logprobs: list[float]) -> float | None:
    if not logprobs:
        return None
    max_logprob = max(logprobs)
    probs = [math.exp(logprob - max_logprob) for logprob in logprobs]
    total = sum(probs)
    if total <= 0:
        return None
    normalized = [prob / total for prob in probs]
    return -sum(prob * math.log(prob) for prob in normalized if prob > 0)


def topk_logprobs_from_logits(
    logits: torch.Tensor,
    *,
    top_k: int,
    logits_max: torch.Tensor,
    sum_exp_logits: torch.Tensor,
    vocab_start: int = 0,
    vocab_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized top-k logprobs without materializing full-vocab logprobs."""
    if logits.size(0) == 0:
        return (
            logits.new_zeros((0, top_k)),
            torch.zeros((0, top_k), device=logits.device, dtype=torch.long),
        )

    target_k = min(top_k, logits.size(-1))
    valid_width = logits.size(-1)
    if vocab_size is not None:
        valid_width = max(0, min(valid_width, vocab_size - vocab_start))

    if valid_width > 0:
        local_k = min(target_k, valid_width)
        top_logits, top_ids = torch.topk(logits[:, :valid_width], k=local_k, dim=-1)
        top_ids = top_ids + vocab_start
    else:
        local_k = 0
        top_logits = logits.new_empty((logits.size(0), 0))
        top_ids = torch.empty((logits.size(0), 0), device=logits.device, dtype=torch.long)

    if local_k < target_k:
        pad_k = target_k - local_k
        top_logits = torch.cat(
            [top_logits, logits.new_full((logits.size(0), pad_k), float("-inf"))],
            dim=-1,
        )
        pad_ids = torch.arange(vocab_start + valid_width, vocab_start + valid_width + pad_k, device=logits.device)
        top_ids = torch.cat([top_ids, pad_ids.unsqueeze(0).expand(logits.size(0), -1)], dim=-1)

    top_logprobs = top_logits - logits_max - sum_exp_logits.log()
    return top_logprobs, top_ids


def compute_topk_overlap_and_entropy_metrics(
    *,
    teacher_top_logprobs: list[list],
    student_topk_logprobs: torch.Tensor,
    student_topk_ids: torch.Tensor,
    loss_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    values = compute_topk_overlap_and_entropy_values(
        teacher_top_logprobs=teacher_top_logprobs,
        student_topk_logprobs=student_topk_logprobs,
        student_topk_ids=student_topk_ids,
    )
    if not values:
        return {}

    length = min(loss_mask.numel(), values["top20_token_overlap_ratio"].numel())
    weights = loss_mask[:length].detach().to(device=student_topk_logprobs.device, dtype=student_topk_logprobs.dtype)
    total_weight = weights.sum()
    if total_weight.item() == 0:
        return {}

    return {key: (value[:length] * weights).sum() / total_weight for key, value in values.items()}


def compute_topk_overlap_and_entropy_values(
    *,
    teacher_top_logprobs: list[list],
    student_topk_logprobs: torch.Tensor,
    student_topk_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    device = student_topk_logprobs.device
    dtype = student_topk_logprobs.dtype
    student_ids = student_topk_ids.detach().cpu().tolist()
    student_logprobs = student_topk_logprobs.detach().cpu().tolist()

    overlap10 = []
    overlap20 = []
    teacher_entropy = []
    student_entropy = []

    for pos in range(min(len(teacher_top_logprobs), len(student_ids))):

        teacher_pos = teacher_top_logprobs[pos] or []
        teacher_ids = [_topk_entry_token_id(entry) for entry in teacher_pos[:20]]
        teacher_ids = [token_id for token_id in teacher_ids if token_id is not None]
        teacher_logprobs = [_topk_entry_logprob(entry) for entry in teacher_pos[:20]]
        teacher_logprobs = [logprob for logprob in teacher_logprobs if logprob is not None]
        student_pos_ids = [int(token_id) for token_id in student_ids[pos][:20]]
        student_pos_logprobs = [float(logprob) for logprob in student_logprobs[pos][:20]]

        cur_teacher_entropy = _entropy_from_logprobs(teacher_logprobs)
        cur_student_entropy = _entropy_from_logprobs(student_pos_logprobs)
        if cur_teacher_entropy is None or cur_student_entropy is None:
            return {}

        overlap10.append(len(set(teacher_ids[:10]).intersection(student_pos_ids[:10])) / 10.0)
        overlap20.append(len(set(teacher_ids[:20]).intersection(student_pos_ids[:20])) / 20.0)
        teacher_entropy.append(cur_teacher_entropy)
        student_entropy.append(cur_student_entropy)

    if not overlap20:
        return {}

    return {
        "top10_token_overlap_ratio": torch.tensor(overlap10, device=device, dtype=dtype),
        "top20_token_overlap_ratio": torch.tensor(overlap20, device=device, dtype=dtype),
        "teacher_top20_entropy": torch.tensor(teacher_entropy, device=device, dtype=dtype),
        "student_top20_entropy": torch.tensor(student_entropy, device=device, dtype=dtype),
    }


def compute_topk_reverse_kl_values(
    *,
    teacher_top_logprobs: list[list],
    student_topk_logprobs: torch.Tensor,
    student_topk_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if len(teacher_top_logprobs) != student_topk_ids.size(0):
        raise ValueError(
            f"teacher_top_logprobs length {len(teacher_top_logprobs)} does not match "
            f"student_topk_ids length {student_topk_ids.size(0)}"
        )

    teacher_logprobs = []
    missing = []
    student_ids = student_topk_ids.detach().cpu().tolist()

    for pos, teacher_pos in enumerate(teacher_top_logprobs):
        teacher_entries = {}
        for entry in teacher_pos or []:
            token_id = _topk_entry_token_id(entry)
            logprob = _topk_entry_logprob(entry)
            if token_id is not None and logprob is not None:
                teacher_entries[token_id] = logprob

        if not teacher_entries:
            raise ValueError(f"teacher_top_logprobs[{pos}] is empty")

        fallback_logprob = min(teacher_entries.values())
        pos_logprobs = []
        pos_missing = []
        for token_id in student_ids[pos]:
            logprob = teacher_entries.get(int(token_id))
            is_missing = logprob is None
            pos_logprobs.append(fallback_logprob if is_missing else logprob)
            pos_missing.append(float(is_missing))
        teacher_logprobs.append(pos_logprobs)
        missing.append(pos_missing)

    teacher_logprobs = torch.tensor(
        teacher_logprobs,
        device=student_topk_logprobs.device,
        dtype=student_topk_logprobs.dtype,
    )
    missing = torch.tensor(missing, device=student_topk_logprobs.device, dtype=student_topk_logprobs.dtype)

    student_logprobs = student_topk_logprobs - student_topk_logprobs.logsumexp(dim=-1, keepdim=True)
    teacher_logprobs = teacher_logprobs - teacher_logprobs.logsumexp(dim=-1, keepdim=True)
    reverse_kl = (student_logprobs.exp() * (student_logprobs - teacher_logprobs)).sum(dim=-1)

    return {
        "topk_reverse_kl": reverse_kl,
        "topk_reverse_kl_missing": missing.mean(dim=-1),
    }


def compute_topk_margin_rank_values(
    *,
    teacher_top_logprobs: list[list],
    student_topk_logprobs: torch.Tensor,
    student_topk_ids: torch.Tensor,
    margin: float,
    missing_margin: float,
) -> dict[str, torch.Tensor]:
    if len(teacher_top_logprobs) != student_topk_ids.size(0):
        raise ValueError(
            f"teacher_top_logprobs length {len(teacher_top_logprobs)} does not match "
            f"student_topk_ids length {student_topk_ids.size(0)}"
        )

    teacher_order_scores = []
    missing = []
    student_ids = student_topk_ids.detach().cpu().tolist()

    for pos, teacher_pos in enumerate(teacher_top_logprobs):
        teacher_ranks = {}
        for rank, entry in enumerate(teacher_pos or []):
            token_id = _topk_entry_token_id(entry)
            if token_id is not None and token_id not in teacher_ranks:
                teacher_ranks[token_id] = rank

        if not teacher_ranks:
            raise ValueError(f"teacher_top_logprobs[{pos}] is empty")

        missing_score = 0.0
        pos_scores = []
        pos_missing = []
        for token_id in student_ids[pos]:
            rank = teacher_ranks.get(int(token_id))
            is_missing = rank is None
            pos_scores.append(missing_score if is_missing else float(len(teacher_ranks) - rank))
            pos_missing.append(float(is_missing))
        teacher_order_scores.append(pos_scores)
        missing.append(pos_missing)

    teacher_order_scores = torch.tensor(
        teacher_order_scores,
        device=student_topk_logprobs.device,
        dtype=student_topk_logprobs.dtype,
    )
    missing = torch.tensor(missing, device=student_topk_logprobs.device, dtype=torch.bool)

    input1 = student_topk_logprobs.unsqueeze(-1).expand(-1, -1, student_topk_logprobs.size(-1))
    input2 = student_topk_logprobs.unsqueeze(-2).expand(-1, student_topk_logprobs.size(-1), -1)
    student_delta = input1 - input2
    teacher_delta = teacher_order_scores.unsqueeze(-1) - teacher_order_scores.unsqueeze(-2)
    target = teacher_delta.sign()

    valid = torch.triu(torch.ones_like(target, dtype=torch.bool), diagonal=1)
    valid = valid & (target != 0) & ~(missing.unsqueeze(-1) & missing.unsqueeze(-2))
    valid_float = valid.to(dtype=student_topk_logprobs.dtype)
    valid_count = valid_float.sum(dim=(-2, -1))
    denom = valid_count.clamp_min(1.0)

    pair_loss = torch.nn.MarginRankingLoss(margin=margin, reduction="none")(input1, input2, target)
    loss = (pair_loss * valid_float).sum(dim=(-2, -1)) / denom
    pair_acc = (((target * student_delta) > 0).to(dtype=student_topk_logprobs.dtype) * valid_float).sum(
        dim=(-2, -1)
    ) / denom
    violation_rate = ((pair_loss > 0).to(dtype=student_topk_logprobs.dtype) * valid_float).sum(dim=(-2, -1)) / denom

    return {
        "topk_margin_rank_loss": loss,
        "topk_margin_rank_missing": missing.to(dtype=student_topk_logprobs.dtype).mean(dim=-1),
        "topk_margin_rank_valid_pairs": valid_count,
        "topk_margin_rank_pair_acc": pair_acc,
        "topk_margin_rank_violation_rate": violation_rate,
    }


def compute_log_probs(logits: torch.Tensor, tokens: torch.Tensor, process_group: dist.ProcessGroup | None):
    # TODO: when megatron is not installed, fall back to naive implementation
    from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy

    # convert to [seq_len, batch_size, vocab_size] as expected by fused_vocab_parallel_cross_entropy
    logits = logits.unsqueeze(1)
    tokens = tokens.unsqueeze(1)
    return -fused_vocab_parallel_cross_entropy(logits, tokens, process_group)


# from https://github.com/volcengine/verl/blob/0bdf7f469854815177e73dcfe9e420836c952e6e/verl/utils/megatron/tensor_parallel.py#L99
class _VocabParallelEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, vocab_parallel_logits: torch.Tensor, process_group: dist.ProcessGroup) -> torch.Tensor:

        @torch.compile(dynamic=True)
        def mul_reduce(a, b):
            return (a * b).sum(dim=-1, keepdim=True)

        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=process_group)
        normalized_vocab_parallel_logits = vocab_parallel_logits - logits_max
        normalized_exp_logits = normalized_vocab_parallel_logits.exp_()
        normalized_sum_exp_logits = normalized_exp_logits.sum(dim=-1, keepdim=True)
        dist.all_reduce(normalized_sum_exp_logits, group=process_group)
        softmax_logits = normalized_exp_logits.div_(normalized_sum_exp_logits)
        sum_softmax_times_logits = mul_reduce(softmax_logits, vocab_parallel_logits)
        dist.all_reduce(sum_softmax_times_logits, group=process_group)
        entropy = logits_max + normalized_sum_exp_logits.log() - sum_softmax_times_logits
        ctx.save_for_backward(vocab_parallel_logits, softmax_logits, sum_softmax_times_logits)
        return entropy.squeeze(dim=-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        vocab_parallel_logits, softmax_logits, sum_softmax_times_logits = ctx.saved_tensors
        # reuse softmax_logits as grad
        vocab_parallel_logits.sub_(sum_softmax_times_logits)
        softmax_logits.mul_(vocab_parallel_logits)
        softmax_logits.mul_(grad_output.unsqueeze(dim=-1))
        # recover vocab_parallel_logits
        vocab_parallel_logits.add_(sum_softmax_times_logits)
        softmax_logits.mul_(-1)
        return softmax_logits, None


def compute_entropy_from_logits(logits: torch.Tensor, process_group) -> torch.Tensor:
    return _VocabParallelEntropy.apply(logits, process_group)


def get_grpo_returns(
    rewards: torch.Tensor,
    kl: list[torch.Tensor],
):
    returns = []
    for i in range(len(rewards)):
        returns.append(torch.ones_like(kl[i]) * rewards[i])
    return returns


def get_reinforce_plus_plus_returns(
    rewards: torch.Tensor,
    kl: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    response_lengths: list[int],
    total_lengths: list[int],
    kl_coef: float,
    gamma: float,
) -> list[torch.Tensor]:
    """
    Calculates discounted returns for REINFORCE++ (https://arxiv.org/pdf/2501.03262)

    Args:
        rewards (Tensor): A tensor of scalar rewards for each sequence.
        kl (List[Tensor]): List of per-token KL divergence tensors for sequence chunks.
        loss_masks (List[Tensor]): List of response-only loss masks for each full sequence.
        response_lengths (List[int]): The full length of each response sequence.
        total_lengths (List[int]): The full length of each sequence (prompt + response).
        kl_coef (float): Coefficient for the KL penalty.
        gamma (float): The discount factor.

    Returns:
        List[torch.Tensor]: A list of return (G_t) tensors for the
                            local sequence chunks owned by the current GPU rank.
    """
    from megatron.core import mpu

    cp_size = mpu.get_context_parallel_world_size()

    final_returns_chunks = []
    for i in range(len(rewards)):
        local_kl_chunk = kl[i]
        total_len, response_len = total_lengths[i], response_lengths[i]

        if cp_size > 1:
            # Step 1,2:Gather all chunks and token_offsets from all ranks and reconstruct the full response tensor by splitting and placing each part
            from slime.backends.megatron_utils.cp_utils import all_gather_with_cp

            full_kl_response = all_gather_with_cp(local_kl_chunk, total_len, response_len)
        else:
            full_kl_response = local_kl_chunk

        # Step 3: Compute returns on full response kl tensor.
        full_mask = loss_masks[i]
        assert full_mask.sum().item() > 0, f"Sequence at index {i} is fully masked."
        masked_kl = full_kl_response * full_mask
        token_level_rewards = -kl_coef * masked_kl
        last_idx = full_mask.nonzero(as_tuple=True)[0][-1]
        token_level_rewards[last_idx] += rewards[i]

        returns_for_seq = torch.zeros_like(token_level_rewards)
        running_return = 0.0
        for t in reversed(range(token_level_rewards.size(0))):
            # G_t = r_t + gamma * G_{t+1}
            running_return = token_level_rewards[t] + gamma * running_return
            returns_for_seq[t] = running_return

        # Step 4: Pick up the results corresponding to our local chunk's parts.
        if cp_size > 1:
            from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp

            local_returns_chunk = slice_log_prob_with_cp(returns_for_seq, total_len, response_len)
        else:
            local_returns_chunk = returns_for_seq

        final_returns_chunks.append(local_returns_chunk)

    return final_returns_chunks


def get_reinforce_plus_plus_baseline_advantages(
    rewards: torch.Tensor,
    kl: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    kl_coef: float,
) -> list[torch.Tensor]:
    """
    Calculates the unwhitened advantages for the REINFORCE++-baseline algorithm.
    Broadcasting the scalar (reward - group_baseline) to each token.

    Args:
        rewards (Tensor): A tensor of scalar rewards, where the group-wise
                                baseline has already been subtracted.
        kl (list[Tensor]): A list of per-token KL divergence tensors. Used to
                                 get the shape for broadcasting.
        loss_masks (list[Tensor]): A list of per-token loss masks.
        kl_coef (float): Coefficient for the KL penalty.

    Returns:
        list[Tensor]: A list of tensors containing the unwhitened advantages.
    """
    # Broadcast to get unwhitened advantages
    unwhitened_advantages = [
        torch.ones_like(kl_tensor) * reward_val - kl_coef * kl_tensor
        for kl_tensor, reward_val in zip(kl, rewards, strict=False)
    ]

    return unwhitened_advantages


def get_advantages_and_returns(
    total_len: int,
    response_len: int,
    values: torch.Tensor,
    rewards: torch.Tensor,
    gamma: float,
    lambd: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Function that computes advantages and returns from rewards and values.
    Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
    Note that rewards may include a KL divergence loss term.

    Advantages looks like this:
    Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
            - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

    Returns looks like this:
    Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

    Input:
    - values: Tensor of shape (response_size,)
    - rewards: Tensor of shape (response_size,)

    Output:
    - advantages: Tensor of shape (response_size,)
    - returns: Tensor of shape (response_size,)
    """
    from megatron.core import mpu

    cp_size = mpu.get_context_parallel_world_size()
    if cp_size > 1:
        from slime.backends.megatron_utils.cp_utils import all_gather_with_cp

        full_rewards = all_gather_with_cp(rewards, total_len, response_len)
        full_values = all_gather_with_cp(values, total_len, response_len)
    else:
        full_rewards = rewards
        full_values = values

    lastgaelam = 0
    advantages_reversed = []

    for t in reversed(range(response_len)):
        nextvalues = full_values[t + 1] if t < response_len - 1 else 0.0
        delta = full_rewards[t] + gamma * nextvalues - full_values[t]
        lastgaelam = delta + gamma * lambd * lastgaelam
        advantages_reversed.append(lastgaelam)
    full_advantages = torch.tensor(advantages_reversed[::-1], dtype=full_values.dtype, device=full_values.device)
    full_returns = full_advantages + full_values

    if cp_size > 1:
        from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp

        advantages = slice_log_prob_with_cp(full_advantages, total_len, response_len)
        returns = slice_log_prob_with_cp(full_returns, total_len, response_len)
    else:
        advantages = full_advantages
        returns = full_returns

    return advantages.detach(), returns


def get_advantages_and_returns_batch(
    total_lengths,
    response_lengths,
    values_list,
    rewards_list,
    gamma,
    lambd,
    chunked: bool = True,
):
    """
    Batched GAE with CP support.
    Input:
        total_lengths:     list[int], each sample's total_len
        response_lengths:  list[int], each sample's response_len
        values_list:       list[Tensor], each shape = [resp_len_i]
        rewards_list:      list[Tensor], same shape
    Output:
        advantages_list:   list[Tensor], each shape = [resp_len_i]
        returns_list:      list[Tensor], same shape
    """

    from megatron.core import mpu

    with torch.no_grad():
        B = len(response_lengths)
        assert B == len(values_list)
        assert B == len(rewards_list)

        cp_size = mpu.get_context_parallel_world_size()
        device = values_list[0].device
        dtype = values_list[0].dtype

        if cp_size > 1:
            from slime.backends.megatron_utils.cp_utils import all_gather_with_cp

            full_values_list = []
            full_rewards_list = []

            for total_len, resp_len, v, r in zip(
                total_lengths, response_lengths, values_list, rewards_list, strict=False
            ):
                full_v = all_gather_with_cp(v, total_len, resp_len)
                full_r = all_gather_with_cp(r, total_len, resp_len)
                full_values_list.append(full_v)
                full_rewards_list.append(full_r)

            # full_values_list[i].shape = [total_len_i]
        else:
            full_values_list = values_list
            full_rewards_list = rewards_list

        # pad to max_len for batched GAE
        max_len = max(response_lengths)

        full_values = torch.zeros(B, max_len, device=device, dtype=dtype)
        full_rewards = torch.zeros(B, max_len, device=device, dtype=dtype)

        for i in range(B):
            L = response_lengths[i]
            full_values[i, :L] = full_values_list[i][:L]
            full_rewards[i, :L] = full_rewards_list[i][:L]

        if not chunked:
            full_advantages, full_returns = vanilla_gae(
                rewards=full_rewards,
                values=full_values,
                gamma=gamma,
                lambd=lambd,
            )
        else:
            full_advantages, full_returns = chunked_gae(
                rewards=full_rewards,
                values=full_values,
                gamma=gamma,
                lambd=lambd,
            )

        advantages_list = []
        returns_list = []

        if cp_size > 1:
            from slime.backends.megatron_utils.cp_utils import slice_log_prob_with_cp

            for total_len, resp_len, adv_row, ret_row in zip(
                total_lengths,
                response_lengths,
                full_advantages,
                full_returns,
                strict=False,
            ):
                adv_full = adv_row  # shape = [resp_len_i padded to max_len]
                ret_full = ret_row

                adv_sliced = slice_log_prob_with_cp(adv_full[:resp_len], total_len, resp_len)
                ret_sliced = slice_log_prob_with_cp(ret_full[:resp_len], total_len, resp_len)

                advantages_list.append(adv_sliced)
                returns_list.append(ret_sliced)

        else:
            for i in range(B):
                L = response_lengths[i]
                advantages_list.append(full_advantages[i, :L])
                returns_list.append(full_returns[i, :L])

    return advantages_list, returns_list


def vanilla_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lambd: float,
):
    B, T = rewards.shape
    device = rewards.device
    dtype = rewards.dtype

    lastgaelam = torch.zeros(B, device=device, dtype=dtype)
    adv_rev = []

    for t in reversed(range(T)):
        next_value = values[:, t + 1] if t < T - 1 else 0.0
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        lastgaelam = delta + gamma * lambd * lastgaelam
        adv_rev.append(lastgaelam)

    full_advantages = torch.stack(adv_rev[::-1], dim=1)  # [B, max_len]
    full_returns = full_advantages + values  # [B, max_len]
    return full_advantages, full_returns


def chunked_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lambd: float,
    chunk_size: int = 128,
):
    """
    Compute Generalized Advantage Estimation (GAE) using a FlashLinearAttention-
    inspired algorithm: parallel prefix scan within chunks and recurrent state
    propagation across chunks.

    This reduces the sequential dependency length from O(T) to O(T / chunk_size),
    while keeping chunk computations fully parallelizable (O(C^2) per chunk).

    Args:
        rewards (Tensor): [B, T] reward sequence.
        values (Tensor):  [B, T] value predictions. The next-value of the final
                          step is assumed to be zero (standard PPO convention).
        gamma (float): discount factor.
        lam (float): GAE lambda.
        chunk_size (int): sequence chunk length for parallel scan.

    Returns:
        advantages (Tensor): [B, T] computed advantages.
        returns (Tensor):    [B, T] advantages + values.
    """

    # -------------------------------------------------------------------------
    # Validate inputs
    # -------------------------------------------------------------------------
    assert rewards.ndim == 2 and values.ndim == 2
    B, T = rewards.shape
    assert values.shape == (B, T)

    device = rewards.device
    dtype = rewards.dtype

    # -------------------------------------------------------------------------
    # Build δ_t = r_t + γ * V_{t+1} - V_t   with V_{T} = 0
    # -------------------------------------------------------------------------
    next_values = torch.cat(
        [values[:, 1:], torch.zeros(B, 1, device=device, dtype=dtype)],
        dim=1,
    )
    deltas = rewards + gamma * next_values - values

    # Reformulate backward GAE as a forward scan on the reversed sequence:
    #   S[i] = Δ[i] + w * S[i - 1],   w = γλ
    w = gamma * lambd
    deltas_rev = torch.flip(deltas, dims=[1])  # [B, T]

    # -------------------------------------------------------------------------
    # Pad to a multiple of chunk_size
    # -------------------------------------------------------------------------
    if T % chunk_size != 0:
        pad = chunk_size - (T % chunk_size)
        deltas_rev = F.pad(deltas_rev, (0, pad))
    else:
        pad = 0

    B, T_pad = deltas_rev.shape
    n_chunks = T_pad // chunk_size

    deltas_chunks = deltas_rev.view(B, n_chunks, chunk_size)

    # -------------------------------------------------------------------------
    # Construct the intra-chunk parallel scan kernel M
    #
    # For a chunk Δ[0..C-1], we want:
    #   S_local[t] = sum_{k=0..t} w^(t-k) * Δ[k]
    #
    # This is implemented as:
    #   S_local = Δ @ M
    #
    # where:
    #   M[i, j] = w^(j - i)    if j >= i
    #             0            otherwise
    # -------------------------------------------------------------------------
    idx = torch.arange(chunk_size, device=device)
    row = idx[:, None]
    col = idx[None, :]
    diff = col - row

    M = torch.zeros(chunk_size, chunk_size, device=device, dtype=dtype)
    mask = diff >= 0

    if w == 0.0:
        M[mask & (diff == 0)] = 1.0
    else:
        M[mask] = w ** diff[mask].to(dtype)

    # pow_vec[t] = w^(t+1), used to inject the recurrent state s_prev
    if w == 0.0:
        pow_vec = torch.zeros(chunk_size, device=device, dtype=dtype)
    else:
        pow_vec = w ** torch.arange(1, chunk_size + 1, device=device, dtype=dtype)

    # -------------------------------------------------------------------------
    # Parallel compute local chunk results (assuming initial state = 0)
    # -------------------------------------------------------------------------
    deltas_flat = deltas_chunks.reshape(B * n_chunks, chunk_size)
    S_local_flat = deltas_flat @ M
    S_local_chunks = S_local_flat.view(B, n_chunks, chunk_size)

    # Effective length of each chunk (the last chunk may be padded)
    lengths = [chunk_size] * n_chunks
    if pad > 0:
        lengths[-1] = chunk_size - pad

    # -------------------------------------------------------------------------
    # Recurrent propagation between chunks
    #
    # Each chunk contributes:
    #   S_global[t] = S_local[t] + w^(t+1) * s_prev
    #
    # And updates:
    #   s_prev = S_global[last_t]
    # -------------------------------------------------------------------------
    S_rev = deltas_rev.new_zeros(B, T_pad)
    s_prev = torch.zeros(B, device=device, dtype=dtype)

    for c in range(n_chunks):
        Lc = lengths[c]
        start = c * chunk_size
        end = start + Lc

        S_local = S_local_chunks[:, c, :Lc]
        S_global = S_local + s_prev.unsqueeze(1) * pow_vec[:Lc]

        S_rev[:, start:end] = S_global
        s_prev = S_global[:, -1]  # state for next chunk

    # Remove padding and flip back to original time order
    if pad > 0:
        S_rev = S_rev[:, :T]

    advantages = torch.flip(S_rev, dims=[1])
    returns = advantages + values

    return advantages, returns


def calculate_log_probs_and_entropy(logits, tokens, tp_group, with_entropy: bool = False, chunk_size: int = -1):
    logits = logits.contiguous()
    # TODO: not sure why we need to clone the logits here.
    # Without the clone, the backward will trigger inplace edit error.
    # It seems that the function with tp will modify the logits inplace.
    entropy = None
    if logits.size(0) != 0:
        if chunk_size > 0:
            num_chunks = (logits.size(0) - 1) // chunk_size + 1
            tokens_chunks = tokens.chunk(num_chunks, dim=0)
            logits_chunks = logits.chunk(num_chunks, dim=0)
            log_probs = []
            for tokens_chunk, logits_chunk in zip(tokens_chunks, logits_chunks, strict=True):
                log_prob = compute_log_probs(logits_chunk.clone(), tokens_chunk, tp_group)
                log_probs.append(log_prob)
            log_prob = torch.cat(log_probs, dim=0)
            if with_entropy:
                entropys = []
                for _, logits_chunk in zip(tokens_chunks, logits_chunks, strict=True):
                    entropy = compute_entropy_from_logits(logits_chunk.clone(), tp_group)
                    entropys.append(entropy)
                entropy = torch.cat(entropys, dim=0)
        else:
            log_prob = compute_log_probs(logits.clone(), tokens, tp_group)
            if with_entropy:
                entropy = compute_entropy_from_logits(logits.clone(), tp_group)
    else:
        log_prob = logits.new_zeros((0,))
        if with_entropy:
            entropy = logits.new_zeros((0,))

    return log_prob, entropy


def compute_advantage_stats(
    advantages: list[torch.Tensor], loss_masks: list[torch.Tensor], prefix: str
) -> dict[str, float]:
    """Compute distribution statistics over masked advantages. Returns plain scalars."""
    all_adv = torch.cat(advantages)
    all_mask = torch.cat(loss_masks).bool()
    masked = all_adv[all_mask].float()

    if masked.numel() == 0:
        return {}

    abs_masked = masked.abs()
    stats = {
        f"{prefix}/mean": masked.mean().item(),
        f"{prefix}/std": masked.std().item(),
        f"{prefix}/min": masked.min().item(),
        f"{prefix}/max": masked.max().item(),
        f"{prefix}/positive_frac": (masked > 0).float().mean().item(),
        f"{prefix}/neg_mean": masked[masked < 0].mean().item() if (masked < 0).any() else 0.0,
        f"{prefix}/pos_mean": masked[masked > 0].mean().item() if (masked > 0).any() else 0.0,
    }
    # Percentiles
    for p in [1, 5, 25, 50, 75, 95, 99]:
        stats[f"{prefix}/p{p:02d}"] = torch.quantile(masked, p / 100.0).item()
    # Tail concentration
    total_abs = abs_masked.sum()
    if total_abs > 0:
        sorted_abs, _ = abs_masked.sort(descending=True)
        n = len(sorted_abs)
        for pct in [1, 5, 10]:
            k = max(1, int(n * pct / 100))
            stats[f"{prefix}/top{pct}pct_conc"] = (sorted_abs[:k].sum() / total_abs).item()

    return stats


def apply_advantage_shaping(
    method: str,
    advantages: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    clip_percentile: float = 0.95,
    raw_rewards: list[float] | None = None,
) -> list[torch.Tensor]:
    """Apply shaping transform to per-sample advantage tensors in-place-style (returns new list)."""
    if method == "raw":
        return advantages

    if method == "correct_positive_only":
        if raw_rewards is None:
            raise ValueError("correct_positive_only requires raw_rewards")
        return [
            adv.clamp(min=0) if float(reward) == 1.0 else adv
            for adv, reward in zip(advantages, raw_rewards, strict=False)
        ]

    if method == "wrong_positive_only":
        if raw_rewards is None:
            raise ValueError("wrong_positive_only requires raw_rewards")
        return [
            adv.clamp(min=0) if float(reward) == 0.0 else adv
            for adv, reward in zip(advantages, raw_rewards, strict=False)
        ]

    if method == "correct_positive_only_strict":
        if raw_rewards is None:
            raise ValueError("correct_positive_only_strict requires raw_rewards")
        return [
            adv.clamp(min=0) if float(reward) == 1.0 else torch.zeros_like(adv)
            for adv, reward in zip(advantages, raw_rewards, strict=False)
        ]

    if method == "wrong_positive_only_strict":
        if raw_rewards is None:
            raise ValueError("wrong_positive_only_strict requires raw_rewards")
        return [
            adv.clamp(min=0) if float(reward) == 0.0 else torch.zeros_like(adv)
            for adv, reward in zip(advantages, raw_rewards, strict=False)
        ]

    if method in ("clip", "clip_center"):
        # Compute clip bound from masked advantages
        all_adv = torch.cat(advantages)
        all_mask = torch.cat(loss_masks).bool()
        masked_abs = all_adv[all_mask].abs()
        c = torch.quantile(masked_abs.float(), clip_percentile).item()
        advantages = [adv.clamp(-c, c) for adv in advantages]

    if method in ("center", "clip_center"):
        # Per-response centering
        shaped = []
        for adv, mask in zip(advantages, loss_masks, strict=False):
            m = mask.bool()
            if m.any():
                mean_val = adv[m].mean()
                adv = adv - mean_val
            shaped.append(adv)
        advantages = shaped

    if method == "positive_only":
        advantages = [adv.clamp(min=0) for adv in advantages]

    return advantages
