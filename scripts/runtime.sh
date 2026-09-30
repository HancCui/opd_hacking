#!/usr/bin/env bash

# Every relative filesystem setting is anchored at the project root.
resolve_project_paths() {
   local key value
   for key in "$@"; do
      value="${!key:-}"
      [ -n "${value}" ] || continue
      case "${value}" in
         /*) ;;
         *) value="${ROOT_DIR}/${value}" ;;
      esac
      printf -v "$key" '%s' "$(realpath -m -- "${value}")"
   done
}

find_free_port() {
   python3 - "$1" <<'PY'
import socket
import sys

start_port = int(sys.argv[1])
for port in range(start_port, 65535):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("", port))
        except OSError:
            continue
        print(port)
        break
else:
    raise SystemExit(f"No free port found from {start_port}")
PY
}

wait_for_http_health() {
   local url="$1"
   local timeout="$2"
   python3 - "$url" "$timeout" <<'PY'
import sys
import time
import urllib.error
import urllib.request

url = sys.argv[1]
deadline = time.time() + int(sys.argv[2])
last_error = None
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            if response.status < 500:
                raise SystemExit(0)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        last_error = exc
    time.sleep(2)

raise SystemExit(f"Timed out waiting for {url}: {last_error}")
PY
}

get_ray_job_status() {
   local address="$1"
   local job_id="$2"
   python3 - "$address" "$job_id" <<'PY'
import sys
from ray.dashboard.modules.job.sdk import JobSubmissionClient

client = JobSubmissionClient(sys.argv[1])
info = client.get_job_info(sys.argv[2])
status = info.status
print(status.value if hasattr(status, "value") else status)
PY
}

build_runtime_env_json() {
   python3 - <<'PY'
import json
import os

env_vars = {
    "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
    "CUDA_DEVICE_MAX_CONNECTIONS": os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", "1"),
    "MASTER_ADDR": os.environ.get("MASTER_ADDR", "127.0.0.1"),
    "PYTHONUNBUFFERED": os.environ.get("PYTHONUNBUFFERED", "1"),
}

for key in (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
    "WANDB_BASE_URL",
    "WANDB_MODE",
    "WANDB_DIR",
    "WANDB_API_KEY",
    "WANDB_HOST",
    "WANDB_PROJECT",
    "WANDB_TEAM",
    "WANDB_ENTITY",
    "SLIME_HOST_IP",
    "TEACHER_TEMPERATURE",
    "SLIME_UPSTREAM_DIR",
    "MEGATRON_PATH",
    "SGLANG_PATH",
    "OPD_STUDENT_EOS_TOKEN_ID",
    "OPD_TEACHER_EOS_TOKEN_ID",
    "ROLLOUT_SAMPLE_FILTER_PATH",
    "DYNAMIC_SAMPLING_SAMPLE_FILTER_PATH",
    "DYNAMIC_SAMPLING_MAX_GENERATED_SAMPLES",
    # JIT/compile caches. The Ray workers inherit the launcher's environment,
    # but forward these explicitly so a worker cannot end up writing to a
    # different cache root than the teacher server does.
    "FLASHINFER_WORKSPACE_BASE",
    "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "VLLM_CACHE_ROOT",
):
    value = os.environ.get(key)
    if value:
        env_vars[key] = value

print(json.dumps({"env_vars": env_vars}))
PY
}

build_no_proxy() {
   python3 - "$@" <<'PY'
import os
import socket
import sys

entries = []

def add(value):
    if not value:
        return
    for item in str(value).split(","):
        item = item.strip()
        if item and item not in entries:
            entries.append(item)

for value in (
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "::1",
    *sys.argv[1:],
    os.environ.get("SLIME_HOST_IP"),
):
    add(value)

hostname = socket.gethostname()
add(hostname)
add(socket.getfqdn())

try:
    _, aliases, addrs = socket.gethostbyname_ex(hostname)
    for value in aliases + addrs:
        add(value)
except OSError:
    pass

for family, target in ((socket.AF_INET, ("8.8.8.8", 80)), (socket.AF_INET6, ("2001:4860:4860::8888", 80))):
    try:
        with socket.socket(family, socket.SOCK_DGRAM) as sock:
            sock.connect(target)
            add(sock.getsockname()[0])
    except OSError:
        pass

add(os.environ.get("no_proxy"))
add(os.environ.get("NO_PROXY"))

print(",".join(entries))
PY
}

