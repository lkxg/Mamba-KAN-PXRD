#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_DIR="${1:-experiments/speed_tests}"
LOG_DIR="$OUT_DIR/logs"
SUMMARY="$OUT_DIR/summary.tsv"
mkdir -p "$LOG_DIR"

CONFIGS=(
  configs/experiments/speed_m53_l0_frontend_only.yaml
  configs/experiments/speed_m53_l1_mamba_ssm.yaml
  configs/experiments/speed_m53_l4_mamba_ssm.yaml
  configs/experiments/speed_m53_l8_mamba_ssm.yaml
  configs/experiments/speed_m53_l8_mamba2.yaml
  configs/experiments/speed_m53_batch256_l8_mamba_ssm.yaml
  configs/experiments/speed_learned_stride8_l8_mamba_ssm.yaml
  configs/experiments/speed_learned_stride16_l8_mamba_ssm.yaml
  configs/experiments/speed_cpuinput_m53_l8_workers0.yaml
  configs/experiments/speed_cpuinput_m53_l8_workers4.yaml
  configs/experiments/speed_cpuinput_m53_l8_workers24.yaml
  configs/experiments/speed_cpuinput_m53_l0_workers0.yaml
)

printf 'experiment\tstatus\tepoch_sec\tbackend\tparams_m\tbatch_size\tnum_workers\tpin_memory\tconfig\tlog\n' > "$SUMMARY"

for config in "${CONFIGS[@]}"; do
  name="$(python3 - "$config" <<'PY'
import sys
from src.utils import load_config
print(load_config(sys.argv[1])["experiment"]["name"])
PY
)"
  log="$LOG_DIR/${name}.log"
  echo
  echo "=== ${name}: ${config} ==="
  started="$(date -Is)"
  echo "START ${started}" > "$log"
  set +e
  python3 scripts/train.py --config "$config" 2>&1 | tee -a "$log"
  rc=${PIPESTATUS[0]}
  set -e
  ended="$(date -Is)"
  echo "END ${ended} status=${rc}" | tee -a "$log"

  python3 - "$config" "$log" "$rc" "$SUMMARY" <<'PY'
import re
import sys
from pathlib import Path
from src.utils import load_config

config_path = Path(sys.argv[1])
log_path = Path(sys.argv[2])
rc = int(sys.argv[3])
summary = Path(sys.argv[4])
cfg = load_config(config_path)
text = log_path.read_text(encoding="utf-8", errors="replace")

epoch_sec = ""
match = re.search(r"epoch\s+\d+/\d+.*\((\d+)s\)", text)
if match:
    epoch_sec = match.group(1)
backend = ""
match = re.search(r"WA mixer backend:\s+(\S+)", text)
if match:
    backend = match.group(1)
params_m = ""
match = re.search(r"Model:\s+\S+\s+\(([^)]+) params\)", text)
if match:
    params_m = match.group(1).replace("\t", " ")

row = [
    cfg["experiment"]["name"],
    "ok" if rc == 0 else f"fail:{rc}",
    epoch_sec,
    backend,
    params_m,
    str(cfg["data"]["batch_size"]),
    str(cfg["data"]["num_workers"]),
    str(cfg["data"]["pin_memory"]),
    str(config_path),
    str(log_path),
]
with summary.open("a", encoding="utf-8") as f:
    f.write("\t".join(row) + "\n")
PY

  if [[ "$rc" -ne 0 ]]; then
    echo "FAILED ${name}; continuing to next config" >&2
  fi
done

echo
echo "Summary written to $SUMMARY"
cat "$SUMMARY"
