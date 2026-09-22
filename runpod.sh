#!/usr/bin/env bash
# ============================================================================
# Single SSH entry point.  On a fresh RunPod pod:
#
#   git clone https://github.com/jinyoung924/MMDL.git /workspace/MMDL && bash /workspace/MMDL/runpod.sh
#
# Re-running after a disconnect resumes where it stopped (per-subject checkpoints).
# Knobs (env vars):  SKIP_SMOKE=1  SKIP_FULL=1  RUN_ABLATIONS=1  OUT_ROOT=/workspace/results
#
# Unattended mode (set these as RunPod Secrets and reference them in the pod template's env):
#   GITHUB_TOKEN         fine-grained PAT, Contents: read/write on this repo  -> results are committed & pushed
#   RUNPOD_USER_API_KEY  (or MY_RUNPOD_USER_API_KEY) RunPod API key, Settings > API Keys -> pod terminates itself
#   AUTO_TERMINATE=0     keep the pod alive even when the key is present
# Nothing secret is ever written into the repo; tokens are read from the environment only.
# ============================================================================
set -euo pipefail
REPO_SLUG="${REPO_SLUG:-jinyoung924/MMDL}"
# SSH sessions on RunPod do NOT inherit the container's env (secrets, RUNPOD_POD_ID), so fall back to
# PID 1's environment for anything missing. Accept either name for the RunPod user key.
_pid1_env() { tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n "s/^$1=//p" | head -1; }
for v in GITHUB_TOKEN RUNPOD_USER_API_KEY MY_RUNPOD_USER_API_KEY RUNPOD_POD_ID; do
  [[ -z "${!v:-}" ]] && declare "$v=$(_pid1_env "$v")"
done
RUNPOD_USER_API_KEY="${RUNPOD_USER_API_KEY:-${MY_RUNPOD_USER_API_KEY:-}}"

# ---- unattended wrap-up: always runs, even when a step above failed ----------------------
wrap_up() {
  local rc=$?
  trap - EXIT
  echo "== wrap-up (exit code of main flow: $rc) =="
  if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    cd "$REPO_DIR"
    git config user.name "${GIT_AUTHOR_NAME:-${REPO_SLUG%%/*}}"
    git config user.email "${GIT_AUTHOR_EMAIL:-${REPO_SLUG%%/*}@users.noreply.github.com}"
    local url="https://x-access-token:${GITHUB_TOKEN}@github.com/${REPO_SLUG}.git"
    mkdir -p results
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) pod=${RUNPOD_POD_ID:-?} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) rc=$rc smoke=${SKIP_SMOKE:-0} full=${SKIP_FULL:-0} ablations=${RUN_ABLATIONS:-0}" >> results/pod_runs.log
    git add results || true
    if git commit -q -m "results: $(hostname) $(date -u +%Y-%m-%dT%H:%MZ) (pushed from pod, main rc=$rc)"; then
      local pushed=0
      for i in 1 2 3 4 5; do
        git push -q "$url" HEAD:main && { pushed=1; break; }
        sleep 20; git pull -q --rebase "$url" main || true
      done
      echo "pushed=$pushed"
    else
      echo "nothing new to commit"
    fi
  else
    echo "GITHUB_TOKEN not set -> results stay on the pod (copy them out before terminating!)"
  fi
  if [[ -n "${RUNPOD_USER_API_KEY:-}" && "${AUTO_TERMINATE:-1}" == "1" && -n "${RUNPOD_POD_ID:-}" ]]; then
    echo "terminating pod $RUNPOD_POD_ID in 60 s (Ctrl-C to keep it)"; sleep 60
    runpodctl config --apiKey "$RUNPOD_USER_API_KEY" >/dev/null 2>&1 || true
    runpodctl remove pod "$RUNPOD_POD_ID" || echo "TERMINATE FAILED - remove the pod from the console"
  else
    echo "auto-terminate skipped (RUNPOD_USER_API_KEY unset or AUTO_TERMINATE=0)"
  fi
}
trap wrap_up EXIT
REPO_DIR="${REPO_DIR:-/workspace/MMDL}"
OUT_ROOT="${OUT_ROOT:-/workspace/results}"
cd "$REPO_DIR"
git pull --ff-only || true

bash scripts/setup_runpod.sh

if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
  echo "== smoke test (Art, 3 questions) =="
  OUT_DIR="$OUT_ROOT/smoke" bash scripts/run_mmmu_eval.sh --subjects Art --limit 3 --no_resume
fi

if [[ "${SKIP_FULL:-0}" != "1" ]]; then
  echo "== full run: 30 subjects x 30 questions (config defaults: mmmu_pro_cot prompt, 8192 tokens) =="
  OUT_DIR="$OUT_ROOT/mmmu_baseline" bash scripts/run_mmmu_eval.sh
  rm -rf results/mmmu_baseline && cp -r "$OUT_ROOT/mmmu_baseline" results/mmmu_baseline
fi

if [[ "${RUN_ABLATIONS:-0}" == "1" ]]; then
  echo "== ablations: official MMMU direct-answer prompt at 1024 and 8192 tokens (same recipe/seed/parser) =="
  OUT_DIR="$OUT_ROOT/ablation_direct_1k" bash scripts/run_mmmu_eval.sh --prompt_style mmmu_direct --max_new_tokens 1024
  OUT_DIR="$OUT_ROOT/ablation_direct_8k" bash scripts/run_mmmu_eval.sh --prompt_style mmmu_direct --max_new_tokens 8192
  for d in ablation_direct_1k ablation_direct_8k; do rm -rf "results/$d" && cp -r "$OUT_ROOT/$d" "results/$d"; done
fi

echo "== done. results are in $REPO_DIR/results =="
