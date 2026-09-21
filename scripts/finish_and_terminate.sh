#!/usr/bin/env bash
# Unattended wrap-up for a RunPod pod (run inside tmux ON THE POD):
#   1. wait until the experiment log contains the end marker (or a hard deadline passes)
#   2. commit the result directories and push them to GitHub
#   3. terminate this pod with the pod-scoped RunPod key
#
# The GitHub token is NOT stored in this repo. The pod owner puts it in $TOKEN_FILE themselves:
#   read -s T && printf '%s' "$T" > /workspace/.ghtoken && chmod 600 /workspace/.ghtoken
#
# Env knobs: LOG_FILE, END_MARKER, RESULT_DIRS (space separated), MAX_HOURS, TOKEN_FILE, REPO_SLUG
set -uo pipefail
LOG_FILE="${LOG_FILE:-/workspace/run_abl.log}"
END_MARKER="${END_MARKER:-ABL_EXIT}"
RESULT_DIRS="${RESULT_DIRS:-/workspace/results/ablation_cot_8k}"
MAX_HOURS="${MAX_HOURS:-3}"
TOKEN_FILE="${TOKEN_FILE:-/workspace/.ghtoken}"
REPO_SLUG="${REPO_SLUG:-jinyoung924/MMDL}"
REPO_DIR="${REPO_DIR:-/workspace/MMDL}"
exec >>/workspace/finish.log 2>&1

[[ -s "$TOKEN_FILE" ]] || { echo "$(date) no token in $TOKEN_FILE -> refusing to run (results would be lost)"; exit 1; }
echo "$(date) finisher armed: waiting for '$END_MARKER' in $LOG_FILE (max ${MAX_HOURS}h)"
HARD=$(( $(date +%s) + MAX_HOURS * 3600 ))
until grep -q "$END_MARKER" "$LOG_FILE" 2>/dev/null || [[ $(date +%s) -ge $HARD ]]; do sleep 30; done
echo "$(date) experiments finished (or deadline reached)"

# Fresh clone so the long-running checkout (and the scripts bash is still reading) is never touched.
URL="https://x-access-token:$(cat "$TOKEN_FILE")@github.com/${REPO_SLUG}.git"
PUSH_DIR=/workspace/_push && rm -rf "$PUSH_DIR"
PUSHED=0
if git clone -q --depth 1 "$URL" "$PUSH_DIR"; then
  cd "$PUSH_DIR"
  git config user.name "${REPO_SLUG%%/*}"; git config user.email "${REPO_SLUG%%/*}@users.noreply.github.com"
  mkdir -p results
  for d in $RESULT_DIRS; do [[ -d "$d" ]] && rm -rf "results/$(basename "$d")" && cp -r "$d" results/; done
  git add results
  git commit -q -m "results: $(for d in $RESULT_DIRS; do basename "$d"; done | xargs) (pushed from pod before auto-terminate)" || echo "nothing to commit"
  for i in 1 2 3 4 5; do
    if git push -q "$URL" HEAD:main; then PUSHED=1; break; fi
    sleep 20; git pull -q --rebase "$URL" main
  done
fi
echo "$(date) pushed=$PUSHED"

export RUNPOD_API_KEY=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^RUNPOD_API_KEY=//p')
POD_ID=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^RUNPOD_POD_ID=//p')
runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null 2>&1
echo "$(date) terminating pod $POD_ID"
runpodctl remove pod "$POD_ID" || runpodctl stop pod "$POD_ID"
echo "$(date) terminate exit=$? (if you can read this, termination FAILED - terminate from the console)"
