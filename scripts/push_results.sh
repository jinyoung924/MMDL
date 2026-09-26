#!/usr/bin/env bash
# Commit and push ONE result directory into results/ right away. Called at the end of
# scripts/run_mmmu_eval.sh (PUSH_EACH=1, default) so a pod that dies later cannot lose finished runs.
# Needs GITHUB_TOKEN (env or PID 1 env, i.e. a RunPod secret); silently does nothing without it.
set -uo pipefail
OUT_DIR="$(cd "$1" && pwd)"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_SLUG="${REPO_SLUG:-jinyoung924/MMDL}"
OUT_ROOT="${OUT_ROOT:-/workspace/results}"
TOKEN="${GITHUB_TOKEN:-$(tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n 's/^GITHUB_TOKEN=//p' | head -1)}"
[[ -z "$TOKEN" ]] && { echo "push_results: no GITHUB_TOKEN -> skipped"; exit 0; }

case "$OUT_DIR" in
  "$REPO_DIR"/results/*) REL="${OUT_DIR#"$REPO_DIR"/results/}" ;;
  "$OUT_ROOT"/*)         REL="${OUT_DIR#"$OUT_ROOT"/}" ;;
  *)                     REL="$(basename "$OUT_DIR")" ;;
esac
DEST="$REPO_DIR/results/$REL"
if [[ "$DEST" != "$OUT_DIR" ]]; then
  mkdir -p "$(dirname "$DEST")" && rm -rf "$DEST" && cp -r "$OUT_DIR" "$DEST"
fi
cd "$REPO_DIR"
git config user.name  "${GIT_AUTHOR_NAME:-${REPO_SLUG%%/*}}"
git config user.email "${GIT_AUTHOR_EMAIL:-${REPO_SLUG%%/*}@users.noreply.github.com}"
git add "results/$REL" 2>/dev/null || { echo "push_results: results/$REL is ignored or missing -> skipped"; exit 0; }
git commit -q -m "results: $REL ($(hostname) $(date -u +%Y-%m-%dT%H:%MZ), pushed from pod)" || { echo "push_results: nothing to commit"; exit 0; }
URL="https://x-access-token:${TOKEN}@github.com/${REPO_SLUG}.git"
for i in 1 2 3 4 5; do
  git push -q "$URL" HEAD:main && { echo "push_results: pushed results/$REL"; exit 0; }
  sleep 15; git pull -q --rebase "$URL" main || true
done
echo "push_results: PUSH FAILED for results/$REL"
