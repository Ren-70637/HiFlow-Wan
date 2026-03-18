#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-origin}"
TARGET_BRANCH="${2:-${TARGET_BRANCH:-}}"
COMMIT_MESSAGE="${1:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
CURRENT_REMOTE_URL="$(git -C "$REPO_ROOT" remote get-url "$REMOTE")"
CURRENT_BRANCH="$(git -C "$REPO_ROOT" branch --show-current)"

if [[ -z "$CURRENT_BRANCH" ]]; then
  echo "Error: failed to detect the current branch." >&2
  exit 1
fi

if [[ -z "$TARGET_BRANCH" ]]; then
  TARGET_BRANCH="$CURRENT_BRANCH"
fi

if [[ -z "$COMMIT_MESSAGE" ]]; then
  COMMIT_MESSAGE="sync: update HiFlow-Wan on $(date '+%Y-%m-%d %H:%M:%S %Z')"
fi

if [[ "$CURRENT_REMOTE_URL" =~ ^https://github\.com/([^/]+)/([^/]+)\.git$ ]]; then
  SSH_REMOTE_URL="git@github.com:${BASH_REMATCH[1]}/${BASH_REMATCH[2]}.git"
  echo "Remote $REMOTE is HTTPS. Switching to SSH:"
  echo "  $CURRENT_REMOTE_URL"
  echo "  -> $SSH_REMOTE_URL"
  git -C "$REPO_ROOT" remote set-url "$REMOTE" "$SSH_REMOTE_URL"
  CURRENT_REMOTE_URL="$SSH_REMOTE_URL"
fi

echo "Repo           : $REPO_ROOT"
echo "Remote         : $REMOTE"
echo "URL            : $CURRENT_REMOTE_URL"
echo "Current branch : $CURRENT_BRANCH"
echo "Target branch  : $TARGET_BRANCH"
echo "Message        : $COMMIT_MESSAGE"
echo
echo "Current status:"
git -C "$REPO_ROOT" status --short
echo

git -C "$REPO_ROOT" add -A

if ! git -C "$REPO_ROOT" diff --cached --quiet; then
  git -C "$REPO_ROOT" commit -m "$COMMIT_MESSAGE"
else
  echo "No staged changes to commit."
fi

echo
echo "Current HEAD: $(git -C "$REPO_ROOT" rev-parse --short HEAD)"

if [[ "$CURRENT_BRANCH" == "$TARGET_BRANCH" ]]; then
  PUSH_REF="$TARGET_BRANCH"
else
  PUSH_REF="HEAD:$TARGET_BRANCH"
fi

if git -C "$REPO_ROOT" push "$REMOTE" "$PUSH_REF"; then
  echo
  echo "Push completed: $REMOTE/$TARGET_BRANCH"
else
  echo
  echo "Push failed."
  echo "The local commit is preserved. Check GitHub authentication for remote $REMOTE."
  echo "Current HEAD: $(git -C "$REPO_ROOT" rev-parse --short HEAD)"
  exit 1
fi
