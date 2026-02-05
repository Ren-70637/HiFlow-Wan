#!/usr/bin/env bash
set -euo pipefail

<<<<<<< HEAD
# ====== Git Push 脚本：服务器项目同步到自己的 GitHub ======
REPO_DIR="/home/rentianhao-20251020/DiffSynth-Studio"
REMOTE_NAME="mygithub"
REMOTE_URL="git@github.com:Ren-70637/HiFlow-Wan.git"
COMMIT_MSG="${1:-Update: sync changes ($(date +'%Y-%m-%d %H:%M:%S'))}"
=======
# ====== 配置区域 ======
# 1. 你服务器上的本地项目路径
REPO_DIR="/mnt/users/rentianhao-20251020/projects/DiffSynth-Studio"

# 2. 你的 GitHub 仓库 SSH 地址
# (注意：根据你的描述，本地文件夹是 DiffSynth-Studio，但你要推送到 HiFlow-Wan 仓库)
MY_GITHUB_SSH="git@github.com:Ren-70637/HiFlow-Wan.git"
# ====================

# 检查目录是否存在
if [ ! -d "$REPO_DIR" ]; then
    echo "Error: 找不到目录 $REPO_DIR"
    exit 1
fi
>>>>>>> 433fc2c (Update: sync latest changes from server)

cd "$REPO_DIR" || exit 1

echo "== [1] Repo info =="
git rev-parse --is-inside-work-tree >/dev/null
BRANCH="$(git branch --show-current || true)"
if [[ -z "$BRANCH" ]]; then
  echo "ERROR: Detached HEAD（当前不在任何分支）"
  echo "Fix: git switch -c <new-branch>"
  exit 2
fi
echo "Current branch: $BRANCH"
git remote -v
git status -sb

echo "== [2] Fetch remotes =="
git fetch --all --prune || true

echo "== [3] What changed? =="
git diff --name-status || true
git diff --stat || true
git log --oneline --decorate --graph --max-count=20

echo "== [4] Commit local changes (if any) =="
git add -A
if git diff --cached --quiet; then
  echo "No changes to commit."
else
  git commit -m "$COMMIT_MSG"
fi

echo "== [5] Ensure remote '$REMOTE_NAME' points to your GitHub =="
if git remote get-url "$REMOTE_NAME" >/dev/null 2>&1; then
  git remote set-url "$REMOTE_NAME" "$REMOTE_URL"
else
  git remote add "$REMOTE_NAME" "$REMOTE_URL"
fi
git remote -v

echo "== [6] Push =="
git push -u "$REMOTE_NAME" "$BRANCH"

echo "DONE: pushed $BRANCH to $REMOTE_NAME ($REMOTE_URL)"
