#!/bin/bash

# ====== 配置区域 ======
# 1. 你服务器上的本地项目路径
REPO_DIR="/home/rentianhao-20251020/DiffSynth-Studio"

# 2. 你的 GitHub 仓库 SSH 地址
# (注意：根据你的描述，本地文件夹是 DiffSynth-Studio，但你要推送到 HiFlow-Wan 仓库)
MY_GITHUB_SSH="git@github.com:Ren-70637/HiFlow-Wan.git"
# ====================

# 检查目录是否存在
if [ ! -d "$REPO_DIR" ]; then
    echo "Error: 找不到目录 $REPO_DIR"
    exit 1
fi

cd "$REPO_DIR" || exit 1
echo "🚀 进入目录: $(pwd)"

echo -e "\n==[1] 检查当前状态 =="
git branch --show-current
git status -sb

echo -e "\n==[2] 拉取最新信息 (Fetch) =="
git fetch --all --prune || true

echo -e "\n==[3] 查看变更 (Diff) =="
# 简单列出哪些文件变了
git diff --name-status HEAD

echo -e "\n==[4] 提交本地修改 (Commit) =="
# 添加所有文件 (注意：请确保你的 .gitignore 已经配置好，忽略了视频和大模型权重)
git add -A
# 尝试提交，如果没有新修改则忽略错误继续运行
git commit -m "Update: sync latest changes from server" || echo "⚠️ 没有新文件需要提交，继续..."

echo -e "\n==[5] 配置远程仓库 (Smart Remote Config) =="
# 逻辑：
# 1. 如果当前的 origin 已经是你的 HiFlow-Wan，则不用动。
# 2. 如果当前的 origin 是别人的库，把它重命名为 upstream，把 origin 设为你自己的。

CURRENT_ORIGIN=$(git remote get-url origin 2>/dev/null)

if [ "$CURRENT_ORIGIN" == "$MY_GITHUB_SSH" ]; then
    echo "✅ origin 已经是你的仓库地址，无需修改。"
else
    echo "🔄 正在重新配置 origin..."
    # 如果 origin 存在，先把它重命名为 upstream (备份原作者的源)
    if [ ! -z "$CURRENT_ORIGIN" ]; then
        # 如果 upstream 已经存在，先删掉旧的 upstream，防止重命名冲突
        git remote remove upstream 2>/dev/null
        git remote rename origin upstream
        echo "   已将原 origin 备份为 upstream"
    fi
    # 添加新的 origin
    git remote add origin "$MY_GITHUB_SSH"
    echo "✅ 已将 origin 设置为: $MY_GITHUB_SSH"
fi

echo -e "\n==[6] 推送到 GitHub (Push) =="
BRANCH="$(git branch --show-current)"
if [ -z "$BRANCH" ]; then
    echo "❌ 错误：当前处于 'Detached HEAD' 游离状态，无法推送。"
    echo "建议先创建分支：git switch -c main"
    exit 1
fi

echo "正在推送到 origin/$BRANCH ..."
git push -u origin "$BRANCH"

echo -e "\n🎉 完成！"
# ====== End ======