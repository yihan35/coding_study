#!/bin/bash

# 快速提交并推送脚本

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}📤 开始同步代码到 GitHub...${NC}"

# 检查是否在 git 仓库中
if [ ! -d .git ]; then
    echo -e "${RED}❌ 错误：当前目录不是 Git 仓库！${NC}"
    echo -e "${YELLOW}💡 请先按照 Git同步说明.md 中的步骤初始化仓库${NC}"
    exit 1
fi

# 获取提交信息
if [ -z "$1" ]; then
    # 如果没有提供参数，使用默认提交信息
    COMMIT_MSG="更新代码 - $(date '+%Y-%m-%d %H:%M:%S')"
else
    # 使用用户提供的提交信息
    COMMIT_MSG="$1"
fi

# 显示当前状态
echo -e "${YELLOW}📝 当前修改的文件：${NC}"
git status --short

# 添加所有修改
echo -e "${YELLOW}➕ 添加所有修改...${NC}"
git add .

# 检查是否有修改需要提交
if git diff --cached --quiet; then
    echo -e "${YELLOW}ℹ️  没有新的修改需要提交${NC}"
else
    # 提交修改
    echo -e "${YELLOW}💾 提交修改：${COMMIT_MSG}${NC}"
    git commit -m "$COMMIT_MSG"
fi

# 先拉取远程代码，避免分叉
echo -e "${YELLOW}⬇️  先拉取远程最新代码...${NC}"
if ! git pull --rebase; then
    echo -e "${RED}❌ 拉取失败或存在冲突！${NC}"
    echo -e "${YELLOW}💡 请手动解决冲突后再推送${NC}"
    echo -e "${YELLOW}  1. 解决冲突文件${NC}"
    echo -e "${YELLOW}  2. git add .${NC}"
    echo -e "${YELLOW}  3. git rebase --continue${NC}"
    echo -e "${YELLOW}  4. git push${NC}"
    exit 1
fi

# 推送到远程仓库
echo -e "${YELLOW}🚀 推送到 GitHub...${NC}"
if git push; then
    echo -e "${GREEN}✅ 代码已成功推送到 GitHub！${NC}"
else
    echo -e "${RED}❌ 推送失败！请检查：${NC}"
    echo -e "${YELLOW}  1. 网络连接是否正常${NC}"
    echo -e "${YELLOW}  2. GitHub 账号权限是否正确${NC}"
    exit 1
fi
