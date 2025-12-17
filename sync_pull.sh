#!/bin/bash

# 快速拉取最新代码脚本

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}📥 开始从 GitHub 拉取最新代码...${NC}"

# 检查是否在 git 仓库中
if [ ! -d .git ]; then
    echo -e "${RED}❌ 错误：当前目录不是 Git 仓库！${NC}"
    echo -e "${YELLOW}💡 请先按照 Git同步说明.md 中的步骤克隆仓库${NC}"
    exit 1
fi

# 检查是否有未提交的修改
if ! git diff-index --quiet HEAD --; then
    echo -e "${YELLOW}⚠️  警告：你有未提交的本地修改${NC}"
    echo -e "${YELLOW}📝 未提交的文件：${NC}"
    git status --short
    echo ""
    read -p "是否先提交这些修改？(y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # 先提交本地修改
        read -p "请输入提交信息: " commit_msg
        git add .
        git commit -m "$commit_msg"
        echo -e "${GREEN}✅ 本地修改已提交${NC}"
    fi
fi

# 拉取远程代码
echo -e "${YELLOW}⬇️  拉取远程代码...${NC}"
if git pull; then
    echo -e "${GREEN}✅ 代码已成功更新！${NC}"

    # 显示最近的提交
    echo -e "${YELLOW}📋 最近的提交记录：${NC}"
    git log --oneline -5
else
    echo -e "${RED}❌ 拉取失败！请检查：${NC}"
    echo -e "${YELLOW}  1. 网络连接是否正常${NC}"
    echo -e "${YELLOW}  2. 是否有需要手动解决的冲突${NC}"
    echo -e "${YELLOW}  3. GitHub 仓库地址是否正确${NC}"
    exit 1
fi
