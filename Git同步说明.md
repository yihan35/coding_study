# 代码同步说明

## 📋 目录结构
```
coding/
├── llm_code/    # 大模型学习代码
├── lee_code/    # 力扣刷题代码
└── Git同步说明.md
```

## 🚀 首次设置（只需执行一次）

### 1. 初始化 Git 仓库并推送到 GitHub

在**当前电脑**上执行以下操作：

```bash
# 进入 coding 目录
cd /Users/liyihan12/coding

# 初始化 Git 仓库
git init

# 创建 .gitignore 文件（忽略不需要同步的文件）
cat > .gitignore << 'EOF'
# macOS 系统文件
.DS_Store
.AppleDouble
.LSOverride

# IDE 配置（根据需要选择是否忽略）
.vscode/
.idea/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
*.egg-info/

# 其他
.env
*.log
EOF

# 添加所有文件
git add .

# 第一次提交
git commit -m "初始提交：添加大模型和力扣学习代码"

# 连接到 GitHub 仓库（需要先在 GitHub 创建仓库）
# 将下面的 YOUR_USERNAME 和 YOUR_REPO_NAME 替换为你的 GitHub 用户名和仓库名
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git remote add origin https://github.com/yihan35/coding_study.git
# 推送到 GitHub
git branch -M main
git push -u origin main
```

### 2. 在另一台电脑上克隆仓库

在**另一台电脑**上执行：

```bash
# 进入你想存放代码的目录
cd ~

# 克隆仓库（替换为你的仓库地址）
git clone https://github.com/yihan35/coding_study.git coding

# 进入目录
cd coding
```

## 💾 日常使用

### 方式一：使用快捷脚本（推荐）

直接运行提供的脚本：

```bash
# 提交并推送代码
./sync_push.sh

# 或者带自定义提交信息
./sync_push.sh "完成了二叉树的题目"

# 拉取最新代码
./sync_pull.sh
```

### 方式二：手动执行 Git 命令

**推送代码到 GitHub：**

```bash
cd /Users/liyihan12/coding

# 查看修改的文件
git status

# 添加所有修改
git add .

# 提交修改
git commit -m "描述你的修改内容"

# 推送到 GitHub
git push
```

**从 GitHub 拉取代码：**

```bash
cd /Users/liyihan12/coding

# 拉取最新代码
git pull
```

## 🔄 工作流程示例

### 在公司电脑上工作：
1. 开始工作前：`./sync_pull.sh` （拉取最新代码）
2. 编写代码...
3. 工作结束后：`./sync_push.sh "今天完成了 XXX"` （推送代码）

### 回到家里电脑：

1. 开始工作前：`./sync_pull.sh` （拉取最新代码）
2. 编写代码...
3. 工作结束后：`./sync_push.sh "完成了 XXX 功能"` （推送代码）

## ⚠️ 注意事项

1. **每次开始工作前先拉取代码**，避免冲突
2. **每次工作结束后及时推送代码**，保持同步
3. 如果遇到冲突，需要手动解决：
   ```bash
   # 查看冲突文件
   git status
   
   # 编辑冲突文件，解决冲突
   # 然后执行
   git add .
   git commit -m "解决冲突"
   git push
   ```

## 📝 常用命令速查

| 命令 | 说明 |
|------|------|
| `git status` | 查看当前状态 |
| `git log` | 查看提交历史 |
| `git diff` | 查看修改内容 |
| `git pull` | 拉取远程代码 |
| `git push` | 推送本地代码 |
| `git add .` | 添加所有修改 |
| `git commit -m "xxx"` | 提交修改 |

## 🔧 故障排除

### 推送失败
- 检查网络连接
- 确保已经先 `git pull` 拉取了最新代码
- 检查 GitHub 账号权限

### 拉取冲突
- 先提交本地修改：`git add . && git commit -m "保存本地修改"`
- 再拉取：`git pull`
- 如有冲突，手动解决后提交

### SSH 密钥设置（可选，避免每次输入密码）
```bash
# 生成 SSH 密钥
ssh-keygen -t ed25519 -C "your_email@example.com"

# 复制公钥到 GitHub
cat ~/.ssh/id_ed25519.pub

# 然后修改远程仓库地址为 SSH 格式
git remote set-url origin git@github.com:YOUR_USERNAME/YOUR_REPO_NAME.git
```
