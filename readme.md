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
