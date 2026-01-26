### **git **

（基于 SSH 密钥已配置，配置密钥见最下方）

#### **git push**

步骤如下：

1. 打开终端，进入你的项目文件夹

2. 初始化 Git 仓库

   > git init

3. 添加所有文件并提交

   >  git add .
   >  git commit -m "首次提交"

4. 添加远程仓库地址（使用 SSH）

   > git remote add origin ssh://git@codehub.srdcloud.cn:29418/你的用户名/你的仓库名.git
   >
   > git remote add origin ssh://git@github.com:yihan35/ai-chat.git

   给远程仓库起了个名字叫 `origin`

5. 将当前分支命名为 main 并推送到远程仓库

   >  git branch -M main
   >  git push -u origin main

以后每次更新项目，只需在当前文件夹下执行以下命令：

 git add .
 git commit -m "更新说明"
 git push



远程移除操作

git remote remove origin

查看远程地址

git remote -v



#### **git pull**

1. 举例

   > git clone ssh://git@codehub.srdcloud.cn:29418/lyh182/wandb-repo.git

2. 如果想要clone到指定文件夹

   > git clone ssh://git@codehub.srdcloud.cn:29418/lyh182/wandb-repo.git my-folder

3. 同步云端的新内容，只需执行

   > git pull

4. 指定分支

   > git pull origin main



#### git 分支操作

1. 查看分支：git branch

2. 新建并切换：git checkout -b 新分支名(develop)

3. 切换分支：git checkout 分支名

4. 在分支上进行修改 并推送分支：

   > git add .
   > git commit -m "在 develop 分支上进行更新"
   >
   > git push -u origin 分支名

5. 合并分支,把 develop 的内容合并回 main：

   > git merge 分支名
   >
   > git push origin main



#### **生成密钥的操作**

> 1. ssh-keygen -t rsa -b 4096 -C "347816314@qq.com"
>
>    一路回车
>
>    会生成两个文件：
>
>    - `~/.ssh/id_rsa`（私钥）
>    - `~/.ssh/id_rsa.pub`（公钥）
>
> 2. cat ~/.ssh/id_rsa.pub 复制公钥
>
>    cat ~/.ssh/id_rsa
>
> 3. 登录云平台 → 添加 SSH 公钥
>
> 4. 测试连接 ssh -T git@codehub.srdcloud.cn -p 29418





### wandb

1. 服务器中对需要同步的wandb**文件夹压缩**

   > zip -r wandb_0830.zip filename

2. 服务器同步到云桌面

   > 注意：路径末尾的 / 很重要，它决定了是同步文件夹本身还是其内容
   >
   > 直接同步到wandb0829文件夹（此文件夹固定，不需要修改名称），压缩之后直接push

   ```python
   # 直接复制文件夹的路径
   rsync -avz -e "ssh -p 30022" liyihain@root@ssh-418.default@10.30.129.200:/gemini/space/private/lyh/Code/role_grpo_exp5/wandb/wandb_exp5_2.zip /drives/d/wandb0829/
       
       
   # 直接复制文件夹的路径
   rsync -avz -e "ssh -p 30022" liyihain@root@ssh-418.default@10.30.129.200:/gemini/space/private/lyh/Code/role_grpo.zip /drives/d/wandb0829/ 
   ```

3. git push到远程仓库

   >  git add .
   >  git commit -m "更新wandb"
   >  git push

4. 下载wandb压缩包，在vscode wandb文件夹中

   > wandb sync 文件名  例如offline-run-20250829_100000-aaaaaaaa



### conda

1. conda create --name 6g_env python=3.10

2. conda env list

3. conda activate pachong

4. conda deactivate



### tmux

1. **列出所有会话**

   ```bash
   tmux ls
   ```

2. **新建会话**

   ```bash
   # 简单
   tmux
   # 新建（不指定名字，系统会自动编号）
   tmux new
   # 指定名字
   tmux new -s mysession
   ```

3. **进入已有会话**

   ```bash
   tmux attach -t 1（会话名称）
   # 或简写
tmux a -t 1
   ```

4. **分离（退出会话但不关闭）**

   ```bash
   Ctrl+b d
   ```

5. **杀掉（关闭）会话**

   ```bash
   # 关闭指定对话
   tmux kill-session -t 0
   ```

6. **解决tmux中文显示下划线的问题**

   ```bash
   vim ~/.bashrc
   
   export LANG=zh_CN.UTF-8
   export LC_ALL=zh_CN.UTF-8
   
   source ~/.bashrc
   ```

   

解压的操作

unzip your_file.zip -d file







