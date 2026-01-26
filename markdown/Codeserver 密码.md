Codeserver 密码

``` python
,.Kwai666iawK.,
```



``` python
# 以 Ubuntu 为例，其它系统版本稍微修改脚本即可

# clone repo
git clone https://git.corp.kuaishou.com/guojianzhu/kml-vscode.git
cd kml-vscode 

# 在 KML 服务器端
# 安装 code server 以及一些推荐的插件
# 默认密码为   ,.Kwai666iawK.,
# 不建议使用默认密码，请在 https://git.corp.kuaishou.com/guojianzhu/kml-vscode/-/blob/master/run_code_server.sh#L16 进行修改！
./install_code_server_and_plugins.sh

apt-get install lsof -y # 若无，需要先安装 lsof
./run_code_server.sh # kill 8888 端口对应的服务，并在 8888 端口启动 code server

# 在客户端：macOS
./run_code_client.sh ${你的 kml 地址} #（地址通过点击 kml 页面的 juputer 来复制，多端口点击右边的三个点展开）

# Windows
# 打开 run_code_client_windows.bat 设置 kml 链接，然后双击！


```

