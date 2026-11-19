#!/bin/bash

# 这道闸是 2026-11-19 加的安全约束，不要删。
# 本脚本会让后端进程带 debugpy 调试口起来（默认 127.0.0.1:5678/5679，无鉴权、进程跑在 root 下）。
# 不想被调试模式误起服务就正常用 ./start_server.sh；真要起调试模式必须显式确认。
if [ "${ALLOW_DEBUG_LISTEN}" != "1" ]; then
    echo "拒绝启动：本脚本会以 debugpy 起 ragflow_server / task_executor（调试口无鉴权）。"
    echo "确认要起调试模式：ALLOW_DEBUG_LISTEN=1 ./start_server_debug.sh"
    echo "只想正常起服务：./start_server.sh"
    exit 1
fi

echo "切换当前python脚本到虚拟解释器 .venv/bin/activate"
source .venv/bin/activate

# 检查并删除现有的 .pid 文件
if ls *.pid 1> /dev/null 2>&1; then
    echo "发现现有的 .pid 文件，正在删除..."
    rm -v *.pid
else
    echo "未发现现有的 .pid 文件。"
fi

# 打印启动命令
echo "启动命令：nohup bash launch_backend_service_debug.sh > nohup.out 2>&1 &"
# 启动后台进程
nohup bash docker/launch_backend_service_debug.sh > nohup.out 2>&1 &

# 获取 PID
PID=$!

# 生成以 PID 为文件名的空文件
touch "${PID}.pid"

# 打印提示信息
echo "launch_backend_service_debug.sh 已启动，PID 为 ${PID}，空文件已生成：${PID}.pid"