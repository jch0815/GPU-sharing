#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Linux Worker节点
"""

import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

# 导入模块
from core.gpu_detector import GPUMonitor
from communication.client import CommunicationClient
from communication.discovery import DeviceDiscovery
from core.task_executor import TaskExecutorFactory

logger = logging.getLogger(__name__)

class LinuxWorker:
    """Linux Worker节点"""

    def __init__(self, worker_id=None, master_host=None, master_port=None):
        # Worker配置
        self.worker_id = worker_id or f"linux-worker-{os.uname().nodename}-{int(time.time())}"
        self.master_host = master_host or os.environ.get('MASTER_HOST', 'gpu-share-master')
        self.master_port = int(master_port or os.environ.get('MASTER_PORT', '5001'))

        # GPU监控器
        self.gpu_monitor = GPUMonitor(update_interval=5)
        self.gpu_monitor.add_callback(self._on_gpu_update)

        # 通信客户端
        self.comm_client = CommunicationClient(
            server_host=self.master_host,
            server_port=self.master_port,
            worker_id=self.worker_id,
            update_callback=self._on_gpu_update
        )

        # 任务执行器
        self.task_executor = None

        # 运行状态
        self.running = False

        # 设备发现
        self.discovery = None
        if os.environ.get('AUTO_DISCOVERY', 'true').lower() == 'true':
            self.discovery = DeviceDiscovery(device_type="worker")

    def start(self):
        """启动Worker"""
        if self.running:
            logger.warning("Worker已在运行中")
            return

        logger.info(f"启动Linux Worker: {self.worker_id}")

        # 初始化GPU监控
        self.gpu_monitor.start()

        # 初始化任务执行器
        self.task_executor = TaskExecutorFactory.create_executor("linux", self.gpu_monitor)

        # 启动通信客户端
        self.comm_client.start()

        # 启动设备发现
        if self.discovery:
            self.discovery.start()

        self.running = True
        logger.info("Linux Worker已启动")

    def stop(self):
        """停止Worker"""
        if not self.running:
            return

        logger.info("正在停止Linux Worker...")

        # 停止通信客户端
        self.comm_client.stop()

        # 停止GPU监控
        self.gpu_monitor.stop()

        # 停止设备发现
        if self.discovery:
            self.discovery.stop()

        self.running = False
        logger.info("Linux Worker已停止")

    def _on_gpu_update(self, gpu_list):
        """GPU更新回调"""
        # 更新通信客户端中的GPU信息
        self.comm_client.update_gpu_info(gpu_list)

    def handle_task_assignment(self, task_info):
        """处理任务分配"""
        logger.info(f"收到任务分配: {task_info['task_id']}")

        # 执行任务
        result = self.task_executor.execute_task(task_info)

        # 报告任务结果
        self.comm_client.report_task_result(
            task_id=task_info['task_id'],
            result=result.get("result", {}),
            success=result.get("success", False),
            error_message=result.get("error", "")
        )

        return result.get("success", False)

    def get_system_info(self) -> Dict:
        """获取系统信息"""
        import platform

        # 获取GPU信息
        gpu_list = []
        for gpu in self.gpu_monitor.get_gpu_info():
            gpu_list.append({
                "gpu_id": gpu.gpu_id,
                "name": gpu.name,
                "memory_total": gpu.memory_total,
                "memory_used": gpu.memory_used,
                "utilization": gpu.utilization,
                "temperature": gpu.temperature,
                "power_usage": gpu.power_usage,
                "compute_capability": gpu.compute_capability,
                "is_available": gpu.is_available
            })

        return {
            "worker_id": self.worker_id,
            "hostname": os.uname().nodename,
            "platform": "Linux",
            "status": "online",
            "last_heartbeat": time.time(),
            "gpus": gpu_list
        }

def create_systemd_service():
    """创建systemd服务文件"""
    service_content = """[Unit]
Description=GPU共享平台Worker
After=network.target

[Service]
Type=simple
User=gpu-share
WorkingDirectory=/opt/gpu-share
ExecStart=/opt/gpu-share/start-worker.sh
ExecStop=/opt/gpu-share/stop-worker.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""

    try:
        # 创建目录
        os.makedirs('/opt/gpu-share', exist_ok=True)

        # 写入服务文件
        with open('/etc/systemd/system/gpu-share-worker.service', 'w') as f:
            f.write(service_content)

        # 重新加载systemd
        import subprocess
        subprocess.run(['systemctl', 'daemon-reload'], check=True)

        # 启用服务
        subprocess.run(['systemctl', 'enable', 'gpu-share-worker.service'], check=True)

        logger.info("systemd服务已创建并启用")
        return True
    except Exception as e:
        logger.error(f"创建systemd服务失败: {str(e)}")
        return False

def create_startup_scripts():
    """创建启动和停止脚本"""
    # 启动脚本
    start_script = """#!/bin/bash
# 启动GPU共享平台Worker

cd /opt/gpu-share
/opt/gpu-share/venv/bin/python workers/linux/worker.py "$@" &
echo $! > /opt/gpu-share/worker.pid
"""

    # 停止脚本
    stop_script = """#!/bin/bash
# 停止GPU共享平台Worker

if [ -f /opt/gpu-share/worker.pid ]; then
    PID=$(cat /opt/gpu-share/worker.pid)
    kill $PID
    rm -f /opt/gpu-share/worker.pid
fi
"""

    try:
        # 写入启动脚本
        with open('/opt/gpu-share/start-worker.sh', 'w') as f:
            f.write(start_script)

        # 写入停止脚本
        with open('/opt/gpu-share/stop-worker.sh', 'w') as f:
            f.write(stop_script)

        # 设置执行权限
        os.chmod('/opt/gpu-share/start-worker.sh', 0o755)
        os.chmod('/opt/gpu-share/stop-worker.sh', 0o755)

        logger.info("启动和停止脚本已创建")
        return True
    except Exception as e:
        logger.error(f"创建启动和停止脚本失败: {str(e)}")
        return False

def install_worker():
    """安装Worker"""
    # 创建目录
    os.makedirs('/opt/gpu-share', exist_ok=True)

    # 复制项目文件
    import shutil
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    shutil.copytree(project_dir, '/opt/gpu-share', dirs_exist_ok=True)

    # 创建虚拟环境
    subprocess.run(['python3', '-m', 'venv', '/opt/gpu-share/venv'], check=True)

    # 安装依赖
    subprocess.run(['/opt/gpu-share/venv/bin/pip', 'install', '-r', '/opt/gpu-share/requirements.txt'], check=True)

    # 创建systemd服务和启动脚本
    create_systemd_service()
    create_startup_scripts()

    logger.info("Worker已安装")
    return True

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Linux GPU Worker")
    parser.add_argument("--worker-id", help="Worker ID")
    parser.add_argument("--master-host", help="主节点主机地址")
    parser.add_argument("--master-port", type=int, help="主节点端口")
    parser.add_argument("--install", action="store_true", help="安装为系统服务")
    parser.add_argument("--log-level", default="INFO", help="日志级别")

    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('linux-worker.log'),
            logging.StreamHandler()
        ]
    )

    # 安装为系统服务
    if args.install:
        return install_worker()

    # 创建并启动Worker
    worker = LinuxWorker(
        worker_id=args.worker_id,
        master_host=args.master_host,
        master_port=args.master_port
    )

    try:
        worker.start()

        # 保持运行
        while worker.running:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("收到终止信号，正在关闭Worker...")
    finally:
        worker.stop()

if __name__ == "__main__":
    main()
