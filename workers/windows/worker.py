#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Windows Worker节点
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

class WindowsWorker:
    """Windows Worker节点"""

    def __init__(self, worker_id=None, master_host=None, master_port=None):
        # Worker配置
        self.worker_id = worker_id or f"windows-worker-{os.environ.get('COMPUTERNAME', 'unknown')}-{int(time.time())}"
        self.master_host = master_host or os.environ.get('MASTER_HOST', 'localhost')
        self.master_port = int(master_port or os.environ.get('MASTER_PORT', '50051'))

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

        logger.info(f"启动Windows Worker: {self.worker_id}")

        # 初始化GPU监控
        self.gpu_monitor.start()

        # 初始化任务执行器
        self.task_executor = TaskExecutorFactory.create_executor("windows", self.gpu_monitor)

        # 启动通信客户端
        self.comm_client.start()

        # 启动设备发现
        if self.discovery:
            self.discovery.start()

        self.running = True
        logger.info("Windows Worker已启动")

    def stop(self):
        """停止Worker"""
        if not self.running:
            return

        logger.info("正在停止Windows Worker...")

        # 停止通信客户端
        self.comm_client.stop()

        # 停止GPU监控
        self.gpu_monitor.stop()

        # 停止设备发现
        if self.discovery:
            self.discovery.stop()

        self.running = False
        logger.info("Windows Worker已停止")

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
            "hostname": os.environ.get('COMPUTERNAME', 'unknown'),
            "platform": "Windows",
            "status": "online",
            "last_heartbeat": time.time(),
            "gpus": gpu_list
        }

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Windows GPU Worker")
    parser.add_argument("--worker-id", help="Worker ID")
    parser.add_argument("--master-host", help="主节点主机地址")
    parser.add_argument("--master-port", type=int, help="主节点端口")
    parser.add_argument("--config", help="配置文件路径")
    parser.add_argument("--log-level", default="INFO", help="日志级别")

    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('windows-worker.log'),
            logging.StreamHandler()
        ]
    )

    # 加载配置文件
    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)

    # 创建并启动Worker
    worker = WindowsWorker(
        worker_id=args.worker_id or config.get('worker_id'),
        master_host=args.master_host or config.get('master_host'),
        master_port=args.master_port or config.get('master_port')
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
