#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Windows平台Worker实现
"""

import os
import sys
import time
import json
import logging
import platform
import socket
import threading
import subprocess
from typing import Dict, List, Optional, Any

import psutil
import pynvml

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from communication.client import CommunicationClient

logger = logging.getLogger(__name__)

class WindowsGPUWorker:
    """Windows平台GPU Worker"""

    def __init__(self, server_host, server_port, worker_id=None):
        self.server_host = server_host
        self.server_port = server_port
        self.worker_id = worker_id or f"win-{socket.gethostname()}-{int(time.time())}"

        # 系统信息
        self.hostname = socket.gethostname()
        self.platform = "Windows"

        # GPU信息
        self.gpu_info = []

        # 通信客户端
        self.comm_client = None

        # 运行状态
        self.running = False

        # 初始化NVIDIA管理库
        try:
            pynvml.nvmlInit()
            self.nvml_available = True
        except Exception as e:
            logger.warning(f"无法初始化NVIDIA管理库: {str(e)}")
            self.nvml_available = False

    def start(self):
        """启动Worker"""
        if self.running:
            logger.warning("Worker已在运行中")
            return

        logger.info(f"启动Windows GPU Worker: {self.worker_id}")

        # 初始化GPU信息
        self._init_gpu_info()

        # 初始化通信客户端
        self.comm_client = CommunicationClient(
            server_host=self.server_host,
            server_port=self.server_port,
            worker_id=self.worker_id,
            gpu_info=self.gpu_info,
            update_callback=self._update_gpu_info
        )

        # 启动通信客户端
        self.comm_client.start()

        self.running = True
        logger.info("Windows GPU Worker已启动")

    def stop(self):
        """停止Worker"""
        if not self.running:
            return

        logger.info("正在停止Windows GPU Worker...")
        self.running = False

        # 停止通信客户端
        if self.comm_client:
            self.comm_client.stop()

        logger.info("Windows GPU Worker已停止")

    def _init_gpu_info(self) -> List[Dict]:
        """初始化GPU信息"""
        self.gpu_info = []

        if not self.nvml_available:
            logger.warning("NVIDIA管理库不可用，无法检测GPU")
            return self.gpu_info

        try:
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                # 获取GPU名称
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')

                # 获取内存信息
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_total = mem_info.total // (1024 * 1024)  # 转换为MB
                memory_used = mem_info.used // (1024 * 1024)    # 转换为MB

                # 获取利用率
                util_rate = pynvml.nvmlDeviceGetUtilizationRates(handle)
                utilization = util_rate.gpu

                # 获取温度
                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    temperature = 0

                # 获取功耗
                try:
                    power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # 转换为瓦特
                except:
                    power_usage = 0

                # 获取计算能力
                try:
                    major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                    compute_capability = f"{major}.{minor}"
                except:
                    compute_capability = "0.0"

                gpu_id = f"{self.worker_id}-gpu-{i}"

                self.gpu_info.append({
                    "gpu_id": gpu_id,
                    "name": name,
                    "memory_total": memory_total,
                    "memory_used": memory_used,
                    "utilization": utilization,
                    "temperature": temperature,
                    "power_usage": power_usage,
                    "compute_capability": compute_capability,
                    "is_available": True
                })

                logger.info(f"检测到GPU: {name} (ID: {gpu_id})")

        except Exception as e:
            logger.error(f"初始化GPU信息时出错: {str(e)}")

        return self.gpu_info

    def _update_gpu_info(self) -> List[Dict]:
        """更新GPU信息"""
        if not self.nvml_available:
            return self.gpu_info

        try:
            for i, gpu in enumerate(self.gpu_info):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                # 更新内存信息
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu["memory_used"] = mem_info.used // (1024 * 1024)  # 转换为MB

                # 更新利用率
                util_rate = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu["utilization"] = util_rate.gpu

                # 更新温度
                try:
                    gpu["temperature"] = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    pass

                # 更新功耗
                try:
                    gpu["power_usage"] = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # 转换为瓦特
                except:
                    pass

        except Exception as e:
            logger.error(f"更新GPU信息时出错: {str(e)}")

        return self.gpu_info

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Windows GPU Worker")
    parser.add_argument("--host", default="localhost", help="服务器地址")
    parser.add_argument("--port", type=int, default=50051, help="服务器端口")
    parser.add_argument("--worker-id", help="Worker ID")
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

    # 创建并启动Worker
    worker = WindowsGPUWorker(
        server_host=args.host,
        server_port=args.port,
        worker_id=args.worker_id
    )

    try:
        worker.start()

        # 保持运行
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("接收到终止信号，正在关闭Worker...")
    finally:
        worker.stop()

if __name__ == "__main__":
    main()
