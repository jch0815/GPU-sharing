#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Windows GPU检测模块
"""

import logging
import os
import sys
import time
from typing import Dict, List, Optional, Callable

logger = logging.getLogger(__name__)

class WindowsGPUInfo:
    """Windows GPU信息类"""

    def __init__(self, gpu_id, name, memory_total, memory_used, utilization, 
                 temperature, power_usage, compute_capability, is_available):
        self.gpu_id = gpu_id
        self.name = name
        self.memory_total = memory_total
        self.memory_used = memory_used
        self.utilization = utilization
        self.temperature = temperature
        self.power_usage = power_usage
        self.compute_capability = compute_capability
        self.is_available = is_available

class WindowsGPUDetector:
    """Windows GPU检测器"""

    def __init__(self, update_interval=5):
        self.update_interval = update_interval
        self.gpus = []
        self.callbacks = []
        self.running = False

        # 尝试导入pynvml
        self.nvml_available = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.nvml_available = True
            logger.info("NVML库已加载")
        except ImportError:
            logger.warning("pynvml库未安装，无法获取GPU详细信息")
        except Exception as e:
            logger.error(f"初始化NVML失败: {str(e)}")

        # 尝试导入PyCUDA
        self.cuda_available = False
        try:
            import pycuda.driver as cuda
            cuda.init()
            self.cuda = cuda
            self.cuda_available = True
            logger.info("PyCUDA库已加载")
        except ImportError:
            logger.warning("PyCUDA库未安装，无法执行GPU计算")
        except Exception as e:
            logger.error(f"初始化PyCUDA失败: {str(e)}")

    def start(self):
        """启动GPU检测"""
        if self.running:
            logger.warning("GPU检测器已在运行中")
            return

        self.running = True
        logger.info("启动Windows GPU检测器")

        # 初始检测
        self._detect_gpus()

        # 启动更新线程
        import threading
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()

    def stop(self):
        """停止GPU检测"""
        if not self.running:
            return

        self.running = False
        logger.info("停止Windows GPU检测器")

        # 等待更新线程结束
        if hasattr(self, 'update_thread'):
            self.update_thread.join()

    def add_callback(self, callback: Callable[[List[WindowsGPUInfo]], None]):
        """添加更新回调"""
        self.callbacks.append(callback)

    def _detect_gpus(self):
        """检测GPU"""
        old_gpus = self.gpus.copy()
        self.gpus = []

        if self.nvml_available:
            try:
                # 获取GPU数量
                device_count = self.nvml.nvmlDeviceGetCount()
                logger.info(f"检测到 {device_count} 个NVIDIA GPU")

                # 遍历GPU
                for i in range(device_count):
                    handle = self.nvml.nvmlDeviceGetHandleByIndex(i)

                    # 获取GPU信息
                    name = self.nvml.nvmlDeviceGetName(handle).decode('utf-8')
                    memory_info = self.nvml.nvmlDeviceGetMemoryInfo(handle)
                    memory_total = memory_info.total // (1024 * 1024)  # MB
                    memory_used = memory_info.used // (1024 * 1024)  # MB

                    try:
                        utilization = self.nvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_util = utilization.gpu
                    except:
                        gpu_util = 0.0

                    try:
                        temperature = self.nvml.nvmlDeviceGetTemperature(handle, self.nvml.NVML_TEMPERATURE_GPU)
                    except:
                        temperature = 0.0

                    try:
                        power_usage = self.nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # W
                    except:
                        power_usage = 0.0

                    try:
                        compute_capability = f"{self.nvml.nvmlDeviceGetCudaComputeCapability(handle)[0]}.{self.nvml.nvmlDeviceGetCudaComputeCapability(handle)[1]}"
                    except:
                        compute_capability = "0.0"

                    # 创建GPU信息
                    gpu_info = WindowsGPUInfo(
                        gpu_id=f"gpu-{i}",
                        name=name,
                        memory_total=memory_total,
                        memory_used=memory_used,
                        utilization=gpu_util,
                        temperature=temperature,
                        power_usage=power_usage,
                        compute_capability=compute_capability,
                        is_available=True
                    )

                    self.gpus.append(gpu_info)
                    logger.debug(f"检测到GPU: {name}, 内存: {memory_total}MB, 利用率: {gpu_util}%")

            except Exception as e:
                logger.error(f"检测NVIDIA GPU失败: {str(e)}")

        # 如果没有检测到NVIDIA GPU，尝试使用PyCUDA
        if not self.gpus and self.cuda_available:
            try:
                # 获取GPU数量
                device_count = self.cuda.Device.count()
                logger.info(f"PyCUDA检测到 {device_count} 个CUDA设备")

                # 遍历GPU
                for i in range(device_count):
                    device = self.cuda.Device(i)

                    # 获取GPU信息
                    name = device.name()
                    total_memory = device.total_memory() // (1024 * 1024)  # MB

                    # 创建GPU信息
                    gpu_info = WindowsGPUInfo(
                        gpu_id=f"gpu-{i}",
                        name=name,
                        memory_total=total_memory,
                        memory_used=0,  # PyCUDA无法获取已用内存
                        utilization=0,  # PyCUDA无法获取利用率
                        temperature=0,  # PyCUDA无法获取温度
                        power_usage=0,  # PyCUDA无法获取功耗
                        compute_capability=f"{device.compute_capability()[0]}.{device.compute_capability()[1]}",
                        is_available=True
                    )

                    self.gpus.append(gpu_info)
                    logger.debug(f"检测到CUDA设备: {name}, 内存: {total_memory}MB")

            except Exception as e:
                logger.error(f"检测CUDA设备失败: {str(e)}")

        # 如果没有检测到GPU，创建一个虚拟GPU用于测试
        if not self.gpus:
            logger.warning("未检测到GPU，创建虚拟GPU用于测试")
            gpu_info = WindowsGPUInfo(
                gpu_id="gpu-virtual",
                name="Virtual GPU",
                memory_total=4096,  # 4GB
                memory_used=1024,   # 1GB
                utilization=25.0,
                temperature=65.0,
                power_usage=150.0,
                compute_capability="6.0",
                is_available=True
            )

            self.gpus.append(gpu_info)

        # 通知回调
        if old_gpus != self.gpus:
            for callback in self.callbacks:
                callback(self.gpus)

    def _update_loop(self):
        """更新循环"""
        while self.running:
            # 更新GPU信息
            self._detect_gpus()

            # 等待
            time.sleep(self.update_interval)

    def get_gpu_info(self) -> List[WindowsGPUInfo]:
        """获取GPU信息"""
        return self.gpus

    def get_gpu_info_by_id(self, gpu_id: str) -> Optional[WindowsGPUInfo]:
        """根据ID获取GPU信息"""
        for gpu in self.gpus:
            if gpu.gpu_id == gpu_id:
                return gpu
        return None

    def get_available_gpus(self) -> List[WindowsGPUInfo]:
        """获取可用GPU"""
        return [gpu for gpu in self.gpus if gpu.is_available]

    def get_total_memory(self) -> int:
        """获取总内存"""
        return sum(gpu.memory_total for gpu in self.gpus)

    def get_used_memory(self) -> int:
        """获取已用内存"""
        return sum(gpu.memory_used for gpu in self.gpus)

    def get_free_memory(self) -> int:
        """获取空闲内存"""
        return self.get_total_memory() - self.get_used_memory()

    def get_average_utilization(self) -> float:
        """获取平均利用率"""
        if not self.gpus:
            return 0.0
        return sum(gpu.utilization for gpu in self.gpus) / len(self.gpus)

    def get_average_temperature(self) -> float:
        """获取平均温度"""
        if not self.gpus:
            return 0.0
        return sum(gpu.temperature for gpu in self.gpus) / len(self.gpus)

    def get_total_power_usage(self) -> float:
        """获取总功耗"""
        return sum(gpu.power_usage for gpu in self.gpus)

# 创建Windows GPU检测器实例
def get_gpu_detector(update_interval=5) -> WindowsGPUDetector:
    """获取Windows GPU检测器实例"""
    return WindowsGPUDetector(update_interval)

# 兼容性函数
def get_gpu_info() -> List[Dict]:
    """获取GPU信息（兼容性函数）"""
    detector = WindowsGPUDetector()
    detector._detect_gpus()

    result = []
    for gpu in detector.gpus:
        result.append({
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

    return result
