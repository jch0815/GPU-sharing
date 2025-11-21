#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPU资源检测模块
"""

import abc
import json
import logging
import os
import platform
import subprocess
import time
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

class GPUInfo:
    """GPU信息数据类"""

    def __init__(self, gpu_id: str, name: str, memory_total: int, memory_used: int, 
                 utilization: float, temperature: float, power_usage: float,
                 compute_capability: str, is_available: bool = True):
        self.gpu_id = gpu_id
        self.name = name
        self.memory_total = memory_total  # MB
        self.memory_used = memory_used    # MB
        self.utilization = utilization    # 0-100%
        self.temperature = temperature    # °C
        self.power_usage = power_usage    # W
        self.compute_capability = compute_capability
        self.is_available = is_available
        self.last_update = time.time()

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "gpu_id": self.gpu_id,
            "name": self.name,
            "memory_total": self.memory_total,
            "memory_used": self.memory_used,
            "utilization": self.utilization,
            "temperature": self.temperature,
            "power_usage": self.power_usage,
            "compute_capability": self.compute_capability,
            "is_available": self.is_available,
            "last_update": self.last_update
        }

    def to_json(self) -> str:
        """转换为JSON格式"""
        return json.dumps(self.to_dict())

class BaseGPUDetector(abc.ABC):
    """GPU检测器基类"""

    @abc.abstractmethod
    def detect_gpus(self) -> List[GPUInfo]:
        """检测所有可用的GPU"""
        pass

    @abc.abstractmethod
    def update_gpu_info(self, gpu_info: GPUInfo) -> bool:
        """更新GPU信息"""
        pass

    @staticmethod
    def get_detector() -> 'BaseGPUDetector':
        """根据平台获取合适的检测器"""
        system = platform.system()

        if system == "Windows":
            return WindowsGPUDetector()
        elif system == "Linux":
            # 检查是否在Docker容器内
            if os.path.exists('/.dockerenv'):
                return DockerGPUDetector()
            else:
                return LinuxGPUDetector()
        else:
            logger.warning(f"不支持的操作系统: {system}")
            return None

class WindowsGPUDetector(BaseGPUDetector):
    """Windows平台GPU检测器"""

    def __init__(self):
        self.nvml_available = self._init_nvml()
        self.opencl_available = self._init_opencl()

    def _init_nvml(self) -> bool:
        """初始化NVIDIA管理库"""
        try:
            import pynvml
            pynvml.nvmlInit()
            self.pynvml = pynvml
            logger.info("NVIDIA管理库初始化成功")
            return True
        except Exception as e:
            logger.warning(f"无法初始化NVIDIA管理库: {str(e)}")
            return False

    def _init_opencl(self) -> bool:
        """初始化OpenCL"""
        try:
            import pyopencl as cl
            self.cl = cl
            logger.info("OpenCL初始化成功")
            return True
        except Exception as e:
            logger.warning(f"无法初始化OpenCL: {str(e)}")
            return False

    def detect_gpus(self) -> List[GPUInfo]:
        """检测所有可用的GPU"""
        gpu_list = []

        # 首先尝试使用NVML检测NVIDIA GPU
        if self.nvml_available:
            gpu_list.extend(self._detect_nvidia_gpus())

        # 然后尝试使用OpenCL检测其他GPU
        if self.opencl_available:
            gpu_list.extend(self._detect_opencl_gpus())

        return gpu_list

    def _detect_nvidia_gpus(self) -> List[GPUInfo]:
        """使用NVML检测NVIDIA GPU"""
        gpu_list = []

        try:
            device_count = self.pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = self.pynvml.nvmlDeviceGetHandleByIndex(i)

                # 获取GPU名称
                name = self.pynvml.nvmlDeviceGetName(handle).decode('utf-8')

                # 获取内存信息
                mem_info = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_total = mem_info.total // (1024 * 1024)  # 转换为MB
                memory_used = mem_info.used // (1024 * 1024)    # 转换为MB

                # 获取利用率
                util_rate = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
                utilization = util_rate.gpu

                # 获取温度
                try:
                    temperature = self.pynvml.nvmlDeviceGetTemperature(handle, self.pynvml.NVML_TEMPERATURE_GPU)
                except:
                    temperature = 0

                # 获取功耗
                try:
                    power_usage = self.pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # 转换为瓦特
                except:
                    power_usage = 0

                # 获取计算能力
                try:
                    major, minor = self.pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                    compute_capability = f"{major}.{minor}"
                except:
                    compute_capability = "0.0"

                gpu_id = f"nvidia-{i}"

                gpu_list.append(GPUInfo(
                    gpu_id=gpu_id,
                    name=name,
                    memory_total=memory_total,
                    memory_used=memory_used,
                    utilization=utilization,
                    temperature=temperature,
                    power_usage=power_usage,
                    compute_capability=compute_capability,
                    is_available=True
                ))

                logger.info(f"检测到NVIDIA GPU: {name} (ID: {gpu_id})")

        except Exception as e:
            logger.error(f"检测NVIDIA GPU时出错: {str(e)}")

        return gpu_list

    def _detect_opencl_gpus(self) -> List[GPUInfo]:
        """使用OpenCL检测GPU"""
        gpu_list = []

        try:
            platforms = self.cl.get_platforms()

            for p_idx, platform in enumerate(platforms):
                devices = platform.get_devices(device_type=self.cl.device_type.GPU)

                for d_idx, device in enumerate(devices):
                    # 获取设备名称
                    name = device.name

                    # 获取内存信息
                    memory_total = device.global_mem_size // (1024 * 1024)  # 转换为MB
                    memory_used = 0  # OpenCL没有直接获取已用内存的方法

                    # 获取计算单元数作为利用率的参考
                    compute_units = device.max_compute_units

                    # 获取其他信息
                    extensions = device.extensions.strip().split(' ')
                    is_cuda_available = 'cl_khr_gl_sharing' in extensions or 'cl_nv_device_attribute_query' in extensions

                    gpu_id = f"opencl-{p_idx}-{d_idx}"

                    gpu_list.append(GPUInfo(
                        gpu_id=gpu_id,
                        name=f"{name} (OpenCL)",
                        memory_total=memory_total,
                        memory_used=memory_used,
                        utilization=0,  # OpenCL没有直接获取利用率的方法
                        temperature=0,   # OpenCL没有直接获取温度的方法
                        power_usage=0,  # OpenCL没有直接获取功耗的方法
                        compute_capability="OpenCL",
                        is_available=True
                    ))

                    logger.info(f"检测到OpenCL GPU: {name} (ID: {gpu_id})")

        except Exception as e:
            logger.error(f"检测OpenCL GPU时出错: {str(e)}")

        return gpu_list

    def update_gpu_info(self, gpu_info: GPUInfo) -> bool:
        """更新GPU信息"""
        if gpu_info.gpu_id.startswith("nvidia-"):
            return self._update_nvidia_gpu(gpu_info)
        elif gpu_info.gpu_id.startswith("opencl-"):
            return self._update_opencl_gpu(gpu_info)
        else:
            logger.warning(f"未知的GPU ID格式: {gpu_info.gpu_id}")
            return False

    def _update_nvidia_gpu(self, gpu_info: GPUInfo) -> bool:
        """更新NVIDIA GPU信息"""
        if not self.nvml_available:
            return False

        try:
            # 从ID中提取设备索引
            device_index = int(gpu_info.gpu_id.split('-')[1])
            handle = self.pynvml.nvmlDeviceGetHandleByIndex(device_index)

            # 更新内存信息
            mem_info = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_info.memory_used = mem_info.used // (1024 * 1024)  # 转换为MB

            # 更新利用率
            util_rate = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_info.utilization = util_rate.gpu

            # 更新温度
            try:
                gpu_info.temperature = self.pynvml.nvmlDeviceGetTemperature(handle, self.pynvml.NVML_TEMPERATURE_GPU)
            except:
                pass

            # 更新功耗
            try:
                gpu_info.power_usage = self.pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # 转换为瓦特
            except:
                pass

            # 更新时间戳
            gpu_info.last_update = time.time()

            return True

        except Exception as e:
            logger.error(f"更新NVIDIA GPU信息时出错: {str(e)}")
            return False

    def _update_opencl_gpu(self, gpu_info: GPUInfo) -> bool:
        """更新OpenCL GPU信息"""
        # OpenCL API通常不提供实时状态信息，因此这里只更新时间戳
        gpu_info.last_update = time.time()
        return True

class LinuxGPUDetector(BaseGPUDetector):
    """Linux平台GPU检测器"""

    def __init__(self):
        self.nvml_available = self._init_nvml()
        self.opencl_available = self._init_opencl()
        self.nvidia_smi_available = self._check_nvidia_smi()

    def _init_nvml(self) -> bool:
        """初始化NVIDIA管理库"""
        try:
            import pynvml
            pynvml.nvmlInit()
            self.pynvml = pynvml
            logger.info("NVIDIA管理库初始化成功")
            return True
        except Exception as e:
            logger.warning(f"无法初始化NVIDIA管理库: {str(e)}")
            return False

    def _init_opencl(self) -> bool:
        """初始化OpenCL"""
        try:
            import pyopencl as cl
            self.cl = cl
            logger.info("OpenCL初始化成功")
            return True
        except Exception as e:
            logger.warning(f"无法初始化OpenCL: {str(e)}")
            return False

    def _check_nvidia_smi(self) -> bool:
        """检查nvidia-smi是否可用"""
        try:
            result = subprocess.run(['nvidia-smi', '--version'], 
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE,
                                    check=True)
            logger.info("nvidia-smi命令可用")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("nvidia-smi命令不可用")
            return False

    def detect_gpus(self) -> List[GPUInfo]:
        """检测所有可用的GPU"""
        gpu_list = []

        # 优先使用nvidia-smi检测NVIDIA GPU
        if self.nvidia_smi_available:
            gpu_list.extend(self._detect_nvidia_gpus_smi())
        # 如果nvidia-smi不可用，尝试使用NVML
        elif self.nvml_available:
            gpu_list.extend(self._detect_nvidia_gpus_nvml())

        # 尝试使用OpenCL检测其他GPU
        if self.opencl_available:
            gpu_list.extend(self._detect_opencl_gpus())

        return gpu_list

    def _detect_nvidia_gpus_smi(self) -> List[GPUInfo]:
        """使用nvidia-smi检测NVIDIA GPU"""
        gpu_list = []

        try:
            # 执行nvidia-smi命令获取GPU信息
            cmd = [
                'nvidia-smi', 
                '--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,compute_cap',
                '--format=csv,noheader,nounits'
            ]

            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)

            for line in result.stdout.strip().split('
'):
                parts = line.split(', ')
                if len(parts) < 8:
                    continue

                index = parts[0].strip()
                name = parts[1].strip()
                memory_total = int(parts[2].strip())
                memory_used = int(parts[3].strip())
                utilization = float(parts[4].strip())
                temperature = float(parts[5].strip())
                power_usage = float(parts[6].strip())
                compute_capability = parts[7].strip()

                gpu_id = f"nvidia-{index}"

                gpu_list.append(GPUInfo(
                    gpu_id=gpu_id,
                    name=name,
                    memory_total=memory_total,
                    memory_used=memory_used,
                    utilization=utilization,
                    temperature=temperature,
                    power_usage=power_usage,
                    compute_capability=compute_capability,
                    is_available=True
                ))

                logger.info(f"检测到NVIDIA GPU: {name} (ID: {gpu_id})")

        except Exception as e:
            logger.error(f"使用nvidia-smi检测NVIDIA GPU时出错: {str(e)}")

        return gpu_list

    def _detect_nvidia_gpus_nvml(self) -> List[GPUInfo]:
        """使用NVML检测NVIDIA GPU"""
        gpu_list = []

        try:
            device_count = self.pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = self.pynvml.nvmlDeviceGetHandleByIndex(i)

                # 获取GPU名称
                name = self.pynvml.nvmlDeviceGetName(handle).decode('utf-8')

                # 获取内存信息
                mem_info = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_total = mem_info.total // (1024 * 1024)  # 转换为MB
                memory_used = mem_info.used // (1024 * 1024)    # 转换为MB

                # 获取利用率
                util_rate = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
                utilization = util_rate.gpu

                # 获取温度
                try:
                    temperature = self.pynvml.nvmlDeviceGetTemperature(handle, self.pynvml.NVML_TEMPERATURE_GPU)
                except:
                    temperature = 0

                # 获取功耗
                try:
                    power_usage = self.pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # 转换为瓦特
                except:
                    power_usage = 0

                # 获取计算能力
                try:
                    major, minor = self.pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                    compute_capability = f"{major}.{minor}"
                except:
                    compute_capability = "0.0"

                gpu_id = f"nvidia-{i}"

                gpu_list.append(GPUInfo(
                    gpu_id=gpu_id,
                    name=name,
                    memory_total=memory_total,
                    memory_used=memory_used,
                    utilization=utilization,
                    temperature=temperature,
                    power_usage=power_usage,
                    compute_capability=compute_capability,
                    is_available=True
                ))

                logger.info(f"检测到NVIDIA GPU: {name} (ID: {gpu_id})")

        except Exception as e:
            logger.error(f"使用NVML检测NVIDIA GPU时出错: {str(e)}")

        return gpu_list

    def _detect_opencl_gpus(self) -> List[GPUInfo]:
        """使用OpenCL检测GPU"""
        gpu_list = []

        try:
            platforms = self.cl.get_platforms()

            for p_idx, platform in enumerate(platforms):
                devices = platform.get_devices(device_type=self.cl.device_type.GPU)

                for d_idx, device in enumerate(devices):
                    # 获取设备名称
                    name = device.name

                    # 获取内存信息
                    memory_total = device.global_mem_size // (1024 * 1024)  # 转换为MB
                    memory_used = 0  # OpenCL没有直接获取已用内存的方法

                    # 获取计算单元数作为利用率的参考
                    compute_units = device.max_compute_units

                    # 获取其他信息
                    extensions = device.extensions.strip().split(' ')
                    is_cuda_available = 'cl_khr_gl_sharing' in extensions or 'cl_nv_device_attribute_query' in extensions

                    gpu_id = f"opencl-{p_idx}-{d_idx}"

                    gpu_list.append(GPUInfo(
                        gpu_id=gpu_id,
                        name=f"{name} (OpenCL)",
                        memory_total=memory_total,
                        memory_used=memory_used,
                        utilization=0,  # OpenCL没有直接获取利用率的方法
                        temperature=0,   # OpenCL没有直接获取温度的方法
                        power_usage=0,  # OpenCL没有直接获取功耗的方法
                        compute_capability="OpenCL",
                        is_available=True
                    ))

                    logger.info(f"检测到OpenCL GPU: {name} (ID: {gpu_id})")

        except Exception as e:
            logger.error(f"检测OpenCL GPU时出错: {str(e)}")

        return gpu_list

    def update_gpu_info(self, gpu_info: GPUInfo) -> bool:
        """更新GPU信息"""
        if gpu_info.gpu_id.startswith("nvidia-"):
            if self.nvidia_smi_available:
                return self._update_nvidia_gpu_smi(gpu_info)
            elif self.nvml_available:
                return self._update_nvidia_gpu_nvml(gpu_info)
        elif gpu_info.gpu_id.startswith("opencl-"):
            return self._update_opencl_gpu(gpu_info)
        else:
            logger.warning(f"未知的GPU ID格式: {gpu_info.gpu_id}")
            return False

        return False

    def _update_nvidia_gpu_smi(self, gpu_info: GPUInfo) -> bool:
        """使用nvidia-smi更新NVIDIA GPU信息"""
        try:
            # 从ID中提取设备索引
            device_index = gpu_info.gpu_id.split('-')[1]

            # 执行nvidia-smi命令获取GPU信息
            cmd = [
                'nvidia-smi', 
                f'--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw',
                f'--id={device_index}',
                '--format=csv,noheader,nounits'
            ]

            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
            line = result.stdout.strip()
            parts = line.split(', ')

            if len(parts) >= 4:
                gpu_info.memory_used = int(parts[0].strip())
                gpu_info.utilization = float(parts[1].strip())
                gpu_info.temperature = float(parts[2].strip())
                gpu_info.power_usage = float(parts[3].strip())

            # 更新时间戳
            gpu_info.last_update = time.time()

            return True

        except Exception as e:
            logger.error(f"使用nvidia-smi更新NVIDIA GPU信息时出错: {str(e)}")
            return False

    def _update_nvidia_gpu_nvml(self, gpu_info: GPUInfo) -> bool:
        """使用NVML更新NVIDIA GPU信息"""
        if not self.nvml_available:
            return False

        try:
            # 从ID中提取设备索引
            device_index = int(gpu_info.gpu_id.split('-')[1])
            handle = self.pynvml.nvmlDeviceGetHandleByIndex(device_index)

            # 更新内存信息
            mem_info = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_info.memory_used = mem_info.used // (1024 * 1024)  # 转换为MB

            # 更新利用率
            util_rate = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_info.utilization = util_rate.gpu

            # 更新温度
            try:
                gpu_info.temperature = self.pynvml.nvmlDeviceGetTemperature(handle, self.pynvml.NVML_TEMPERATURE_GPU)
            except:
                pass

            # 更新功耗
            try:
                gpu_info.power_usage = self.pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # 转换为瓦特
            except:
                pass

            # 更新时间戳
            gpu_info.last_update = time.time()

            return True

        except Exception as e:
            logger.error(f"使用NVML更新NVIDIA GPU信息时出错: {str(e)}")
            return False

    def _update_opencl_gpu(self, gpu_info: GPUInfo) -> bool:
        """更新OpenCL GPU信息"""
        # OpenCL API通常不提供实时状态信息，因此这里只更新时间戳
        gpu_info.last_update = time.time()
        return True

class DockerGPUDetector(LinuxGPUDetector):
    """Docker容器内GPU检测器"""

    def __init__(self):
        super().__init__()
        # 检查是否使用了nvidia-docker
        self.nvidia_docker_available = self._check_nvidia_docker()

    def _check_nvidia_docker(self) -> bool:
        """检查是否使用了nvidia-docker"""
        try:
            # 检查nvidia-smi是否可用
            result = subprocess.run(['nvidia-smi', '--version'], 
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE)
            if result.returncode == 0:
                logger.info("检测到nvidia-docker环境")
                return True
        except FileNotFoundError:
            pass

        logger.warning("未检测到nvidia-docker环境")
        return False

    def detect_gpus(self) -> List[GPUInfo]:
        """检测所有可用的GPU"""
        if not self.nvidia_docker_available:
            logger.warning("在Docker容器内但未使用nvidia-docker，无法检测GPU")
            return []

        return super().detect_gpus()

class AndroidGPUDetector(BaseGPUDetector):
    """Android平台GPU检测器"""

    def __init__(self, adb_path='adb'):
        self.adb_path = adb_path
        self.adb_available = self._check_adb()

    def _check_adb(self) -> bool:
        """检查ADB是否可用"""
        try:
            result = subprocess.run([self.adb_path, 'version'], 
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE,
                                    check=True)
            logger.info("ADB命令可用")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("ADB命令不可用")
            return False

    def detect_gpus(self) -> List[GPUInfo]:
        """检测所有可用的GPU"""
        if not self.adb_available:
            return []

        gpu_list = []

        try:
            # 获取GPU信息
            cmd = [self.adb_path, 'shell', 'dumpsys', 'gpu']
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)

            # 解析输出
            gpu_name = "Android GPU"
            memory_total = 0  # Android GPU内存信息不易获取
            memory_used = 0
            utilization = 0  # Android GPU利用率信息不易获取
            temperature = 0  # Android GPU温度信息不易获取
            power_usage = 0  # Android GPU功耗信息不易获取
            compute_capability = "OpenCL ES"  # Android通常使用OpenCL ES

            # 尝试从dumpsys输出中提取GPU名称
            for line in result.stdout.split('
'):
                if 'GL_RENDERER' in line or 'GPU' in line:
                    parts = line.split('=')
                    if len(parts) > 1:
                        gpu_name = parts[1].strip()
                        break

            gpu_id = "android-gpu"

            gpu_list.append(GPUInfo(
                gpu_id=gpu_id,
                name=gpu_name,
                memory_total=memory_total,
                memory_used=memory_used,
                utilization=utilization,
                temperature=temperature,
                power_usage=power_usage,
                compute_capability=compute_capability,
                is_available=True
            ))

            logger.info(f"检测到Android GPU: {gpu_name} (ID: {gpu_id})")

        except Exception as e:
            logger.error(f"检测Android GPU时出错: {str(e)}")

        return gpu_list

    def update_gpu_info(self, gpu_info: GPUInfo) -> bool:
        """更新GPU信息"""
        # Android GPU信息不易实时获取，这里只更新时间戳
        gpu_info.last_update = time.time()
        return True

# 资源监控器
class GPUMonitor:
    """GPU资源监控器"""

    def __init__(self, detector: BaseGPUDetector, update_interval: int = 5):
        self.detector = detector
        self.update_interval = update_interval
        self.gpu_info: Dict[str, GPUInfo] = {}  # gpu_id -> GPUInfo
        self.callbacks: List[Callable[[List[GPUInfo]], None]] = []
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

    def start(self):
        """启动监控"""
        with self.lock:
            if self.running:
                return

            self.running = True
            self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.thread.start()

            # 初始检测
            self._detect_gpus()

            logger.info("GPU监控器已启动")

    def stop(self):
        """停止监控"""
        with self.lock:
            if not self.running:
                return

            self.running = False

            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=10)

            logger.info("GPU监控器已停止")

    def get_gpu_info(self) -> List[GPUInfo]:
        """获取GPU信息"""
        with self.lock:
            return list(self.gpu_info.values())

    def get_gpu_info_by_id(self, gpu_id: str) -> Optional[GPUInfo]:
        """根据ID获取GPU信息"""
        with self.lock:
            return self.gpu_info.get(gpu_id)

    def add_callback(self, callback: Callable[[List[GPUInfo]], None]):
        """添加更新回调函数"""
        with self.lock:
            self.callbacks.append(callback)

    def remove_callback(self, callback: Callable[[List[GPUInfo]], None]):
        """移除更新回调函数"""
        with self.lock:
            if callback in self.callbacks:
                self.callbacks.remove(callback)

    def _detect_gpus(self):
        """检测GPU"""
        with self.lock:
            # 检测所有GPU
            detected_gpus = self.detector.detect_gpus()

            # 更新GPU信息
            new_gpu_info = {}
            for gpu in detected_gpus:
                new_gpu_info[gpu.gpu_id] = gpu

            self.gpu_info = new_gpu_info

            # 通知回调
            self._notify_callbacks()

    def _update_gpus(self):
        """更新GPU信息"""
        with self.lock:
            updated = False

            for gpu_id, gpu_info in self.gpu_info.items():
                if self.detector.update_gpu_info(gpu_info):
                    updated = True

            if updated:
                self._notify_callbacks()

    def _notify_callbacks(self):
        """通知所有回调函数"""
        gpu_list = list(self.gpu_info.values())
        for callback in self.callbacks:
            try:
                callback(gpu_list)
            except Exception as e:
                logger.error(f"GPU信息更新回调出错: {str(e)}")

    def _monitor_loop(self):
        """监控循环"""
        while self.running:
            try:
                self._update_gpus()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"GPU监控循环出错: {str(e)}")
                time.sleep(1)  # 出错后短暂休眠

# 工具函数
def get_gpu_detector() -> BaseGPUDetector:
    """获取适合当前平台的GPU检测器"""
    return BaseGPUDetector.get_detector()

def get_gpu_monitor(update_interval: int = 5) -> GPUMonitor:
    """获取GPU监控器"""
    detector = get_gpu_detector()
    if detector is None:
        raise RuntimeError("无法为当前平台创建GPU检测器")

    return GPUMonitor(detector, update_interval)

def detect_gpus() -> List[GPUInfo]:
    """检测所有可用的GPU"""
    detector = get_gpu_detector()
    if detector is None:
        logger.warning("无法为当前平台创建GPU检测器")
        return []

    return detector.detect_gpus()

def get_gpu_report() -> str:
    """获取GPU资源报告(JSON格式)"""
    gpus = detect_gpus()
    gpu_dicts = [gpu.to_dict() for gpu in gpus]
    return json.dumps(gpu_dicts, indent=2)

            # 尝试从输出中提取GPU名称
            for line in result.stdout.split('
'):
                if 'GL_RENDERER' in line:
                    parts = line.split('=')
                    if len(parts) > 1:
                        gpu_name = parts[1].strip()
                    break

            gpu_id = "android-gpu-0"

            gpu_list.append(GPUInfo(
                gpu_id=gpu_id,
                name=gpu_name,
                memory_total=memory_total,
                memory_used=memory_used,
                utilization=utilization,
                temperature=temperature,
                power_usage=power_usage,
                compute_capability=compute_capability,
                is_available=True
            ))

            logger.info(f"检测到Android GPU: {gpu_name} (ID: {gpu_id})")

        except Exception as e:
            logger.error(f"检测Android GPU时出错: {str(e)}")

        return gpu_list

    def update_gpu_info(self, gpu_info: GPUInfo) -> bool:
        """更新GPU信息"""
        # Android GPU信息更新较为复杂，这里只更新时间戳
        gpu_info.last_update = time.time()
        return True

def get_gpu_detector() -> BaseGPUDetector:
    """获取适合当前平台的GPU检测器"""
    return BaseGPUDetector.get_detector()

def detect_all_gpus() -> List[GPUInfo]:
    """检测所有可用的GPU"""
    detector = get_gpu_detector()
    if detector:
        return detector.detect_gpus()
    return []

def update_gpu_info(gpu_info: GPUInfo) -> bool:
    """更新GPU信息"""
    detector = get_gpu_detector()
    if detector:
        return detector.update_gpu_info(gpu_info)
    return False

def get_gpu_report() -> str:
    """获取GPU资源报告(JSON格式)"""
    gpu_list = detect_all_gpus()
    gpu_report = {
        "timestamp": time.time(),
        "gpu_count": len(gpu_list),
        "gpus": [gpu.to_dict() for gpu in gpu_list]
    }
    return json.dumps(gpu_report, indent=2)st[GPUInfo]:
        """使用nvidia-smi检测NVIDIA GPU"""
        gpu_list = []

        try:
            # 获取GPU数量
            result = subprocess.run(['nvidia-smi', '--query-gpu=count', '--format=csv,noheader,nounits'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    check=True,
                                    text=True)

            gpu_count = int(result.stdout.strip())

            # 获取GPU详细信息
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,compute_cap',
                '--format=csv,noheader,nounits'
            ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True)

            lines = result.stdout.strip().split('
')

            for line in lines:
                parts = [part.strip() for part in line.split(',')]

                if len(parts) < 8:
                    continue

                index = parts[0]
                name = parts[1]
                memory_total = int(parts[2])
                memory_used = int(parts[3])
                utilization = float(parts[4])
                temperature = float(parts[5]) if parts[5] != '[Not Supported]' else 0
                power_usage = float(parts[6].replace('W', '')) if parts[6] != '[Not Supported]' else 0
                compute_capability = parts[7]

                gpu_id = f"nvidia-{index}"

                gpu_list.append(GPUInfo(
                    gpu_id=gpu_id,
                    name=name,
                    memory_total=memory_total,
                    memory_used=memory_used,
                    utilization=utilization,
                    temperature=temperature,
                    power_usage=power_usage,
                    compute_capability=compute_capability,
                    is_available=True
                ))

                logger.info(f"检测到NVIDIA GPU: {name} (ID: {gpu_id})")

        except subprocess.CalledProcessError as e:
            logger.error(f"使用nvidia-smi检测GPU时出错: {str(e)}")
        except Exception as e:
            logger.error(f"解析nvidia-smi输出时出错: {str(e)}")

        return gpu_list

    def _detect_nvidia_gpus_nvml(self) -> List[GPUInfo]:
        """使用NVML检测NVIDIA GPU"""
        gpu_list = []

        try:
            device_count = self.pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = self.pynvml.nvmlDeviceGetHandleByIndex(i)

                # 获取GPU名称
                name = self.pynvml.nvmlDeviceGetName(handle).decode('utf-8')

                # 获取内存信息
                mem_info = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_total = mem_info.total // (1024 * 1024)  # 转换为MB
                memory_used = mem_info.used // (1024 * 1024)    # 转换为MB

                # 获取利用率
                util_rate = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
                utilization = util_rate.gpu

                # 获取温度
                try:
                    temperature = self.pynvml.nvmlDeviceGetTemperature(handle, self.pynvml.NVML_TEMPERATURE_GPU)
                except:
                    temperature = 0

                # 获取功耗
                try:
                    power_usage = self.pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # 转换为瓦特
                except:
                    power_usage = 0

                # 获取计算能力
                try:
                    major, minor = self.pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                    compute_capability = f"{major}.{minor}"
                except:
                    compute_capability = "0.0"

                gpu_id = f"nvidia-{i}"

                gpu_list.append(GPUInfo(
                    gpu_id=gpu_id,
                    name=name,
                    memory_total=memory_total,
                    memory_used=memory_used,
                    utilization=utilization,
                    temperature=temperature,
                    power_usage=power_usage,
                    compute_capability=compute_capability,
                    is_available=True
                ))

                logger.info(f"检测到NVIDIA GPU: {name} (ID: {gpu_id})")

        except Exception as e:
            logger.error(f"检测NVIDIA GPU时出错: {str(e)}")

        return gpu_list

    def _detect_opencl_gpus(self) -> List[GPUInfo]:
        """使用OpenCL检测GPU"""
        gpu_list = []

        try:
            platforms = self.cl.get_platforms()

            for p_idx, platform in enumerate(platforms):
                devices = platform.get_devices(device_type=self.cl.device_type.GPU)

                for d_idx, device in enumerate(devices):
                    # 获取设备名称
                    name = device.name

                    # 获取内存信息
                    memory_total = device.global_mem_size // (1024 * 1024)  # 转换为MB
                    memory_used = 0  # OpenCL没有直接获取已用内存的方法

                    # 获取计算单元数作为利用率的参考
                    compute_units = device.max_compute_units

                    # 获取其他信息
                    extensions = device.extensions.strip().split(' ')
                    is_cuda_available = 'cl_khr_gl_sharing' in extensions or 'cl_nv_device_attribute_query' in extensions

                    gpu_id = f"opencl-{p_idx}-{d_idx}"

                    gpu_list.append(GPUInfo(
                        gpu_id=gpu_id,
                        name=f"{name} (OpenCL)",
                        memory_total=memory_total,
                        memory_used=memory_used,
                        utilization=0,  # OpenCL没有直接获取利用率的方法
                        temperature=0,   # OpenCL没有直接获取温度的方法
                        power_usage=0,  # OpenCL没有直接获取功耗的方法
                        compute_capability="OpenCL",
                        is_available=True
                    ))

                    logger.info(f"检测到OpenCL GPU: {name} (ID: {gpu_id})")

        except Exception as e:
            logger.error(f"检测OpenCL GPU时出错: {str(e)}")

        return gpu_list

    def update_gpu_info(self, gpu_info: GPUInfo) -> bool:
        """更新GPU信息"""
        if gpu_info.gpu_id.startswith("nvidia-"):
            # 优先使用nvidia-smi更新
            if self.nvidia_smi_available:
                return self._update_nvidia_gpu_smi(gpu_info)
            # 如果nvidia-smi不可用，尝试使用NVML
            elif self.nvml_available:
                return self._update_nvidia_gpu_nvml(gpu_info)
        elif gpu_info.gpu_id.startswith("opencl-"):
            return self._update_opencl_gpu(gpu_info)
        else:
            logger.warning(f"未知的GPU ID格式: {gpu_info.gpu_id}")
            return False

    def _update_nvidia_gpu_smi(self, gpu_info: GPUInfo) -> bool:
        """使用nvidia-smi更新NVIDIA GPU信息"""
        try:
            # 从ID中提取设备索引
            device_index = gpu_info.gpu_id.split('-')[1]

            # 获取GPU状态信息
            result = subprocess.run([
                'nvidia-smi', 
                f'--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits',
                f'--id={device_index}'
            ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True)

            parts = [part.strip() for part in result.stdout.strip().split(',')]

            if len(parts) >= 4:
                gpu_info.memory_used = int(parts[0])
                gpu_info.utilization = float(parts[1])
                gpu_info.temperature = float(parts[2]) if parts[2] != '[Not Supported]' else 0
                gpu_info.power_usage = float(parts[3].replace('W', '')) if parts[3] != '[Not Supported]' else 0

                # 更新时间戳
                gpu_info.last_update = time.time()

                return True

        except subprocess.CalledProcessError as e:
            logger.error(f"使用nvidia-smi更新GPU信息时出错: {str(e)}")
        except Exception as e:
            logger.error(f"解析nvidia-smi输出时出错: {str(e)}")

        return False

    def _update_nvidia_gpu_nvml(self, gpu_info: GPUInfo) -> bool:
        """使用NVML更新NVIDIA GPU信息"""
        if not self.nvml_available:
            return False

        try:
            # 从ID中提取设备索引
            device_index = int(gpu_info.gpu_id.split('-')[1])
            handle = self.pynvml.nvmlDeviceGetHandleByIndex(device_index)

            # 更新内存信息
            mem_info = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_info.memory_used = mem_info.used // (1024 * 1024)  # 转换为MB

            # 更新利用率
            util_rate = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_info.utilization = util_rate.gpu

            # 更新温度
            try:
                gpu_info.temperature = self.pynvml.nvmlDeviceGetTemperature(handle, self.pynvml.NVML_TEMPERATURE_GPU)
            except:
                pass

            # 更新功耗
            try:
                gpu_info.power_usage = self.pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # 转换为瓦特
            except:
                pass

            # 更新时间戳
            gpu_info.last_update = time.time()

            return True

        except Exception as e:
            logger.error(f"更新NVIDIA GPU信息时出错: {str(e)}")
            return False

    def _update_opencl_gpu(self, gpu_info: GPUInfo) -> bool:
        """更新OpenCL GPU信息"""
        # OpenCL API通常不提供实时状态信息，因此这里只更新时间戳
        gpu_info.last_update = time.time()
        return True

class DockerGPUDetector(LinuxGPUDetector):
    """Docker容器内GPU检测器"""

    def __init__(self):
        super().__init__()
        # 检查是否使用了nvidia-docker
        self.nvidia_docker_available = self._check_nvidia_docker()

    def _check_nvidia_docker(self) -> bool:
        """检查是否使用了nvidia-docker"""
        try:
            # 检查nvidia-smi是否在容器内可用
            result = subprocess.run(['nvidia-smi', '--version'], 
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE,
                                    check=True)

            # 检查设备目录
            if os.path.exists('/dev/nvidia0'):
                logger.info("检测到nvidia-docker环境")
                return True
            else:
                logger.warning("nvidia-smi可用，但未检测到GPU设备")
                return False

        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("nvidia-docker环境检测失败")
            return False

    def detect_gpus(self) -> List[GPUInfo]:
        """检测所有可用的GPU"""
        if not self.nvidia_docker_available:
            logger.warning("未在nvidia-docker环境中运行，无法检测GPU")
            return []

        return super().detect_gpus()

    def update_gpu_info(self, gpu_info: GPUInfo) -> bool:
        """更新GPU信息"""
        if not self.nvidia_docker_available:
            return False

        return super().update_gpu_info(gpu_info)

def detect_gpus() -> List[GPUInfo]:
    """检测系统中的所有GPU"""
    detector = BaseGPUDetector.get_detector()
    if detector is None:
        logger.error("无法获取GPU检测器")
        return []

    return detector.detect_gpus()

def update_gpu_info(gpu_info: GPUInfo) -> bool:
    """更新GPU信息"""
    detector = BaseGPUDetector.get_detector()
    if detector is None:
        logger.error("无法获取GPU检测器")
        return False

    return detector.update_gpu_info(gpu_info)

def print_gpu_report():
    """打印GPU资源报告"""
    gpu_list = detect_gpus()

    if not gpu_list:
        print("未检测到GPU")
        return

    print("GPU资源报告:")
    print("=" * 80)
    print(f"{'ID':<15} {'名称':<25} {'内存(MB)':<15} {'利用率':<10} {'温度(°C)':<10} {'功耗(W)':<10}")
    print("-" * 80)

    for gpu in gpu_list:
        print(f"{gpu.gpu_id:<15} {gpu.name:<25} {gpu.memory_used}/{gpu.memory_total:<15} "
              f"{gpu.utilization:.1f}%:<10 {gpu.temperature:.1f}:<10 {gpu.power_usage:.1f}:<10")

    print("=" * 80)

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 打印GPU报告
    print_gpu_report()
