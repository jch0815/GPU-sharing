#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPU资源调度器
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"       # 等待中
    RUNNING = "running"       # 运行中
    COMPLETED = "completed"   # 已完成
    FAILED = "failed"         # 失败
    CANCELLED = "cancelled"   # 已取消

@dataclass
class GPUTask:
    """GPU任务数据类"""
    task_id: str
    task_type: str           # 任务类型，如"inference", "training"
    requirements: Dict       # 资源需求，如{"memory": "4GB", "compute_capability": "6.0"}
    status: TaskStatus = TaskStatus.PENDING
    assigned_gpu_id: Optional[str] = None
    worker_id: Optional[str] = None
    submit_time: float = 0.0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Optional[Dict] = None

@dataclass
class GPUInfo:
    """GPU信息数据类"""
    gpu_id: str
    worker_id: str
    name: str                # GPU型号
    memory_total: int        # 总内存(MB)
    memory_used: int         # 已用内存(MB)
    utilization: float       # 利用率(0-100)
    temperature: float       # 温度(°C)
    power_usage: float       # 功耗(W)
    compute_capability: str  # 计算能力
    is_available: bool = True
    last_update: float = 0.0

class GPUScheduler:
    """GPU资源调度器"""

    def __init__(self):
        self._gpu_resources: Dict[str, GPUInfo] = {}  # gpu_id -> GPUInfo
        self._tasks: Dict[str, GPUTask] = {}          # task_id -> GPUTask
        self._task_queue: List[str] = []              # 待处理任务ID队列
        self._lock = threading.RLock()
        self._scheduler_thread = None
        self._running = False
        self._scheduling_interval = 5  # 调度间隔(秒)

    def start(self):
        """启动调度器"""
        with self._lock:
            if self._running:
                return

            self._running = True
            self._scheduler_thread = threading.Thread(target=self._scheduling_loop, daemon=True)
            self._scheduler_thread.start()
            logger.info("GPU调度器已启动")

    def stop(self):
        """停止调度器"""
        with self._lock:
            if not self._running:
                return

            self._running = False
            if self._scheduler_thread and self._scheduler_thread.is_alive():
                self._scheduler_thread.join(timeout=10)
            logger.info("GPU调度器已停止")

    def register_gpu(self, gpu_info: GPUInfo) -> bool:
        """注册GPU资源"""
        with self._lock:
            if gpu_info.gpu_id in self._gpu_resources:
                logger.warning(f"GPU {gpu_info.gpu_id} 已存在，更新信息")

            gpu_info.last_update = time.time()
            self._gpu_resources[gpu_info.gpu_id] = gpu_info
            logger.info(f"已注册GPU {gpu_info.gpu_id} ({gpu_info.name}) 来自Worker {gpu_info.worker_id}")
            return True

    def update_gpu_status(self, gpu_id: str, gpu_info: GPUInfo) -> bool:
        """更新GPU状态"""
        with self._lock:
            if gpu_id not in self._gpu_resources:
                logger.warning(f"未知GPU ID: {gpu_id}")
                return False

            # 更新时间戳
            gpu_info.last_update = time.time()
            self._gpu_resources[gpu_id] = gpu_info
            return True

    def unregister_gpu(self, gpu_id: str) -> bool:
        """注销GPU资源"""
        with self._lock:
            if gpu_id not in self._gpu_resources:
                logger.warning(f"尝试注销不存在的GPU: {gpu_id}")
                return False

            # 检查是否有正在运行的任务
            running_tasks = [task_id for task_id, task in self._tasks.items() 
                             if task.status == TaskStatus.RUNNING and task.assigned_gpu_id == gpu_id]

            if running_tasks:
                logger.warning(f"GPU {gpu_id} 上有运行中的任务 {running_tasks}，无法注销")
                return False

            del self._gpu_resources[gpu_id]
            logger.info(f"已注销GPU {gpu_id}")
            return True

    def submit_task(self, task: GPUTask) -> str:
        """提交任务"""
        with self._lock:
            task.submit_time = time.time()
            self._tasks[task.task_id] = task
            self._task_queue.append(task.task_id)
            logger.info(f"已提交任务 {task.task_id} (类型: {task.task_type})")
            return task.task_id

    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        with self._lock:
            if task_id not in self._tasks:
                logger.warning(f"尝试取消不存在的任务: {task_id}")
                return False

            task = self._tasks[task_id]

            if task.status == TaskStatus.RUNNING:
                # 通知Worker停止任务
                logger.info(f"任务 {task_id} 正在运行，发送停止信号")
                # TODO: 实现向Worker发送停止信号的逻辑

            task.status = TaskStatus.CANCELLED
            task.end_time = time.time()

            # 从队列中移除（如果还在队列中）
            if task_id in self._task_queue:
                self._task_queue.remove(task_id)

            # 释放GPU资源
            if task.assigned_gpu_id:
                self._release_gpu(task.assigned_gpu_id)

            logger.info(f"已取消任务 {task_id}")
            return True

    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """获取任务状态"""
        with self._lock:
            if task_id not in self._tasks:
                return None
            return self._tasks[task_id].status

    def get_task_result(self, task_id: str) -> Optional[Dict]:
        """获取任务结果"""
        with self._lock:
            if task_id not in self._tasks:
                return None

            task = self._tasks[task_id]
            if task.status == TaskStatus.COMPLETED:
                return task.result
            return None

    def get_available_gpus(self) -> List[GPUInfo]:
        """获取可用GPU列表"""
        with self._lock:
            current_time = time.time()
            # 过滤出可用且最近更新过的GPU
            return [gpu for gpu in self._gpu_resources.values() 
                   if gpu.is_available and (current_time - gpu.last_update) < 60]

    def get_all_gpus(self) -> List[GPUInfo]:
        """获取所有GPU列表"""
        with self._lock:
            return list(self._gpu_resources.values())

    def get_running_tasks(self) -> List[GPUTask]:
        """获取正在运行的任务列表"""
        with self._lock:
            return [task for task in self._tasks.values() if task.status == TaskStatus.RUNNING]

    def get_pending_tasks(self) -> List[GPUTask]:
        """获取等待中的任务列表"""
        with self._lock:
            return [task for task in self._tasks.values() if task.status == TaskStatus.PENDING]

    def _scheduling_loop(self):
        """调度循环"""
        while self._running:
            try:
                self._schedule_tasks()
                self._check_gpu_health()
                time.sleep(self._scheduling_interval)
            except Exception as e:
                logger.error(f"调度循环出错: {str(e)}")
                time.sleep(1)  # 出错后短暂休眠

    def _schedule_tasks(self):
        """调度任务"""
        with self._lock:
            if not self._task_queue:
                return

            # 获取可用GPU
            available_gpus = self.get_available_gpus()
            if not available_gpus:
                return

            # 遍历任务队列
            for task_id in self._task_queue[:]:  # 复制队列以避免在迭代中修改
                if task_id not in self._tasks:
                    self._task_queue.remove(task_id)
                    continue

                task = self._tasks[task_id]
                if task.status != TaskStatus.PENDING:
                    self._task_queue.remove(task_id)
                    continue

                # 查找合适的GPU
                suitable_gpu = self._find_suitable_gpu(task, available_gpus)
                if suitable_gpu:
                    # 分配任务到GPU
                    self._assign_task_to_gpu(task, suitable_gpu)
                    self._task_queue.remove(task_id)

    def _find_suitable_gpu(self, task: GPUTask, available_gpus: List[GPUInfo]) -> Optional[GPUInfo]:
        """查找适合任务的GPU"""
        # 获取任务需求
        required_memory = int(task.requirements.get("memory", "0").replace("GB", "")) * 1024  # GB转MB
        min_compute_capability = task.requirements.get("compute_capability", "0.0")

        # 筛选满足条件的GPU
        suitable_gpus = []
        for gpu in available_gpus:
            # 检查内存是否足够
            if gpu.memory_total - gpu.memory_used < required_memory:
                continue

            # 检查计算能力是否满足
            if self._compare_compute_capability(gpu.compute_capability, min_compute_capability) < 0:
                continue

            suitable_gpus.append(gpu)

        if not suitable_gpus:
            return None

        # 选择利用率最低的GPU
        return min(suitable_gpus, key=lambda g: g.utilization)

    def _compare_compute_capability(self, current: str, required: str) -> int:
        """比较计算能力版本，返回1表示current>required，0表示相等，-1表示current<required"""
        try:
            curr_major, curr_minor = map(int, current.split('.'))
            req_major, req_minor = map(int, required.split('.'))

            if curr_major > req_major:
                return 1
            elif curr_major < req_major:
                return -1
            else:  # 主版本相同
                if curr_minor > req_minor:
                    return 1
                elif curr_minor < req_minor:
                    return -1
                else:
                    return 0
        except (ValueError, AttributeError):
            # 如果解析失败，假设当前版本满足需求
            return 1

    def _assign_task_to_gpu(self, task: GPUTask, gpu: GPUInfo):
        """分配任务到GPU"""
        task.status = TaskStatus.RUNNING
        task.assigned_gpu_id = gpu.gpu_id
        task.worker_id = gpu.worker_id
        task.start_time = time.time()

        # 更新GPU状态
        gpu.is_available = False

        logger.info(f"任务 {task.task_id} 已分配到GPU {gpu.gpu_id} (Worker: {gpu.worker_id})")

        # TODO: 向Worker发送任务执行请求
        # 这里应该调用通信模块向对应的Worker发送任务

    def _release_gpu(self, gpu_id: str):
        """释放GPU资源"""
        if gpu_id in self._gpu_resources:
            self._gpu_resources[gpu_id].is_available = True
            logger.info(f"已释放GPU {gpu_id}")

    def _check_gpu_health(self):
        """检查GPU健康状态"""
        current_time = time.time()
        timeout = 60  # 60秒无更新视为超时

        with self._lock:
            for gpu_id, gpu in list(self._gpu_resources.items()):
                if current_time - gpu.last_update > timeout:
                    logger.warning(f"GPU {gpu_id} (Worker: {gpu.worker_id}) 状态更新超时，标记为不可用")
                    gpu.is_available = False

                    # 检查是否有运行中的任务
                    running_tasks = [task_id for task_id, task in self._tasks.items() 
                                    if task.status == TaskStatus.RUNNING and task.assigned_gpu_id == gpu_id]

                    if running_tasks:
                        logger.error(f"GPU {gpu_id} 超时，但其上有运行中的任务 {running_tasks}")
                        # TODO: 处理超时任务，可能需要重新调度
