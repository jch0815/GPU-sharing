#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
任务调度模块
"""

import json
import logging
import queue
import time
import threading
from enum import Enum
from typing import Dict, List, Optional, Callable, Any

logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    """任务优先级枚举"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3

class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"       # 等待中
    RUNNING = "running"       # 运行中
    COMPLETED = "completed"   # 已完成
    FAILED = "failed"         # 失败
    CANCELLED = "cancelled"   # 已取消
    TIMEOUT = "timeout"       # 超时

class Task:
    """任务类"""

    def __init__(self, task_id: str, task_type: str, parameters: Dict, 
                 priority: TaskPriority = TaskPriority.NORMAL, 
                 timeout: Optional[int] = None):
        self.task_id = task_id
        self.task_type = task_type
        self.parameters = parameters
        self.priority = priority
        self.timeout = timeout  # 超时时间(秒)，None表示无超时

        # 状态信息
        self.status = TaskStatus.PENDING
        self.submit_time = time.time()
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.assigned_gpu_id: Optional[str] = None
        self.assigned_worker_id: Optional[str] = None

        # 结果信息
        self.result: Optional[Dict] = None
        self.error_message: Optional[str] = None

        # 重试机制
        self.retry_count = 0
        self.max_retries = parameters.get("max_retries", 3)

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "parameters": self.parameters,
            "priority": self.priority.value,
            "timeout": self.timeout,
            "status": self.status.value,
            "submit_time": self.submit_time,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "assigned_gpu_id": self.assigned_gpu_id,
            "assigned_worker_id": self.assigned_worker_id,
            "result": self.result,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries
        }

    def to_json(self) -> str:
        """转换为JSON格式"""
        return json.dumps(self.to_dict())

    def start(self, gpu_id: str, worker_id: str):
        """开始执行任务"""
        self.status = TaskStatus.RUNNING
        self.start_time = time.time()
        self.assigned_gpu_id = gpu_id
        self.assigned_worker_id = worker_id

    def complete(self, result: Dict):
        """完成任务"""
        self.status = TaskStatus.COMPLETED
        self.end_time = time.time()
        self.result = result

    def fail(self, error_message: str):
        """任务失败"""
        self.status = TaskStatus.FAILED
        self.end_time = time.time()
        self.error_message = error_message

    def cancel(self):
        """取消任务"""
        self.status = TaskStatus.CANCELLED
        self.end_time = time.time()

    def timeout(self):
        """任务超时"""
        self.status = TaskStatus.TIMEOUT
        self.end_time = time.time()
        self.error_message = f"任务执行超时(超过{self.timeout}秒)"

    def can_retry(self) -> bool:
        """检查是否可以重试"""
        return self.retry_count < self.max_retries and self.status in [TaskStatus.FAILED, TaskStatus.TIMEOUT]

    def retry(self):
        """重试任务"""
        if not self.can_retry():
            return False

        self.retry_count += 1
        self.status = TaskStatus.PENDING
        self.start_time = None
        self.end_time = None
        self.assigned_gpu_id = None
        self.assigned_worker_id = None
        self.result = None
        self.error_message = None

        return True

    def get_execution_time(self) -> Optional[float]:
        """获取任务执行时间(秒)"""
        if self.start_time is None or self.end_time is None:
            return None
        return self.end_time - self.start_time

    def get_waiting_time(self) -> Optional[float]:
        """获取任务等待时间(秒)"""
        if self.start_time is None:
            return None
        return self.start_time - self.submit_time

class TaskQueue:
    """任务队列"""

    def __init__(self):
        self._queue = queue.PriorityQueue()
        self._tasks: Dict[str, Task] = {}  # task_id -> Task
        self._lock = threading.Lock()
        self._counter = 0  # 用于保持相同优先级任务的FIFO顺序

    def put(self, task: Task):
        """添加任务到队列"""
        with self._lock:
            # 使用负优先级值，因为PriorityQueue是最小堆
            priority = -task.priority.value

            # 添加计数器以保持相同优先级任务的FIFO顺序
            self._queue.put((priority, self._counter, task.task_id))
            self._tasks[task.task_id] = task
            self._counter += 1

            logger.debug(f"任务 {task.task_id} 已添加到队列，优先级: {task.priority.name}")

    def get(self) -> Optional[Task]:
        """从队列获取任务"""
        with self._lock:
            try:
                _, _, task_id = self._queue.get_nowait()
                task = self._tasks.get(task_id)

                if task:
                    logger.debug(f"从队列获取任务 {task_id}")

                return task
            except queue.Empty:
                return None

    def peek(self) -> Optional[Task]:
        """查看队列中的下一个任务，但不移除"""
        with self._lock:
            try:
                _, _, task_id = self._queue.queue[0]
                return self._tasks.get(task_id)
            except (IndexError, KeyError):
                return None

    def remove(self, task_id: str) -> bool:
        """从队列中移除任务"""
        with self._lock:
            if task_id not in self._tasks:
                return False

            # 创建新队列，排除要移除的任务
            new_queue = queue.PriorityQueue()
            removed = False

            while not self._queue.empty():
                try:
                    item = self._queue.get_nowait()
                    if item[2] == task_id:
                        removed = True
                    else:
                        new_queue.put(item)
                except queue.Empty:
                    break

            self._queue = new_queue

            if removed:
                del self._tasks[task_id]
                logger.debug(f"任务 {task_id} 已从队列中移除")

            return removed

    def get_task(self, task_id: str) -> Optional[Task]:
        """根据ID获取任务"""
        with self._lock:
            return self._tasks.get(task_id)

    def update_task(self, task_id: str, task: Task) -> bool:
        """更新任务"""
        with self._lock:
            if task_id not in self._tasks:
                return False

            self._tasks[task_id] = task
            return True

    def size(self) -> int:
        """获取队列大小"""
        return self._queue.qsize()

    def empty(self) -> bool:
        """检查队列是否为空"""
        return self._queue.empty()

    def get_all_tasks(self) -> List[Task]:
        """获取所有任务"""
        with self._lock:
            return list(self._tasks.values())

    def get_tasks_by_status(self, status: TaskStatus) -> List[Task]:
        """根据状态获取任务"""
        with self._lock:
            return [task for task in self._tasks.values() if task.status == status]

class TaskScheduler:
    """任务调度器"""

    def __init__(self, gpu_detector, task_timeout_check_interval: int = 10):
        self.gpu_detector = gpu_detector
        self.task_queue = TaskQueue()
        self.running_tasks: Dict[str, Task] = {}  # task_id -> Task

        # 调度参数
        self.task_timeout_check_interval = task_timeout_check_interval

        # 回调函数
        self.task_callbacks: Dict[str, List[Callable[[Task], None]]] = {
            "submitted": [],
            "started": [],
            "completed": [],
            "failed": [],
            "cancelled": [],
            "timeout": []
        }

        # 运行状态
        self.running = False
        self.scheduler_thread = None
        self.timeout_thread = None
        self.lock = threading.Lock()

    def start(self):
        """启动调度器"""
        with self.lock:
            if self.running:
                return

            self.running = True

            # 启动调度线程
            self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
            self.scheduler_thread.start()

            # 启动超时检查线程
            self.timeout_thread = threading.Thread(target=self._timeout_check_loop, daemon=True)
            self.timeout_thread.start()

            logger.info("任务调度器已启动")

    def stop(self):
        """停止调度器"""
        with self.lock:
            if not self.running:
                return

            self.running = False

            # 等待线程结束
            if self.scheduler_thread and self.scheduler_thread.is_alive():
                self.scheduler_thread.join(timeout=10)

            if self.timeout_thread and self.timeout_thread.is_alive():
                self.timeout_thread.join(timeout=10)

            logger.info("任务调度器已停止")

    def submit_task(self, task_type: str, parameters: Dict, 
                   priority: TaskPriority = TaskPriority.NORMAL, 
                   timeout: Optional[int] = None) -> str:
        """提交任务"""
        # 生成任务ID
        task_id = f"task_{int(time.time() * 1000)}"

        # 创建任务
        task = Task(task_id, task_type, parameters, priority, timeout)

        # 添加到队列
        self.task_queue.put(task)

        # 触发回调
        self._trigger_callbacks("submitted", task)

        logger.info(f"已提交任务 {task_id} (类型: {task_type}, 优先级: {priority.name})")

        return task_id

    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        # 检查是否在队列中
        task = self.task_queue.get_task(task_id)
        if task and task.status == TaskStatus.PENDING:
            # 从队列中移除
            if self.task_queue.remove(task_id):
                task.cancel()
                self._trigger_callbacks("cancelled", task)
                logger.info(f"已取消任务 {task_id}")
                return True

        # 检查是否在运行中
        if task_id in self.running_tasks:
            task = self.running_tasks[task_id]
            if task.status == TaskStatus.RUNNING:
                # TODO: 向Worker发送取消信号
                task.cancel()
                del self.running_tasks[task_id]
                self._trigger_callbacks("cancelled", task)
                logger.info(f"已取消运行中的任务 {task_id}")
                return True

        return False

    def get_task(self, task_id: str) -> Optional[Task]:
        """获取任务"""
        # 先检查队列中的任务
        task = self.task_queue.get_task(task_id)
        if task:
            return task

        # 再检查运行中的任务
        return self.running_tasks.get(task_id)

    def get_all_tasks(self) -> List[Task]:
        """获取所有任务"""
        queue_tasks = self.task_queue.get_all_tasks()
        running_tasks = list(self.running_tasks.values())
        return queue_tasks + running_tasks

    def get_tasks_by_status(self, status: TaskStatus) -> List[Task]:
        """根据状态获取任务"""
        queue_tasks = self.task_queue.get_tasks_by_status(status)
        running_tasks = [task for task in self.running_tasks.values() if task.status == status]
        return queue_tasks + running_tasks

    def add_task_callback(self, event: str, callback: Callable[[Task], None]):
        """添加任务事件回调"""
        if event in self.task_callbacks:
            self.task_callbacks[event].append(callback)

    def remove_task_callback(self, event: str, callback: Callable[[Task], None]):
        """移除任务事件回调"""
        if event in self.task_callbacks and callback in self.task_callbacks[event]:
            self.task_callbacks[event].remove(callback)

    def complete_task(self, task_id: str, result: Dict) -> bool:
        """完成任务"""
        if task_id not in self.running_tasks:
            return False

        task = self.running_tasks[task_id]
        task.complete(result)

        # 从运行任务列表中移除
        del self.running_tasks[task_id]

        # 触发回调
        self._trigger_callbacks("completed", task)

        logger.info(f"任务 {task_id} 已完成")
        return True

    def fail_task(self, task_id: str, error_message: str) -> bool:
        """任务失败"""
        if task_id not in self.running_tasks:
            return False

        task = self.running_tasks[task_id]
        task.fail(error_message)

        # 检查是否可以重试
        if task.can_retry():
            task.retry()
            self.task_queue.put(task)
            logger.info(f"任务 {task_id} 失败，将进行第 {task.retry_count} 次重试")
        else:
            # 从运行任务列表中移除
            del self.running_tasks[task_id]

            # 触发回调
            self._trigger_callbacks("failed", task)

            logger.error(f"任务 {task_id} 失败: {error_message}")

        return True

    def _scheduler_loop(self):
        """调度循环"""
        while self.running:
            try:
                self._schedule_tasks()
                time.sleep(1)  # 每秒检查一次
            except Exception as e:
                logger.error(f"调度循环出错: {str(e)}")
                time.sleep(1)  # 出错后短暂休眠

    def _schedule_tasks(self):
        """调度任务"""
        # 获取可用GPU
        available_gpus = self.gpu_detector.get_gpu_info()
        if not available_gpus:
            return

        # 获取可用GPU ID列表
        available_gpu_ids = [gpu.gpu_id for gpu in available_gpus if gpu.is_available]
        if not available_gpu_ids:
            return

        # 调度任务
        while not self.task_queue.empty():
            # 获取下一个任务
            task = self.task_queue.get()
            if not task:
                break

            # 选择最合适的GPU
            suitable_gpu_id = self._select_gpu_for_task(task, available_gpus)
            if not suitable_gpu_id:
                # 没有合适的GPU，将任务放回队列
                self.task_queue.put(task)
                break

            # 分配任务到GPU
            self._assign_task_to_gpu(task, suitable_gpu_id)

    def _select_gpu_for_task(self, task: Task, available_gpus: List) -> Optional[str]:
        """为任务选择合适的GPU"""
        # 获取任务需求
        required_memory = task.parameters.get("memory", 0)  # MB
        min_compute_capability = task.parameters.get("compute_capability", "0.0")

        # 筛选满足条件的GPU
        suitable_gpus = []
        for gpu in available_gpus:
            if not gpu.is_available:
                continue

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
        best_gpu = min(suitable_gpus, key=lambda g: g.utilization)
        return best_gpu.gpu_id

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

    def _assign_task_to_gpu(self, task: Task, gpu_id: str):
        """分配任务到GPU"""
        # 更新任务状态
        task.start(gpu_id, "worker-placeholder")  # TODO: 获取实际的Worker ID

        # 添加到运行任务列表
        self.running_tasks[task.task_id] = task

        # 触发回调
        self._trigger_callbacks("started", task)

        # TODO: 向Worker发送任务执行请求
        logger.info(f"任务 {task.task_id} 已分配到GPU {gpu_id}")

    def _timeout_check_loop(self):
        """超时检查循环"""
        while self.running:
            try:
                self._check_task_timeouts()
                time.sleep(self.task_timeout_check_interval)
            except Exception as e:
                logger.error(f"超时检查循环出错: {str(e)}")
                time.sleep(1)  # 出错后短暂休眠

    def _check_task_timeouts(self):
        """检查任务超时"""
        current_time = time.time()
        timeout_tasks = []

        # 检查运行中的任务
        for task_id, task in list(self.running_tasks.items()):
            if task.timeout is not None and task.start_time is not None:
                elapsed = current_time - task.start_time
                if elapsed > task.timeout:
                    timeout_tasks.append(task)

        # 处理超时任务
        for task in timeout_tasks:
            task.timeout()

            # 从运行任务列表中移除
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]

            # 检查是否可以重试
            if task.can_retry():
                task.retry()
                self.task_queue.put(task)
                logger.info(f"任务 {task.task_id} 超时，将进行第 {task.retry_count} 次重试")
            else:
                # 触发回调
                self._trigger_callbacks("timeout", task)
                logger.error(f"任务 {task.task_id} 执行超时")

    def _trigger_callbacks(self, event: str, task: Task):
        """触发回调函数"""
        if event in self.task_callbacks:
            for callback in self.task_callbacks[event]:
                try:
                    callback(task)
                except Exception as e:
                    logger.error(f"任务事件回调出错: {str(e)}")

# 任务执行器
class TaskExecutor:
    """任务执行器"""

    def __init__(self, task_scheduler: TaskScheduler):
        self.task_scheduler = task_scheduler
        self.task_handlers: Dict[str, Callable] = {}

    def register_task_handler(self, task_type: str, handler: Callable):
        """注册任务处理器"""
        self.task_handlers[task_type] = handler
        logger.info(f"已注册任务处理器: {task_type}")

    def execute_task(self, task: Task) -> Dict:
        """执行任务"""
        handler = self.task_handlers.get(task.task_type)
        if not handler:
            raise ValueError(f"未找到任务类型 {task.task_type} 的处理器")

        try:
            # 执行任务处理器
            result = handler(task)
            return {"success": True, "result": result}
        except Exception as e:
            logger.error(f"执行任务 {task.task_id} 时出错: {str(e)}")
            return {"success": False, "error": str(e)}

# 任务类型定义
class TaskTypes:
    """任务类型常量"""
    MATRIX_MULTIPLICATION = "matrix_multiplication"  # 矩阵乘法
    MODEL_INFERENCE = "model_inference"               # 模型推理
    DATA_PROCESSING = "data_processing"               # 数据处理
    CUSTOM_KERNEL = "custom_kernel"                   # 自定义内核
