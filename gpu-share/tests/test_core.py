#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
核心模块测试
"""

import pytest
import tempfile
import time
from unittest.mock import Mock, patch

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# 导入核心模块
from core.gpu_detector import GPUMonitor, get_gpu_detector
from core.task_scheduler import TaskScheduler, Task, TaskTypes, TaskPriority, TaskStatus

class TestGPUInfo:
    """GPU信息测试类"""

    def __init__(self, gpu_id="test-gpu", name="Test GPU", 
                 memory_total=4096, memory_used=1024, utilization=50.0,
                 temperature=65.0, power_usage=150.0, 
                 compute_capability="6.0", is_available=True):
        self.gpu_id = gpu_id
        self.name = name
        self.memory_total = memory_total
        self.memory_used = memory_used
        self.utilization = utilization
        self.temperature = temperature
        self.power_usage = power_usage
        self.compute_capability = compute_capability
        self.is_available = is_available

    def to_dict(self):
        """转换为字典"""
        return {
            "gpu_id": self.gpu_id,
            "name": self.name,
            "memory_total": self.memory_total,
            "memory_used": self.memory_used,
            "utilization": self.utilization,
            "temperature": self.temperature,
            "power_usage": self.power_usage,
            "compute_capability": self.compute_capability,
            "is_available": self.is_available
        }

class TestTask:
    """任务测试类"""

    def __init__(self, task_id="test-task", task_type="test_type", 
                 requirements={"memory": "2GB"}, priority=TaskPriority.NORMAL):
        self.task_id = task_id
        self.task_type = task_type
        self.requirements = requirements
        self.priority = priority
        self.status = TaskStatus.PENDING
        self.submit_time = time.time()
        self.start_time = None
        self.end_time = None
        self.assigned_gpu_id = None
        self.assigned_worker_id = None
        self.result = None
        self.error_message = None
        self.retry_count = 0
        self.max_retries = 3

    def complete(self, result=None):
        """完成任务"""
        self.status = TaskStatus.COMPLETED
        self.end_time = time.time()
        self.result = result

    def fail(self, error_message="Test error"):
        """任务失败"""
        self.status = TaskStatus.FAILED
        self.end_time = time.time()
        self.error_message = error_message

    def to_dict(self):
        """转换为字典"""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status.value,
            "priority": self.priority.value,
            "submit_time": self.submit_time,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "assigned_gpu_id": self.assigned_gpu_id,
            "assigned_worker_id": self.assigned_worker_id,
            "requirements": self.requirements,
            "result": self.result,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries
        }

class TestWorkerInfo:
    """Worker信息测试类"""

    def __init__(self, worker_id="test-worker", hostname="test-host", 
                 platform="test-platform", status="online"):
        self.worker_id = worker_id
        self.hostname = hostname
        self.platform = platform
        self.status = status
        self.last_heartbeat = time.time()
        self.gpus = [
            TestGPUInfo("test-gpu-1", "Test GPU 1", 4096, 1024, 50.0, 65.0, 150.0, "6.0", True),
            TestGPUInfo("test-gpu-2", "Test GPU 2", 2048, 512, 75.0, 70.0, 120.0, "5.0", True)
        ]

    def to_dict(self):
        """转换为字典"""
        return {
            "worker_id": self.worker_id,
            "hostname": self.hostname,
            "platform": self.platform,
            "status": self.status,
            "last_heartbeat": self.last_heartbeat,
            "gpu_count": len(self.gpus),
            "gpus": [gpu.to_dict() for gpu in self.gpus]
        }

class MockGPUDetector:
    """模拟GPU检测器"""

    def __init__(self):
        self.gpus = []

    def detect_gpus(self):
        """检测GPU"""
        return self.gpus

    def add_gpu(self, gpu_info):
        """添加GPU"""
        self.gpus.append(gpu_info)

    def update_gpu_info(self, gpu_info):
        """更新GPU信息"""
        for i, gpu in enumerate(self.gpus):
            if gpu.gpu_id == gpu_info.gpu_id:
                self.gpus[i] = gpu_info
                return True
        return False

    def get_gpu_info_by_id(self, gpu_id):
        """根据ID获取GPU信息"""
        for gpu in self.gpus:
            if gpu.gpu_id == gpu_id:
                return gpu
        return None

    def get_all_gpus(self):
        """获取所有GPU信息"""
        return self.gpus

class MockTaskScheduler:
    """模拟任务调度器"""

    def __init__(self):
        self.tasks = {}
        self.task_queue = []
        self.next_task_id = 1

    def submit_task(self, task_type, requirements, priority=TaskPriority.NORMAL):
        """提交任务"""
        task = TestTask(
            task_id=f"task-{self.next_task_id}",
            task_type=task_type,
            requirements=requirements,
            priority=priority
        )

        self.tasks[task.task_id] = task
        self.task_queue.append(task.task_id)
        self.next_task_id += 1

        return task.task_id

    def cancel_task(self, task_id):
        """取消任务"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.cancel()
            return True
        return False

    def get_task(self, task_id):
        """获取任务"""
        return self.tasks.get(task_id)

    def get_all_tasks(self):
        """获取所有任务"""
        return list(self.tasks.values())

    def get_task_result(self, task_id):
        """获取任务结果"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            if task.status == TaskStatus.COMPLETED:
                return task.result
        return None

    def handle_task_result(self, task_id, result, success, error_message):
        """处理任务结果"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            if success:
                task.complete(result)
            else:
                task.fail(error_message)
            return True
        return False

    def get_gpu_info_by_id(self, gpu_id):
        """根据ID获取GPU信息"""
        return None  # 模拟实现中没有GPU信息

class TestWorkerManager:
    """模拟Worker管理器"""

    def __init__(self):
        self.workers = {}

    def register_worker(self, worker_info):
        """注册Worker"""
        self.workers[worker_info["worker_id"]] = worker_info
        return True

    def update_worker_gpus(self, worker_id, gpu_list):
        """更新Worker GPU信息"""
        if worker_id in self.workers:
            self.workers[worker_id]["gpus"] = gpu_list
            return True
        return False

    def update_worker_heartbeat(self, worker_id):
        """更新Worker心跳"""
        if worker_id in self.workers:
            self.workers[worker_id]["last_heartbeat"] = time.time()
            return True
        return False

    def get_all_workers(self):
        """获取所有Worker信息"""
        return list(self.workers.values())

# 测试用例
class TestGPUDetector:
    """GPU检测器测试"""

    def test_detect_gpus(self):
        """测试GPU检测"""
        detector = MockGPUDetector()

        # 添加测试GPU
        detector.add_gpu(TestGPUInfo("test-gpu-1", "Test GPU 1", 4096, 1024, 50.0, 65.0, 150.0, "6.0", True))
        detector.add_gpu(TestGPUInfo("test-gpu-2", "Test GPU 2", 2048, 512, 75.0, 70.0, 120.0, "5.0", True))

        # 检测GPU
        gpus = detector.detect_gpus()

        # 验证结果
        assert len(gpus) == 2
        assert gpus[0].gpu_id == "test-gpu-1"
        assert gpus[0].name == "Test GPU 1"
        assert gpus[1].gpu_id == "test-gpu-2"
        assert gpus[1].name == "Test GPU 2"

    def test_update_gpu_info(self):
        """测试GPU信息更新"""
        detector = MockGPUDetector()

        # 添加测试GPU
        detector.add_gpu(TestGPUInfo("test-gpu-1", "Test GPU 1", 4096, 1024, 50.0, 65.0, 150.0, "6.0", True))

        # 更新GPU信息
        updated_gpu = TestGPUInfo("test-gpu-1", "Updated Test GPU 1", 4096, 2048, 75.0, 60.0, 70.0, 180.0, "6.0", True)
        assert detector.update_gpu_info(updated_gpu)

        # 验证更新结果
        gpu = detector.get_gpu_info_by_id("test-gpu-1")
        assert gpu is not None
        assert gpu.name == "Updated Test GPU 1"
        assert gpu.memory_used == 2048

class TestTaskScheduler:
    """任务调度器测试"""

    def test_submit_task(self):
        """测试任务提交"""
        scheduler = MockTaskScheduler()

        # 提交任务
        task_id = scheduler.submit_task("test_type", {"memory": "1GB"})

        # 验证任务提交
        assert task_id is not None
        assert scheduler.get_task(task_id) is not None
        assert scheduler.get_task(task_id).task_type == "test_type"
        assert scheduler.get_task(task_id).requirements["memory"] == "1GB"

        # 提交第二个任务
        task_id2 = scheduler.submit_task("test_type", {"memory": "2GB"}, TaskPriority.HIGH)

        # 验证优先级
        assert scheduler.get_task(task_id2).priority == TaskPriority.HIGH

        # 验证任务数量
        assert len(scheduler.get_all_tasks()) == 2

        # 提交第三个任务
        task_id3 = scheduler.submit_task("test_type", {"memory": "3GB"}, TaskPriority.URGENT)

        # 验证紧急优先级
        assert scheduler.get_task(task_id3).priority == TaskPriority.URGENT

        # 验证任务数量
        assert len(scheduler.get_all_tasks()) == 3

        return True

    def test_cancel_task(self):
        """测试任务取消"""
        scheduler = MockTaskScheduler()

        # 提交任务
        task_id = scheduler.submit_task("test_type", {"memory": "1GB"})

        # 取消任务
        assert scheduler.cancel_task(task_id)

        # 验证任务状态
        assert scheduler.get_task(task_id).status == TaskStatus.CANCELLED

        # 验证任务数量
        assert len(scheduler.get_all_tasks()) == 0

        return True

    def test_task_completion(self):
        """测试任务完成"""
        scheduler = MockTaskScheduler()

        # 提交任务
        task_id = scheduler.submit_task("test_type", {"memory": "1GB"})

        # 模拟任务完成
        result = {"output": "test result"}
        assert scheduler.handle_task_result(task_id, result, True, "")

        # 验证任务状态
        assert scheduler.get_task(task_id).status == TaskStatus.COMPLETED
        assert scheduler.get_task_result(task_id) == result

        # 验证任务数量
        assert len(scheduler.get_all_tasks()) == 1

        return True

    def test_task_failure(self):
        """测试任务失败"""
        scheduler = MockTaskScheduler()

        # 提交任务
        task_id = scheduler.submit_task("test_type", {"memory": "1GB"})

        # 模拟任务失败
        assert scheduler.handle_task_result(task_id, None, False, "Test error")

        # 验证任务状态
        assert scheduler.get_task(task_id).status == TaskStatus.FAILED
        assert scheduler.get_task(task_id).error_message == "Test error"

        # 验证任务数量
        assert len(scheduler.get_all_tasks()) == 1

        return True

class TestWorkerManager:
    """Worker管理器测试"""

    def test_register_worker(self):
        """测试Worker注册"""
        manager = TestWorkerManager()

        # 注册Worker
        worker_info = TestWorkerInfo()
        assert manager.register_worker(worker_info)

        # 验证Worker数量
        assert len(manager.get_all_workers()) == 1

        # 获取Worker信息
        retrieved_worker = manager.get_all_workers()[worker_info["worker_id"]]
        assert retrieved_worker["worker_id"] == worker_info["worker_id"]
        assert retrieved_worker["hostname"] == worker_info["hostname"]
        assert retrieved_worker["platform"] == worker_info["platform"]

        return True

    def test_update_worker_gpus(self):
        """测试更新Worker GPU信息"""
        manager = TestWorkerManager()

        # 注册Worker
        worker_info = TestWorkerInfo()
        manager.register_worker(worker_info)

        # 更新GPU信息
        gpu_list = [
            TestGPUInfo("test-gpu-1", "Updated Test GPU 1", 4096, 2048, 80.0, 70.0, 200.0, "6.0", True)
        ]

        assert manager.update_worker_gpus(worker_info["worker_id"], gpu_list)

        # 验证更新结果
        retrieved_worker = manager.get_all_workers()[worker_info["worker_id"]]
        assert len(retrieved_worker["gpus"]) == 1
        assert retrieved_worker["gpus"][0].name == "Updated Test GPU 1"
        assert retrieved_worker["gpus"][0].memory_used == 2048

        return True

    def test_update_worker_heartbeat(self):
        """测试更新Worker心跳"""
        manager = TestWorkerManager()

        # 注册Worker
        worker_info = TestWorkerInfo()
        manager.register_worker(worker_info)

        # 更新心跳
        assert manager.update_worker_heartbeat(worker_info["worker_id"])

        # 验证更新结果
        retrieved_worker = manager.get_all_workers()[worker_info["worker_id"]]
        current_time = time.time()
        assert abs(retrieved_worker["last_heartbeat"] - current_time) < 5  # 允许5秒误差

        return True

    def test_unregister_worker(self):
        """测试注销Worker"""
        manager = TestWorkerManager()

        # 注册Worker
        worker_info = TestWorkerInfo()
        manager.register_worker(worker_info)

        # 注销Worker
        assert manager.unregister_worker(worker_info["worker_id"])

        # 验证注销结果
        assert len(manager.get_all_workers()) == 0

        return True

# 运行测试
if __name__ == "__main__":
    pytest.main(['-v', 'tests/test_core.py'])
manager.register_worker(worker_info.to_dict())

        # 验证Worker数量
        assert len(manager.get_all_workers()) == 1

        # 验证Worker信息
        worker = manager.get_all_workers()[0]
        assert worker.worker_id == "test-worker"
        assert worker.hostname == "test-host"
        assert worker.platform == "test-platform"
        assert worker.status == "online"
        assert len(worker.gpus) == 2

        return True

    def test_update_worker_gpus(self):
        """测试Worker GPU更新"""
        manager = TestWorkerManager()

        # 注册Worker
        worker_info = TestWorkerInfo()
        manager.register_worker(worker_info.to_dict())

        # 更新GPU信息
        updated_gpus = [
            TestGPUInfo("test-gpu-1", "Updated Test GPU 1", 4096, 2048, 75.0, 60.0, 70.0, 180.0, "6.0", True),
            TestGPUInfo("test-gpu-2", "Updated Test GPU 2", 2048, 1024, 80.0, 65.0, 140.0, "5.0", True)
        ]

        assert manager.update_worker_gpus(worker_info.worker_id, updated_gpus)

        # 验证更新结果
        worker = manager.get_all_workers()[0]
        assert len(worker.gpus) == 2
        assert worker.gpus[0].memory_used == 2048
        assert worker.gpus[1].memory_used == 1024

        return True

    def test_worker_heartbeat(self):
        """测试Worker心跳"""
        manager = TestWorkerManager()

        # 注册Worker
        worker_info = TestWorkerInfo()
        manager.register_worker(worker_info.to_dict())

        # 更新心跳
        old_time = worker_info.last_heartbeat
        assert manager.update_worker_heartbeat(worker_info.worker_id)

        # 验证心跳更新
        worker = manager.get_all_workers()[0]
        assert worker.last_heartbeat > old_time

        return True

# 集成测试
class TestIntegration:
    """集成测试"""

    def test_device_registration(self):
        """测试设备注册流程"""
        # 这个测试应该模拟整个设备注册和任务分配流程
        # 包括GPU检测、Worker注册、任务提交和执行

        # 1. 创建模拟组件
        gpu_detector = MockGPUDetector()
        gpu_detector.add_gpu(TestGPUInfo("test-gpu-1", "Test GPU 1", 4096, 1024, 50.0, 65.0, 150.0, "6.0", True))

        scheduler = MockTaskScheduler()
        manager = TestWorkerManager()

        # 2. 注册Worker
        worker_info = TestWorkerInfo()
        manager.register_worker(worker_info.to_dict())

        # 3. 模拟任务提交和执行
        task_id = scheduler.submit_task("test_type", {"memory": "1GB"})

        # 模拟任务执行
        result = {"output": "test result"}
        assert scheduler.handle_task_result(task_id, result, True, "")

        # 4. 验证结果
        assert scheduler.get_task(task_id).status == TaskStatus.COMPLETED
        assert scheduler.get_task_result(task_id) == result

        return True
