#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gRPC服务端
"""

import os
import sys
import time
import logging
from concurrent import futures
import grpc

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

# 导入gRPC模块
from communication.grpc_service import gpu_share_pb2
from communication.grpc_service import gpu_share_pb2_grpc
from core.scheduler import GPUScheduler
from workers.manager import WorkerManager

logger = logging.getLogger(__name__)

class GPUServer(gpu_share_pb2_grpc.GPUShareServiceServicer):
    """gRPC服务实现"""

    def __init__(self, scheduler, worker_manager):
        self.scheduler = scheduler
        self.worker_manager = worker_manager

    def RegisterWorker(self, request, context):
        """注册Worker"""
        logger.info(f"注册Worker: {request.worker_id}")

        # 创建Worker信息
        worker_info = {
            "worker_id": request.worker_id,
            "hostname": request.hostname,
            "platform": request.platform,
            "status": "online",
            "last_heartbeat": time.time(),
            "gpus": []
        }

        # 添加GPU信息
        for gpu in request.gpus:
            worker_info["gpus"].append({
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

        # 注册Worker
        success = self.worker_manager.register_worker(worker_info)

        # 返回响应
        return gpu_share_pb2.RegisterWorkerResponse(success=success)

    def UpdateWorkerGpus(self, request, context):
        """更新Worker GPU信息"""
        logger.info(f"更新Worker GPU信息: {request.worker_id}")

        # 创建GPU列表
        gpu_list = []
        for gpu in request.gpus:
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

        # 更新GPU信息
        success = self.worker_manager.update_worker_gpus(request.worker_id, gpu_list)

        # 返回响应
        return gpu_share_pb2.UpdateWorkerGpusResponse(success=success)

    def UpdateWorkerHeartbeat(self, request, context):
        """更新Worker心跳"""
        logger.debug(f"更新Worker心跳: {request.worker_id}")

        # 更新心跳
        success = self.worker_manager.update_worker_heartbeat(request.worker_id)

        # 返回响应
        return gpu_share_pb2.UpdateWorkerHeartbeatResponse(success=success)

    def SubmitTask(self, request, context):
        """提交任务"""
        logger.info(f"提交任务: {request.task_type}")

        # 创建任务参数
        parameters = {}
        for key, value in request.parameters.items():
            parameters[key] = value

        # 提交任务
        task_id = self.scheduler.submit_task(request.task_type, parameters)

        # 返回响应
        return gpu_share_pb2.SubmitTaskResponse(task_id=task_id)

    def CancelTask(self, request, context):
        """取消任务"""
        logger.info(f"取消任务: {request.task_id}")

        # 取消任务
        success = self.scheduler.cancel_task(request.task_id)

        # 返回响应
        return gpu_share_pb2.CancelTaskResponse(success=success)

    def GetTaskResult(self, request, context):
        """获取任务结果"""
        logger.info(f"获取任务结果: {request.task_id}")

        # 获取任务结果
        result = self.scheduler.get_task_result(request.task_id)

        # 创建响应
        response = gpu_share_pb2.GetTaskResultResponse()
        if result is not None:
            response.success = True
            for key, value in result.items():
                response.result[key] = str(value)
        else:
            response.success = False
            response.error_message = "任务未完成或不存在"

        return response

    def GetAvailableGpus(self, request, context):
        """获取可用GPU"""
        logger.info("获取可用GPU")

        # 获取GPU信息
        gpus = []
        for gpu in self.scheduler.get_all_gpus():
            if gpu.is_available:
                gpus.append({
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

        # 创建响应
        response = gpu_share_pb2.GetAvailableGpusResponse()
        for gpu in gpus:
            gpu_info = response.gpus.add()
            gpu_info.gpu_id = gpu["gpu_id"]
            gpu_info.name = gpu["name"]
            gpu_info.memory_total = gpu["memory_total"]
            gpu_info.memory_used = gpu["memory_used"]
            gpu_info.utilization = gpu["utilization"]
            gpu_info.temperature = gpu["temperature"]
            gpu_info.power_usage = gpu["power_usage"]
            gpu_info.compute_capability = gpu["compute_capability"]
            gpu_info.is_available = gpu["is_available"]

        return response

def serve():
    """启动gRPC服务"""
    # 创建调度器和Worker管理器
    scheduler = GPUScheduler()
    worker_manager = WorkerManager()

    # 启动调度器
    scheduler.start()

    # 启动Worker管理器
    worker_manager.start()

    # 创建gRPC服务器
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    # 添加服务
    gpu_share_pb2_grpc.add_GPUShareServiceServicer_to_server(
        GPUServer(scheduler, worker_manager), server
    )

    # 绑定端口
    port = os.environ.get('GRPC_PORT', '50051')
    server.add_insecure_port(f'[::]:{port}')

    # 启动服务器
    server.start()
    logger.info(f"gRPC服务器已启动，端口: {port}")

    try:
        while True:
            time.sleep(86400)  # 一天
    except KeyboardInterrupt:
        logger.info("收到终止信号，正在关闭gRPC服务器...")

    # 停止服务器
    server.stop(0)

    # 停止组件
    worker_manager.stop()
    scheduler.stop()

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    serve()
