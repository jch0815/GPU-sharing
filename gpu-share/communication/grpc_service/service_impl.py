#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gRPC服务实现
"""

import logging
from concurrent import futures
import threading
import time

# 导入生成的protobuf文件
try:
    import communication.grpc_service.gpu_share_pb2 as pb2
    import communication.grpc_service.gpu_share_pb2_grpc as pb2_grpc
except ImportError:
    # 如果protobuf文件尚未生成，创建占位符
    pb2 = None
    pb2_grpc = None

logger = logging.getLogger(__name__)

class GPUMasterServiceImpl(pb2_grpc.GPUMasterServiceServicer):
    """GPU主节点服务实现"""

    def __init__(self, scheduler, worker_manager, file_transfer_manager):
        self.scheduler = scheduler
        self.worker_manager = worker_manager
        self.file_transfer_manager = file_transfer_manager

    def RegisterWorker(self, request, context):
        """注册Worker"""
        worker_id = request.worker_info.worker_id
        logger.info(f"收到Worker注册请求: {worker_id}")

        # 创建Worker信息
        worker_info = {
            "worker_id": worker_id,
            "hostname": request.worker_info.hostname,
            "platform": request.worker_info.platform,
            "status": "online",
            "last_heartbeat": time.time(),
            "gpus": []
        }

        # 添加GPU信息
        for gpu_info in request.worker_info.gpu_list:
            worker_info["gpus"].append({
                "gpu_id": gpu_info.gpu_id,
                "name": gpu_info.name,
                "memory_total": gpu_info.memory_total,
                "memory_used": gpu_info.memory_used,
                "utilization": gpu_info.utilization,
                "temperature": gpu_info.temperature,
                "power_usage": gpu_info.power_usage,
                "compute_capability": gpu_info.compute_capability,
                "is_available": gpu_info.is_available
            })

        # 添加元数据
        for key, value in request.worker_info.metadata.items():
            worker_info.setdefault("metadata", {})[key] = value

        # 注册Worker
        success = self.worker_manager.register_worker(worker_info)

        # 返回注册结果
        return pb2.RegisterResponse(
            success=success,
            message="Worker注册成功" if success else "Worker注册失败"
        )

    def UpdateGPUStatus(self, request, context):
        """更新GPU状态"""
        worker_id = request.worker_id

        # 创建GPU信息列表
        gpu_list = []
        for gpu_info in request.gpu_list:
            gpu_list.append({
                "gpu_id": gpu_info.gpu_id,
                "name": gpu_info.name,
                "memory_total": gpu_info.memory_total,
                "memory_used": gpu_info.memory_used,
                "utilization": gpu_info.utilization,
                "temperature": gpu_info.temperature,
                "power_usage": gpu_info.power_usage,
                "compute_capability": gpu_info.compute_capability,
                "is_available": gpu_info.is_available
            })

        # 更新GPU状态
        success = self.worker_manager.update_worker_gpus(worker_id, gpu_list)

        # 更新Worker心跳
        if success:
            self.worker_manager.update_worker_heartbeat(worker_id)

        # 返回更新结果
        return pb2.UpdateResponse(
            success=success,
            message="GPU状态更新成功" if success else "GPU状态更新失败"
        )

    def Heartbeat(self, request, context):
        """处理心跳"""
        worker_id = request.worker_id

        # 更新Worker心跳
        success = self.worker_manager.update_worker_heartbeat(worker_id)

        # 返回心跳响应
        return pb2.HeartbeatResponse(
            success=success,
            timestamp=int(time.time())
        )

    def ReportTaskResult(self, request, context):
        """报告任务结果"""
        worker_id = request.worker_id
        task_result = {
            "task_id": request.task_result.task_id,
            "status": request.task_result.status,
            "result": dict(request.task_result.result),
            "error_message": request.task_result.error_message,
            "retry_count": request.task_result.retry_count
        }

        # 处理任务结果
        success = self.scheduler.handle_task_result(
            task_result["task_id"], 
            task_result["result"], 
            task_result["status"] == pb2.TaskStatus.COMPLETED,
            task_result["error_message"]
        )

        # 返回结果响应
        return pb2.ResultResponse(
            success=success,
            message="任务结果处理成功" if success else "任务结果处理失败"
        )

    def UploadFile(self, request_iterator, context):
        """处理文件上传"""
        # 获取第一个请求，包含文件信息
        first_request = next(request_iterator)
        file_id = first_request.file_id
        file_name = first_request.file_name
        file_size = first_request.file_size
        total_chunks = first_request.total_chunks
        is_compressed = first_request.is_compressed

        # 开始接收文件
        file_path = self.file_transfer_manager.start_download(
            file_id, file_name, file_size, total_chunks, compress=is_compressed
        )

        if not file_path:
            yield pb2.FileTransferResponse(
                file_id=file_id,
                success=False,
                message="无法开始文件下载"
            )
            return

        # 处理第一个请求的块数据
        if first_request.chunk_data:
            if not self.file_transfer_manager.add_download_chunk(
                file_id, first_request.chunk_index, first_request.chunk_data
            ):
                yield pb2.FileTransferResponse(
                    file_id=file_id,
                    success=False,
                    message=f"无法添加块 {first_request.chunk_index}"
                )
                return

        # 处理剩余的请求
        for request in request_iterator:
            if not self.file_transfer_manager.add_download_chunk(
                file_id, request.chunk_index, request.chunk_data
            ):
                yield pb2.FileTransferResponse(
                    file_id=file_id,
                    success=False,
                    message=f"无法添加块 {request.chunk_index}"
                )
                return

        # 检查下载是否完成
        transfer_info = self.file_transfer_manager.get_transfer_info(file_id)
        if transfer_info and transfer_info["status"] == "completed":
            yield pb2.FileTransferResponse(
                file_id=file_id,
                success=True,
                message="文件上传成功"
            )
        else:
            yield pb2.FileTransferResponse(
                file_id=file_id,
                success=False,
                message="文件上传不完整"
            )

    def DownloadFile(self, request, context):
        """处理文件下载"""
        file_id = request.file_id
        file_name = request.file_name

        # 获取传输信息
        transfer_info = self.file_transfer_manager.get_transfer_info(file_id)
        if not transfer_info:
            yield pb2.FileTransferResponse(
                file_id=file_id,
                success=False,
                message="找不到文件传输信息"
            )
            return

        # 发送所有块
        for chunk_index in range(transfer_info["total_chunks"]):
            # 获取块数据
            chunk_data = self.file_transfer_manager.get_upload_chunk(file_id, chunk_index)
            if not chunk_data:
                yield pb2.FileTransferResponse(
                    file_id=file_id,
                    success=False,
                    message=f"无法获取块 {chunk_index}"
                )
                return

            # 发送块
            yield pb2.FileTransferRequest(
                file_id=file_id,
                file_name=file_name,
                file_size=transfer_info["file_size"],
                chunk_data=chunk_data,
                chunk_index=chunk_index,
                total_chunks=transfer_info["total_chunks"],
                is_compressed=transfer_info["compress"]
            )

            logger.debug(f"已发送块 {chunk_index}/{transfer_info['total_chunks']}")

        # 完成上传
        self.file_transfer_manager.complete_upload(file_id)

class GPUWorkerServiceImpl(pb2_grpc.GPUWorkerServiceServicer):
    """GPU Worker节点服务实现"""

    def __init__(self, task_executor, file_transfer_manager):
        self.task_executor = task_executor
        self.file_transfer_manager = file_transfer_manager
        self.running_tasks = {}
        self.lock = threading.Lock()

    def AssignTask(self, request, context):
        """分配任务"""
        worker_id = request.worker_id
        task_info = request.task_info

        # 创建任务对象
        from core.task_scheduler import Task, TaskStatus, TaskPriority

        task = Task(
            task_id=task_info.task_id,
            task_type=task_info.task_type,
            parameters=dict(task_info.parameters),
            priority=TaskPriority(task_info.priority),
            timeout=task_info.timeout
        )
        task.max_retries = task_info.max_retries

        # 启动任务执行
        try:
            # 添加到运行任务列表
            with self.lock:
                self.running_tasks[task.task_id] = task

            # 执行任务
            result = self.task_executor.execute_task(task)

            # 创建任务结果
            task_result = {
                "task_id": task.task_id,
                "status": pb2.TaskStatus.COMPLETED if result["success"] else pb2.TaskStatus.FAILED,
                "result": result.get("result", {}),
                "error_message": result.get("error", ""),
                "retry_count": task.retry_count
            }

            # 从运行任务列表中移除
            with self.lock:
                if task.task_id in self.running_tasks:
                    del self.running_tasks[task.task_id]

            # 返回任务分配响应
            return pb2.TaskAssignmentResponse(
                success=True,
                message="任务分配成功"
            )

        except Exception as e:
            logger.error(f"分配任务出错: {str(e)}")

            # 从运行任务列表中移除
            with self.lock:
                if task.task_id in self.running_tasks:
                    del self.running_tasks[task.task_id]

            # 返回错误响应
            return pb2.TaskAssignmentResponse(
                success=False,
                message=f"任务分配失败: {str(e)}"
            )

    def CancelTask(self, request, context):
        """取消任务"""
        task_id = request.task_id

        # 检查任务是否存在
        with self.lock:
            if task_id not in self.running_tasks:
                return pb2.TaskAssignmentResponse(
                    success=False,
                    message=f"任务不存在或未在运行: {task_id}"
                )

            task = self.running_tasks[task_id]

        # 取消任务
        try:
            # TODO: 实现任务取消逻辑
            # 这里应该调用任务执行器的取消方法

            # 从运行任务列表中移除
            with self.lock:
                if task_id in self.running_tasks:
                    del self.running_tasks[task_id]

            return pb2.TaskAssignmentResponse(
                success=True,
                message="任务取消成功"
            )

        except Exception as e:
            logger.error(f"取消任务出错: {str(e)}")
            return pb2.TaskAssignmentResponse(
                success=False,
                message=f"任务取消失败: {str(e)}"
            )

    def UploadFile(self, request_iterator, context):
        """处理文件上传"""
        # 获取第一个请求，包含文件信息
        first_request = next(request_iterator)
        file_id = first_request.file_id
        file_name = first_request.file_name
        file_size = first_request.file_size
        total_chunks = first_request.total_chunks
        is_compressed = first_request.is_compressed

        # 开始接收文件
        file_path = self.file_transfer_manager.start_download(
            file_id, file_name, file_size, total_chunks, compress=is_compressed
        )

        if not file_path:
            yield pb2.FileTransferResponse(
                file_id=file_id,
                success=False,
                message="无法开始文件下载"
            )
            return

        # 处理第一个请求的块数据
        if first_request.chunk_data:
            if not self.file_transfer_manager.add_download_chunk(
                file_id, first_request.chunk_index, first_request.chunk_data
            ):
                yield pb2.FileTransferResponse(
                    file_id=file_id,
                    success=False,
                    message=f"无法添加块 {first_request.chunk_index}"
                )
                return

        # 处理剩余的请求
        for request in request_iterator:
            if not self.file_transfer_manager.add_download_chunk(
                file_id, request.chunk_index, request.chunk_data
            ):
                yield pb2.FileTransferResponse(
                    file_id=file_id,
                    success=False,
                    message=f"无法添加块 {request.chunk_index}"
                )
                return

        # 检查下载是否完成
        transfer_info = self.file_transfer_manager.get_transfer_info(file_id)
        if transfer_info and transfer_info["status"] == "completed":
            yield pb2.FileTransferResponse(
                file_id=file_id,
                success=True,
                message="文件上传成功"
            )
        else:
            yield pb2.FileTransferResponse(
                file_id=file_id,
                success=False,
                message="文件上传不完整"
            )

    def DownloadFile(self, request, context):
        """处理文件下载"""
        file_id = request.file_id
        file_name = request.file_name

        # 获取传输信息
        transfer_info = self.file_transfer_manager.get_transfer_info(file_id)
        if not transfer_info:
            yield pb2.FileTransferResponse(
                file_id=file_id,
                success=False,
                message="找不到文件传输信息"
            )
            return

        # 发送所有块
        for chunk_index in range(transfer_info["total_chunks"]):
            # 获取块数据
            chunk_data = self.file_transfer_manager.get_upload_chunk(file_id, chunk_index)
            if not chunk_data:
                yield pb2.FileTransferResponse(
                    file_id=file_id,
                    success=False,
                    message=f"无法获取块 {chunk_index}"
                )
                return

            # 发送块
            yield pb2.FileTransferRequest(
                file_id=file_id,
                file_name=file_name,
                file_size=transfer_info["file_size"],
                chunk_data=chunk_data,
                chunk_index=chunk_index,
                total_chunks=transfer_info["total_chunks"],
                is_compressed=transfer_info["compress"]
            )

            logger.debug(f"已发送块 {chunk_index}/{transfer_info['total_chunks']}")

        # 完成上传
        self.file_transfer_manager.complete_upload(file_id)
