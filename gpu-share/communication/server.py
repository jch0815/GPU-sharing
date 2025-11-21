#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
通信服务器模块
"""

import logging
import threading
import time
import json
from typing import Dict, List, Optional, Callable
from concurrent import futures

import grpc
import websockets
import asyncio

# 导入生成的protobuf文件
try:
    import communication.grpc_service.gpu_share_pb2 as pb2
    import communication.grpc_service.gpu_share_pb2_grpc as pb2_grpc
except ImportError:
    # 如果protobuf文件尚未生成，创建占位符
    pb2 = None
    pb2_grpc = None

logger = logging.getLogger(__name__)

class GPUServicer(pb2_grpc.GPUShareServicer):
    """gRPC服务实现"""

    def __init__(self, scheduler):
        self._scheduler = scheduler
        self._workers: Dict[str, pb2.WorkerInfo] = {}  # worker_id -> WorkerInfo

    def RegisterWorker(self, request, context):
        """注册Worker"""
        worker_id = request.worker_id
        logger.info(f"收到Worker注册请求: {worker_id}")

        # 保存Worker信息
        self._workers[worker_id] = request

        # 注册Worker提供的所有GPU
        for gpu_info in request.gpu_list:
            self._scheduler.register_gpu(GPUInfo(
                gpu_id=gpu_info.gpu_id,
                worker_id=worker_id,
                name=gpu_info.name,
                memory_total=gpu_info.memory_total,
                memory_used=gpu_info.memory_used,
                utilization=gpu_info.utilization,
                temperature=gpu_info.temperature,
                power_usage=gpu_info.power_usage,
                compute_capability=gpu_info.compute_capability,
                is_available=gpu_info.is_available
            ))

        # 返回注册结果
        return pb2.RegisterResponse(success=True, message="Worker注册成功")

    def UpdateGPUStatus(self, request, context):
        """更新GPU状态"""
        worker_id = request.worker_id
        gpu_id = request.gpu_id

        # 更新GPU状态
        self._scheduler.update_gpu_status(gpu_id, GPUInfo(
            gpu_id=gpu_id,
            worker_id=worker_id,
            name=request.name,
            memory_total=request.memory_total,
            memory_used=request.memory_used,
            utilization=request.utilization,
            temperature=request.temperature,
            power_usage=request.power_usage,
            compute_capability=request.compute_capability,
            is_available=request.is_available
        ))

        return pb2.UpdateResponse(success=True)

    def ReportTaskResult(self, request, context):
        """报告任务结果"""
        task_id = request.task_id
        status = request.status
        result = json.loads(request.result) if request.result else None

        # 更新任务状态和结果
        if task_id in self._scheduler._tasks:
            task = self._scheduler._tasks[task_id]
            if status == "completed":
                task.status = TaskStatus.COMPLETED
                task.result = result
            elif status == "failed":
                task.status = TaskStatus.FAILED

            task.end_time = time.time()

            # 释放GPU
            if task.assigned_gpu_id:
                self._scheduler._release_gpu(task.assigned_gpu_id)

        return pb2.ResultResponse(success=True)

class WebSocketServer:
    """WebSocket服务器"""

    def __init__(self, scheduler, host='0.0.0.0', port=8765):
        self._scheduler = scheduler
        self._host = host
        self._port = port
        self._server = None
        self._clients = set()
        self._loop = None
        self._thread = None
        self._running = False

    def start(self):
        """启动WebSocket服务器"""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()
        logger.info(f"WebSocket服务器已启动: ws://{self._host}:{self._port}")

    def stop(self):
        """停止WebSocket服务器"""
        if not self._running:
            return

        self._running = False

        if self._loop:
            self._loop.call_soon_threadsafe(self._stop_server)

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

        logger.info("WebSocket服务器已停止")

    def _run_server(self):
        """运行WebSocket服务器"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        start_server = websockets.serve(
            self._handle_client,
            self._host,
            self._port
        )

        self._loop.run_until_complete(start_server)
        self._loop.run_until_complete(self._server.wait_closed())
        self._loop.close()

    async def _stop_server(self):
        """停止WebSocket服务器"""
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def _handle_client(self, websocket, path):
        """处理客户端连接"""
        self._clients.add(websocket)
        logger.info(f"新的WebSocket客户端连接: {websocket.remote_address}")

        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._process_message(websocket, data)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "无效的JSON格式"
                    }))
                except Exception as e:
                    logger.error(f"处理WebSocket消息时出错: {str(e)}")
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": f"处理消息时出错: {str(e)}"
                    }))
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket客户端断开连接: {websocket.remote_address}")
        finally:
            self._clients.remove(websocket)

    async def _process_message(self, websocket, data):
        """处理客户端消息"""
        message_type = data.get("type")

        if message_type == "get_gpu_list":
            # 获取GPU列表
            gpu_list = []
            for gpu in self._scheduler.get_all_gpus():
                gpu_list.append({
                    "gpu_id": gpu.gpu_id,
                    "worker_id": gpu.worker_id,
                    "name": gpu.name,
                    "memory_total": gpu.memory_total,
                    "memory_used": gpu.memory_used,
                    "utilization": gpu.utilization,
                    "temperature": gpu.temperature,
                    "power_usage": gpu.power_usage,
                    "compute_capability": gpu.compute_capability,
                    "is_available": gpu.is_available
                })

            await websocket.send(json.dumps({
                "type": "gpu_list",
                "data": gpu_list
            }))

        elif message_type == "get_task_list":
            # 获取任务列表
            task_list = []
            for task in self._scheduler._tasks.values():
                task_list.append({
                    "task_id": task.task_id,
                    "task_type": task.task_type,
                    "status": task.status.value,
                    "assigned_gpu_id": task.assigned_gpu_id,
                    "worker_id": task.worker_id,
                    "submit_time": task.submit_time,
                    "start_time": task.start_time,
                    "end_time": task.end_time
                }))

            await websocket.send(json.dumps({
                "type": "task_list",
                "data": task_list
            }))

        elif message_type == "submit_task":
            # 提交任务
            task_data = data.get("task", {})
            task = GPUTask(
                task_id=task_data.get("task_id", f"task_{int(time.time())}"),
                task_type=task_data.get("task_type", "inference"),
                requirements=task_data.get("requirements", {})
            )

            self._scheduler.submit_task(task)

            await websocket.send(json.dumps({
                "type": "task_submitted",
                "task_id": task.task_id
            }))

        else:
            await websocket.send(json.dumps({
                "type": "error",
                "message": f"未知消息类型: {message_type}"
            }))

class CommunicationServer:
    """通信服务器"""

    def __init__(self, scheduler, grpc_host='0.0.0.0', grpc_port=50051, ws_host='0.0.0.0', ws_port=8765):
        self._scheduler = scheduler
        self._grpc_host = grpc_host
        self._grpc_port = grpc_port
        self._ws_host = ws_host
        self._ws_port = ws_port

        # gRPC服务器
        self._grpc_server = None

        # WebSocket服务器
        self._ws_server = WebSocketServer(scheduler, ws_host, ws_port)

        self._running = False

    def start(self):
        """启动通信服务器"""
        if self._running:
            return

        self._running = True

        # 启动gRPC服务器
        self._start_grpc_server()

        # 启动WebSocket服务器
        self._ws_server.start()

        logger.info("通信服务器已启动")

    def stop(self):
        """停止通信服务器"""
        if not self._running:
            return

        self._running = False

        # 停止gRPC服务器
        if self._grpc_server:
            self._grpc_server.stop(grace=2)
            self._grpc_server = None

        # 停止WebSocket服务器
        self._ws_server.stop()

        logger.info("通信服务器已停止")

    def _start_grpc_server(self):
        """启动gRPC服务器"""
        if pb2 is None or pb2_grpc is None:
            logger.warning("gRPC protobuf文件未生成，跳过gRPC服务器启动")
            return

        self._grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

        # 添加服务
        servicer = GPUServicer(self._scheduler)
        pb2_grpc.add_GPUShareServicer_to_server(servicer, self._grpc_server)

        # 绑定端口
        self._grpc_server.add_insecure_port(f'{self._grpc_host}:{self._grpc_port}')

        # 启动服务器
        self._grpc_server.start()
        logger.info(f"gRPC服务器已启动: {self._grpc_host}:{self._grpc_port}")

# 导入必要的类，避免循环导入
from core.scheduler import GPUInfo, GPUTask, TaskStatus
