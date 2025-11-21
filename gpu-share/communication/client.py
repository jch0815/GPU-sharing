#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
通信客户端模块
"""

import logging
import time
import json
import threading
import queue
from typing import Dict, List, Optional, Callable

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

class GRPCClient:
    """gRPC客户端"""

    def __init__(self, server_host, server_port):
        self.server_host = server_host
        self.server_port = server_port
        self.channel = None
        self.stub = None
        self.connected = False

    def connect(self):
        """连接到服务器"""
        if self.connected:
            return

        try:
            # 创建gRPC通道
            self.channel = grpc.insecure_channel(f'{self.server_host}:{self.server_port}')

            # 创建存根
            if pb2_grpc is not None:
                self.stub = pb2_grpc.GPUShareStub(self.channel)

            self.connected = True
            logger.info(f"gRPC客户端已连接到服务器: {self.server_host}:{self.server_port}")
        except Exception as e:
            logger.error(f"gRPC客户端连接失败: {str(e)}")
            self.connected = False

    def disconnect(self):
        """断开连接"""
        if not self.connected:
            return

        if self.channel:
            self.channel.close()
            self.channel = None
            self.stub = None

        self.connected = False
        logger.info("gRPC客户端已断开连接")

    def register_worker(self, worker_id, hostname, platform, gpu_info):
        """注册Worker"""
        if not self.connected or self.stub is None:
            logger.error("gRPC客户端未连接，无法注册Worker")
            return False

        try:
            # 创建GPU信息列表
            gpu_list = []
            for gpu in gpu_info:
                gpu_info_pb = pb2.GPUInfo(
                    gpu_id=gpu["gpu_id"],
                    name=gpu["name"],
                    memory_total=gpu["memory_total"],
                    memory_used=gpu["memory_used"],
                    utilization=gpu["utilization"],
                    temperature=gpu["temperature"],
                    power_usage=gpu["power_usage"],
                    compute_capability=gpu["compute_capability"],
                    is_available=gpu["is_available"]
                )
                gpu_list.append(gpu_info_pb)

            # 创建Worker信息
            worker_info = pb2.WorkerInfo(
                worker_id=worker_id,
                hostname=hostname,
                platform=platform,
                gpu_list=gpu_list
            )

            # 发送注册请求
            response = self.stub.RegisterWorker(worker_info)

            if response.success:
                logger.info(f"Worker {worker_id} 注册成功")
                return True
            else:
                logger.error(f"Worker {worker_id} 注册失败: {response.message}")
                return False

        except Exception as e:
            logger.error(f"注册Worker时出错: {str(e)}")
            return False

    def update_gpu_status(self, worker_id, gpu_info):
        """更新GPU状态"""
        if not self.connected or self.stub is None:
            logger.error("gRPC客户端未连接，无法更新GPU状态")
            return False

        try:
            # 更新每个GPU的状态
            for gpu in gpu_info:
                gpu_info_pb = pb2.GPUInfo(
                    gpu_id=gpu["gpu_id"],
                    name=gpu["name"],
                    memory_total=gpu["memory_total"],
                    memory_used=gpu["memory_used"],
                    utilization=gpu["utilization"],
                    temperature=gpu["temperature"],
                    power_usage=gpu["power_usage"],
                    compute_capability=gpu["compute_capability"],
                    is_available=gpu["is_available"]
                )

                update_request = pb2.UpdateGPUStatusRequest(
                    worker_id=worker_id,
                    gpu_id=gpu["gpu_id"],
                    **gpu_info_pb.__dict__
                )

                response = self.stub.UpdateGPUStatus(update_request)

                if not response.success:
                    logger.warning(f"更新GPU {gpu['gpu_id']} 状态失败")
                    return False

            return True

        except Exception as e:
            logger.error(f"更新GPU状态时出错: {str(e)}")
            return False

    def report_task_result(self, worker_id, task_id, status, result=None):
        """报告任务结果"""
        if not self.connected or self.stub is None:
            logger.error("gRPC客户端未连接，无法报告任务结果")
            return False

        try:
            result_json = json.dumps(result) if result else ""

            result_request = pb2.ReportTaskResultRequest(
                worker_id=worker_id,
                task_id=task_id,
                status=status,
                result=result_json
            )

            response = self.stub.ReportTaskResult(result_request)

            if response.success:
                logger.info(f"任务 {task_id} 结果报告成功")
                return True
            else:
                logger.error(f"任务 {task_id} 结果报告失败")
                return False

        except Exception as e:
            logger.error(f"报告任务结果时出错: {str(e)}")
            return False

class WebSocketClient:
    """WebSocket客户端"""

    def __init__(self, server_host, server_port):
        self.server_host = server_host
        self.server_port = server_port
        self.websocket = None
        self.connected = False
        self.loop = None
        self.thread = None
        self.running = False
        self.message_queue = queue.Queue()

    def connect(self):
        """连接到服务器"""
        if self.connected:
            return

        self.running = True
        self.thread = threading.Thread(target=self._run_client, daemon=True)
        self.thread.start()

        # 等待连接建立
        timeout = 10
        start_time = time.time()
        while not self.connected and (time.time() - start_time) < timeout:
            time.sleep(0.1)

        if not self.connected:
            logger.error("WebSocket客户端连接超时")
            self.running = False
            return False

        return True

    def disconnect(self):
        """断开连接"""
        if not self.connected:
            return

        self.running = False

        if self.loop:
            self.loop.call_soon_threadsafe(self._close_websocket)

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)

        logger.info("WebSocket客户端已断开连接")

    def send_message(self, message):
        """发送消息"""
        if not self.connected:
            logger.error("WebSocket客户端未连接，无法发送消息")
            return False

        try:
            self.message_queue.put(message)
            return True
        except Exception as e:
            logger.error(f"发送WebSocket消息时出错: {str(e)}")
            return False

    def _run_client(self):
        """运行WebSocket客户端"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            uri = f"ws://{self.server_host}:{self.server_port}"
            self._loop.run_until_complete(self._connect_to_server(uri))
            self._loop.run_until_complete(self._message_loop())
        except Exception as e:
            logger.error(f"WebSocket客户端运行出错: {str(e)}")
        finally:
            self._loop.close()

    async def _connect_to_server(self, uri):
        """连接到WebSocket服务器"""
        try:
            self.websocket = await websockets.connect(uri)
            self.connected = True
            logger.info(f"WebSocket客户端已连接到服务器: {uri}")
        except Exception as e:
            logger.error(f"WebSocket客户端连接失败: {str(e)}")
            self.connected = False

    async def _close_websocket(self):
        """关闭WebSocket连接"""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            self.connected = False

    async def _message_loop(self):
        """消息循环"""
        while self.running and self.connected:
            try:
                # 处理发送队列
                while not self.message_queue.empty():
                    message = self.message_queue.get_nowait()
                    await self.websocket.send(json.dumps(message))

                # 接收消息
                try:
                    message = await asyncio.wait_for(self.websocket.recv(), timeout=1.0)
                    await self._process_message(json.loads(message))
                except asyncio.TimeoutError:
                    pass  # 超时是正常的，继续循环

            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket连接已关闭")
                self.connected = False
                break
            except Exception as e:
                logger.error(f"WebSocket消息循环出错: {str(e)}")
                break

    async def _process_message(self, data):
        """处理接收到的消息"""
        message_type = data.get("type")

        if message_type == "task_assignment":
            # 处理任务分配
            task_id = data.get("task_id")
            task_data = data.get("task", {})

            logger.info(f"收到任务分配: {task_id}")

            # TODO: 执行任务
            # 这里应该调用任务执行模块

            # 模拟任务执行
            await self._execute_task(task_id, task_data)

        else:
            logger.warning(f"未知消息类型: {message_type}")

    async def _execute_task(self, task_id, task_data):
        """执行任务"""
        # 模拟任务执行
        await asyncio.sleep(5)  # 模拟任务执行时间

        # 报告任务结果
        result_message = {
            "type": "task_result",
            "task_id": task_id,
            "status": "completed",
            "result": {"output": "任务执行完成"}
        }

        self.send_message(result_message)

class CommunicationClient:
    """通信客户端"""

    def __init__(self, server_host, server_port, worker_id, gpu_info, update_callback=None):
        self.server_host = server_host
        self.server_port = server_port
        self.worker_id = worker_id
        self.gpu_info = gpu_info
        self.update_callback = update_callback

        # 创建gRPC和WebSocket客户端
        self.grpc_client = GRPCClient(server_host, server_port)
        self.ws_client = WebSocketClient(server_host, server_port + 1)  # WebSocket端口+1

        # 系统信息
        self.hostname = None
        self.platform = None

        # 运行状态
        self.running = False
        self.heartbeat_thread = None

    def start(self):
        """启动客户端"""
        if self.running:
            return

        logger.info(f"启动通信客户端: {self.worker_id}")

        # 获取系统信息
        self._get_system_info()

        # 连接到服务器
        self.grpc_client.connect()
        self.ws_client.connect()

        # 注册Worker
        self._register_worker()

        # 启动心跳线程
        self.running = True
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()

        logger.info("通信客户端已启动")

    def stop(self):
        """停止客户端"""
        if not self.running:
            return

        logger.info("正在停止通信客户端...")
        self.running = False

        # 断开连接
        self.grpc_client.disconnect()
        self.ws_client.disconnect()

        # 等待心跳线程结束
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.heartbeat_thread.join(timeout=5)

        logger.info("通信客户端已停止")

    def _get_system_info(self):
        """获取系统信息"""
        import socket
        import platform

        self.hostname = socket.gethostname()
        self.platform = platform.system()

        logger.info(f"系统信息: {self.hostname} ({self.platform})")

    def _register_worker(self):
        """注册Worker"""
        if self.grpc_client.connected:
            success = self.grpc_client.register_worker(
                worker_id=self.worker_id,
                hostname=self.hostname,
                platform=self.platform,
                gpu_info=self.gpu_info
            )

            if not success:
                logger.error("Worker注册失败")

    def _heartbeat_loop(self):
        """心跳循环"""
        while self.running:
            try:
                # 更新GPU信息
                if self.update_callback:
                    self.gpu_info = self.update_callback()

                # 发送心跳
                if self.grpc_client.connected:
                    self.grpc_client.update_gpu_status(self.worker_id, self.gpu_info)

                # 休眠
                time.sleep(10)  # 每10秒发送一次心跳
            except Exception as e:
                logger.error(f"心跳循环出错: {str(e)}")
                time.sleep(5)  # 出错后短暂休眠在停止通信客户端...")
        self.running = False

        # 断开连接
        self.grpc_client.disconnect()
        self.ws_client.disconnect()

        # 等待心跳线程结束
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.heartbeat_thread.join(timeout=5)

        logger.info("通信客户端已停止")

    def _get_system_info(self):
        """获取系统信息"""
        import socket
        import platform

        self.hostname = socket.gethostname()
        self.platform = platform.system()

    def _register_worker(self):
        """注册Worker"""
        self.grpc_client.register_worker(
            worker_id=self.worker_id,
            hostname=self.hostname,
            platform=self.platform,
            gpu_info=self.gpu_info
        )

    def _heartbeat_loop(self):
        """心跳循环"""
        while self.running:
            try:
                # 更新GPU信息
                if self.update_callback:
                    self.gpu_info = self.update_callback()

                # 通过gRPC更新GPU状态
                self.grpc_client.update_gpu_status(self.worker_id, self.gpu_info)

                # 通过WebSocket发送心跳
                self.ws_client.send_message({
                    "type": "heartbeat",
                    "worker_id": self.worker_id
                })

                # 等待下一次心跳
                time.sleep(10)

            except Exception as e:
                logger.error(f"心跳循环出错: {str(e)}")
                time.sleep(5)  # 出错后短暂休眠
