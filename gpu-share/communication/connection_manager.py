#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
连接管理模块
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Callable

import grpc

# 导入生成的protobuf文件
try:
    import communication.grpc_service.gpu_share_pb2 as pb2
    import communication.grpc_service.gpu_share_pb2_grpc as pb2_grpc
except ImportError:
    # 如果protobuf文件尚未生成，创建占位符
    pb2 = None
    pb2_grpc = None

logger = logging.getLogger(__name__)

class ConnectionManager:
    """连接管理器"""

    def __init__(self, heartbeat_interval=30, connection_timeout=60):
        self.heartbeat_interval = heartbeat_interval  # 心跳间隔(秒)
        self.connection_timeout = connection_timeout  # 连接超时(秒)

        # 连接信息
        self.connections: Dict[str, Dict] = {}  # connection_id -> connection_info
        self.worker_connections: Dict[str, str] = {}  # worker_id -> connection_id

        # 回调函数
        self.callbacks: Dict[str, List[Callable]] = {
            "connected": [],
            "disconnected": [],
            "heartbeat": []
        }

        # 运行状态
        self.running = False
        self.heartbeat_thread = None
        self.lock = threading.Lock()

    def start(self):
        """启动连接管理器"""
        with self.lock:
            if self.running:
                return

            self.running = True
            self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self.heartbeat_thread.start()

            logger.info("连接管理器已启动")

    def stop(self):
        """停止连接管理器"""
        with self.lock:
            if not self.running:
                return

            self.running = False

            if self.heartbeat_thread and self.heartbeat_thread.is_alive():
                self.heartbeat_thread.join(timeout=10)

            # 关闭所有连接
            for connection_id in list(self.connections.keys()):
                self.close_connection(connection_id)

            logger.info("连接管理器已停止")

    def add_connection(self, connection_id: str, worker_id: str, stub, context) -> bool:
        """添加连接"""
        with self.lock:
            if connection_id in self.connections:
                logger.warning(f"连接已存在: {connection_id}")
                return False

            # 创建连接信息
            connection_info = {
                "connection_id": connection_id,
                "worker_id": worker_id,
                "stub": stub,
                "context": context,
                "last_heartbeat": time.time(),
                "status": "connected"
            }

            # 添加到连接列表
            self.connections[connection_id] = connection_info
            self.worker_connections[worker_id] = connection_id

            # 触发连接回调
            self._trigger_callbacks("connected", connection_info)

            logger.info(f"已添加连接: {connection_id} (Worker: {worker_id})")
            return True

    def remove_connection(self, connection_id: str) -> bool:
        """移除连接"""
        with self.lock:
            if connection_id not in self.connections:
                logger.warning(f"连接不存在: {connection_id}")
                return False

            # 获取连接信息
            connection_info = self.connections[connection_id]
            worker_id = connection_info["worker_id"]

            # 从连接列表中移除
            del self.connections[connection_id]
            if worker_id in self.worker_connections:
                del self.worker_connections[worker_id]

            # 关闭连接
            try:
                connection_info["context"].cancel()
            except Exception as e:
                logger.error(f"关闭连接出错: {str(e)}")

            # 触发断开连接回调
            self._trigger_callbacks("disconnected", connection_info)

            logger.info(f"已移除连接: {connection_id} (Worker: {worker_id})")
            return True

    def close_connection(self, connection_id: str) -> bool:
        """关闭连接"""
        with self.lock:
            if connection_id not in self.connections:
                return False

            # 更新连接状态
            self.connections[connection_id]["status"] = "closing"

            # 移除连接
            return self.remove_connection(connection_id)

    def get_connection(self, connection_id: str) -> Optional[Dict]:
        """获取连接信息"""
        with self.lock:
            return self.connections.get(connection_id)

    def get_connection_by_worker(self, worker_id: str) -> Optional[Dict]:
        """根据Worker ID获取连接信息"""
        with self.lock:
            connection_id = self.worker_connections.get(worker_id)
            if connection_id:
                return self.connections.get(connection_id)
            return None

    def get_all_connections(self) -> Dict[str, Dict]:
        """获取所有连接信息"""
        with self.lock:
            return self.connections.copy()

    def update_heartbeat(self, connection_id: str) -> bool:
        """更新连接心跳"""
        with self.lock:
            if connection_id not in self.connections:
                return False

            # 更新心跳时间
            self.connections[connection_id]["last_heartbeat"] = time.time()

            # 触发心跳回调
            self._trigger_callbacks("heartbeat", self.connections[connection_id])

            return True

    def add_callback(self, event: str, callback: Callable):
        """添加连接事件回调"""
        with self.lock:
            if event in self.callbacks:
                self.callbacks[event].append(callback)

    def remove_callback(self, event: str, callback: Callable):
        """移除连接事件回调"""
        with self.lock:
            if event in self.callbacks and callback in self.callbacks[event]:
                self.callbacks[event].remove(callback)

    def _trigger_callbacks(self, event: str, connection_info: Dict):
        """触发连接事件回调"""
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                try:
                    callback(connection_info)
                except Exception as e:
                    logger.error(f"连接事件回调出错: {str(e)}")

    def _heartbeat_loop(self):
        """心跳循环"""
        while self.running:
            try:
                current_time = time.time()
                timeout_connections = []

                with self.lock:
                    # 检查连接超时
                    for connection_id, connection_info in self.connections.items():
                        if current_time - connection_info["last_heartbeat"] > self.connection_timeout:
                            timeout_connections.append(connection_id)

                # 处理超时连接
                for connection_id in timeout_connections:
                    logger.warning(f"连接超时: {connection_id}")
                    self.close_connection(connection_id)

                # 休眠
                time.sleep(self.heartbeat_interval)

            except Exception as e:
                logger.error(f"心跳循环出错: {str(e)}")
                time.sleep(5)  # 出错后短暂休眠

class GRPCConnection:
    """gRPC连接"""

    def __init__(self, host: str, port: int, connection_manager: ConnectionManager):
        self.host = host
        self.port = port
        self.connection_manager = connection_manager
        self.channel = None
        self.stub = None
        self.connected = False

    def connect(self, worker_id: str) -> bool:
        """连接到服务器"""
        if self.connected:
            return True

        try:
            # 创建gRPC通道
            self.channel = grpc.insecure_channel(f'{self.host}:{self.port}')

            # 创建存根
            if pb2_grpc is not None:
                self.stub = pb2_grpc.GPUMasterServiceStub(self.channel)

            # 生成连接ID
            import uuid
            connection_id = str(uuid.uuid4())

            # 添加到连接管理器
            self.connected = self.connection_manager.add_connection(
                connection_id, worker_id, self.stub, None
            )

            if self.connected:
                logger.info(f"已连接到服务器: {self.host}:{self.port}")

            return self.connected

        except Exception as e:
            logger.error(f"连接服务器失败: {str(e)}")
            return False

    def disconnect(self):
        """断开连接"""
        if not self.connected:
            return

        try:
            # 关闭通道
            if self.channel:
                self.channel.close()
                self.channel = None

            self.stub = None
            self.connected = False

            logger.info("已断开与服务器的连接")

        except Exception as e:
            logger.error(f"断开连接出错: {str(e)}")

    def send_heartbeat(self) -> bool:
        """发送心跳"""
        if not self.connected or not self.stub:
            return False

        try:
            # 创建心跳请求
            request = pb2.HeartbeatRequest(
                worker_id=self.connection_manager.get_connection_by_worker(worker_id)["worker_id"],
                timestamp=int(time.time())
            )

            # 发送心跳
            response = self.stub.Heartbeat(request)

            if response.success:
                # 更新连接管理器中的心跳时间
                connection_id = self.connection_manager.get_connection_by_worker(worker_id)["connection_id"]
                self.connection_manager.update_heartbeat(connection_id)
                return True
            else:
                logger.warning(f"心跳失败: {response.message}")
                return False

        except Exception as e:
            logger.error(f"发送心跳出错: {str(e)}")
            return False

    def register_worker(self, worker_info: Dict) -> bool:
        """注册Worker"""
        if not self.connected or not self.stub:
            return False

        try:
            # 创建GPU信息列表
            gpu_list = []
            for gpu in worker_info.get("gpus", []):
                gpu_info = pb2.GPUInfo(
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
                gpu_list.append(gpu_info)

            # 创建Worker信息
            worker_info_pb = pb2.WorkerInfo(
                worker_id=worker_info["worker_id"],
                hostname=worker_info["hostname"],
                platform=worker_info["platform"],
                gpu_list=gpu_list,
                metadata=worker_info.get("metadata", {})
            )

            # 发送注册请求
            response = self.stub.RegisterWorker(pb2.RegisterRequest(worker_info=worker_info_pb))

            if response.success:
                logger.info(f"Worker注册成功: {worker_info['worker_id']}")
                return True
            else:
                logger.error(f"Worker注册失败: {response.message}")
                return False

        except Exception as e:
            logger.error(f"注册Worker出错: {str(e)}")
            return False

    def update_gpu_status(self, worker_id: str, gpu_list: List[Dict]) -> bool:
        """更新GPU状态"""
        if not self.connected or not self.stub:
            return False

        try:
            # 创建GPU信息列表
            gpu_info_list = []
            for gpu in gpu_list:
                gpu_info = pb2.GPUInfo(
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
                gpu_info_list.append(gpu_info)

            # 发送更新请求
            response = self.stub.UpdateGPUStatus(pb2.UpdateGPUStatusRequest(
                worker_id=worker_id,
                gpu_list=gpu_info_list
            ))

            if response.success:
                return True
            else:
                logger.warning(f"更新GPU状态失败: {response.message}")
                return False

        except Exception as e:
            logger.error(f"更新GPU状态出错: {str(e)}")
            return False

    def report_task_result(self, worker_id: str, task_result: Dict) -> bool:
        """报告任务结果"""
        if not self.connected or not self.stub:
            return False

        try:
            # 创建任务结果
            task_result_pb = pb2.TaskResult(
                task_id=task_result["task_id"],
                status=pb2.TaskStatus.Value(task_result["status"]),
                result=task_result.get("result", {}),
                error_message=task_result.get("error_message"),
                retry_count=task_result.get("retry_count", 0)
            )

            # 发送结果报告
            response = self.stub.ReportTaskResult(pb2.ReportTaskResultRequest(
                worker_id=worker_id,
                task_result=task_result_pb
            ))

            if response.success:
                logger.info(f"任务结果报告成功: {task_result['task_id']}")
                return True
            else:
                logger.warning(f"任务结果报告失败: {response.message}")
                return False

        except Exception as e:
            logger.error(f"报告任务结果出错: {str(e)}")
            return False
