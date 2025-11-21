#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gRPC服务器
"""

import logging
import threading
from concurrent import futures
import grpc

# 导入生成的protobuf文件
try:
    import communication.grpc_service.gpu_share_pb2 as pb2
    import communication.grpc_service.gpu_share_pb2_grpc as pb2_grpc
    from .service_impl import GPUMasterServiceImpl, GPUWorkerServiceImpl
except ImportError:
    # 如果protobuf文件尚未生成，创建占位符
    pb2 = None
    pb2_grpc = None
    GPUMasterServiceImpl = None
    GPUWorkerServiceImpl = None

logger = logging.getLogger(__name__)

class GRPCServer:
    """gRPC服务器"""

    def __init__(self, host='0.0.0.0', port=50051, max_workers=10, 
                 service_type="master", scheduler=None, worker_manager=None, 
                 task_executor=None, file_transfer_manager=None):
        self.host = host
        self.port = port
        self.max_workers = max_workers
        self.service_type = service_type  # "master" 或 "worker"

        # 服务组件
        self.scheduler = scheduler
        self.worker_manager = worker_manager
        self.task_executor = task_executor
        self.file_transfer_manager = file_transfer_manager

        # gRPC服务器
        self.server = None

        # 运行状态
        self.running = False
        self.server_thread = None
        self.lock = threading.Lock()

    def start(self):
        """启动gRPC服务器"""
        with self.lock:
            if self.running:
                return

            if pb2 is None or pb2_grpc is None:
                logger.error("protobuf文件未生成，无法启动gRPC服务器")
                return False

            self.running = True

            # 创建gRPC服务器
            self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=self.max_workers))

            # 添加服务
            if self.service_type == "master":
                if not all([self.scheduler, self.worker_manager, self.file_transfer_manager]):
                    logger.error("主节点服务缺少必要组件")
                    self.running = False
                    return False

                service = GPUMasterServiceImpl(
                    self.scheduler, self.worker_manager, self.file_transfer_manager
                )
                pb2_grpc.add_GPUMasterServiceServicer_to_server(service, self.server)
                logger.info("已添加主节点服务")

            elif self.service_type == "worker":
                if not all([self.task_executor, self.file_transfer_manager]):
                    logger.error("Worker节点服务缺少必要组件")
                    self.running = False
                    return False

                service = GPUWorkerServiceImpl(
                    self.task_executor, self.file_transfer_manager
                )
                pb2_grpc.add_GPUWorkerServiceServicer_to_server(service, self.server)
                logger.info("已添加Worker节点服务")

            else:
                logger.error(f"未知的服务类型: {self.service_type}")
                self.running = False
                return False

            # 启动服务器
            self.server_thread = threading.Thread(target=self._run_server, daemon=True)
            self.server_thread.start()

            logger.info(f"gRPC服务器已启动: {self.host}:{self.port}")
            return True

    def stop(self):
        """停止gRPC服务器"""
        with self.lock:
            if not self.running:
                return False

            self.running = False

            if self.server:
                # 停止接受新请求
                self.server.stop(grace=2)

                # 等待服务器线程结束
                if self.server_thread and self.server_thread.is_alive():
                    self.server_thread.join(timeout=5)

                logger.info("gRPC服务器已停止")
                return True

            return False

    def _run_server(self):
        """运行服务器"""
        try:
            # 绑定端口并启动服务
            self.server.add_insecure_port(f'{self.host}:{self.port}')
            self.server.start()

            logger.info(f"gRPC服务器正在监听: {self.host}:{self.port}")

            # 保持服务器运行
            while self.running:
                time.sleep(1)

        except Exception as e:
            logger.error(f"gRPC服务器运行出错: {str(e)}")
        finally:
            # 确保服务器停止
            if self.server:
                self.server.stop(0)
