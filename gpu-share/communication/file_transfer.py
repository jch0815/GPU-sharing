#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据传输模块
"""

import hashlib
import logging
import os
import threading
import time
import zlib
from typing import Dict, Optional, Callable, BinaryIO

logger = logging.getLogger(__name__)

class FileTransferManager:
    """文件传输管理器"""

    def __init__(self, chunk_size=1024*1024):  # 默认1MB块大小
        self.chunk_size = chunk_size
        self.active_transfers: Dict[str, Dict] = {}  # file_id -> transfer_info
        self.lock = threading.Lock()

    def start_upload(self, file_path: str, file_id: Optional[str] = None, 
                   compress: bool = True) -> str:
        """开始文件上传"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 生成文件ID
        if file_id is None:
            file_id = self._generate_file_id(file_path)

        # 获取文件信息
        file_size = os.path.getsize(file_path)
        file_name = os.path.basename(file_path)

        # 计算总块数
        total_chunks = (file_size + self.chunk_size - 1) // self.chunk_size

        # 创建传输信息
        transfer_info = {
            "file_id": file_id,
            "file_path": file_path,
            "file_name": file_name,
            "file_size": file_size,
            "total_chunks": total_chunks,
            "uploaded_chunks": 0,
            "compress": compress,
            "start_time": time.time(),
            "status": "uploading"
        }

        # 添加到活动传输列表
        with self.lock:
            self.active_transfers[file_id] = transfer_info

        logger.info(f"开始文件上传: {file_name} ({file_size} bytes, {total_chunks} chunks)")
        return file_id

    def get_upload_chunk(self, file_id: str, chunk_index: int) -> Optional[bytes]:
        """获取上传块"""
        with self.lock:
            if file_id not in self.active_transfers:
                return None

            transfer_info = self.active_transfers[file_id]

            # 检查块索引是否有效
            if chunk_index < 0 or chunk_index >= transfer_info["total_chunks"]:
                return None

            # 打开文件
            file_path = transfer_info["file_path"]
            with open(file_path, 'rb') as f:
                # 定位到块位置
                f.seek(chunk_index * self.chunk_size)

                # 读取块数据
                chunk_data = f.read(self.chunk_size)

                # 压缩数据（如果需要）
                if transfer_info["compress"]:
                    chunk_data = zlib.compress(chunk_data)

                return chunk_data

    def complete_upload(self, file_id: str) -> bool:
        """完成文件上传"""
        with self.lock:
            if file_id not in self.active_transfers:
                return False

            transfer_info = self.active_transfers[file_id]
            transfer_info["status"] = "completed"
            transfer_info["end_time"] = time.time()

            logger.info(f"文件上传完成: {transfer_info['file_name']}")
            return True

    def start_download(self, file_id: str, file_name: str, file_size: int, 
                     total_chunks: int, output_dir: str = ".", 
                     compress: bool = True) -> str:
        """开始文件下载"""
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 创建输出文件路径
        file_path = os.path.join(output_dir, file_name)

        # 创建传输信息
        transfer_info = {
            "file_id": file_id,
            "file_path": file_path,
            "file_name": file_name,
            "file_size": file_size,
            "total_chunks": total_chunks,
            "downloaded_chunks": 0,
            "compress": compress,
            "start_time": time.time(),
            "status": "downloading",
            "chunk_data": {}  # chunk_index -> chunk_data
        }

        # 添加到活动传输列表
        with self.lock:
            self.active_transfers[file_id] = transfer_info

        logger.info(f"开始文件下载: {file_name} ({file_size} bytes, {total_chunks} chunks)")
        return file_path

    def add_download_chunk(self, file_id: str, chunk_index: int, chunk_data: bytes) -> bool:
        """添加下载块"""
        with self.lock:
            if file_id not in self.active_transfers:
                return False

            transfer_info = self.active_transfers[file_id]

            # 检查块索引是否有效
            if chunk_index < 0 or chunk_index >= transfer_info["total_chunks"]:
                return False

            # 解压缩数据（如果需要）
            if transfer_info["compress"]:
                chunk_data = zlib.decompress(chunk_data)

            # 存储块数据
            transfer_info["chunk_data"][chunk_index] = chunk_data
            transfer_info["downloaded_chunks"] += 1

            # 检查是否所有块都已下载
            if transfer_info["downloaded_chunks"] == transfer_info["total_chunks"]:
                return self._assemble_file(file_id)

            return True

    def _assemble_file(self, file_id: str) -> bool:
        """组装文件"""
        with self.lock:
            if file_id not in self.active_transfers:
                return False

            transfer_info = self.active_transfers[file_id]
            file_path = transfer_info["file_path"]
            chunk_data = transfer_info["chunk_data"]

            try:
                # 按顺序写入所有块
                with open(file_path, 'wb') as f:
                    for i in range(transfer_info["total_chunks"]):
                        if i not in chunk_data:
                            logger.error(f"缺少块 {i}，无法组装文件")
                            return False

                        f.write(chunk_data[i])

                # 更新传输状态
                transfer_info["status"] = "completed"
                transfer_info["end_time"] = time.time()

                logger.info(f"文件下载完成: {transfer_info['file_name']}")
                return True

            except Exception as e:
                logger.error(f"组装文件出错: {str(e)}")
                return False

    def get_transfer_info(self, file_id: str) -> Optional[Dict]:
        """获取传输信息"""
        with self.lock:
            return self.active_transfers.get(file_id)

    def get_all_transfers(self) -> Dict[str, Dict]:
        """获取所有活动传输"""
        with self.lock:
            return self.active_transfers.copy()

    def cancel_transfer(self, file_id: str) -> bool:
        """取消传输"""
        with self.lock:
            if file_id not in self.active_transfers:
                return False

            transfer_info = self.active_transfers[file_id]
            transfer_info["status"] = "cancelled"
            transfer_info["end_time"] = time.time()

            # 如果是下载，删除不完整的文件
            if transfer_info["status"] == "downloading" and os.path.exists(transfer_info["file_path"]):
                try:
                    os.remove(transfer_info["file_path"])
                    logger.info(f"已删除不完整的下载文件: {transfer_info['file_name']}")
                except Exception as e:
                    logger.error(f"删除不完整的下载文件出错: {str(e)}")

            logger.info(f"已取消文件传输: {transfer_info['file_name']}")
            return True

    def _generate_file_id(self, file_path: str) -> str:
        """生成文件ID"""
        # 使用文件路径和修改时间生成唯一ID
        stat = os.stat(file_path)
        content = f"{file_path}_{stat.st_mtime}_{stat.st_size}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()

class FileTransferClient:
    """文件传输客户端"""

    def __init__(self, stub, file_transfer_manager: FileTransferManager):
        self.stub = stub
        self.file_transfer_manager = file_transfer_manager

    def upload_file(self, file_path: str, compress: bool = True) -> Optional[str]:
        """上传文件"""
        # 开始上传
        file_id = self.file_transfer_manager.start_upload(file_path, compress=compress)
        if not file_id:
            return None

        # 获取传输信息
        transfer_info = self.file_transfer_manager.get_transfer_info(file_id)
        if not transfer_info:
            return None

        # 上传所有块
        try:
            for chunk_index in range(transfer_info["total_chunks"]):
                # 获取块数据
                chunk_data = self.file_transfer_manager.get_upload_chunk(file_id, chunk_index)
                if not chunk_data:
                    logger.error(f"无法获取上传块 {chunk_index}")
                    self.file_transfer_manager.cancel_transfer(file_id)
                    return None

                # 发送块
                request = communication.grpc_service.gpu_share_pb2.FileTransferRequest(
                    file_id=file_id,
                    file_name=transfer_info["file_name"],
                    file_size=transfer_info["file_size"],
                    chunk_data=chunk_data,
                    chunk_index=chunk_index,
                    total_chunks=transfer_info["total_chunks"],
                    is_compressed=compress
                )

                response = self.stub.UploadFile(request)
                if not response.success:
                    logger.error(f"上传块 {chunk_index} 失败: {response.message}")
                    self.file_transfer_manager.cancel_transfer(file_id)
                    return None

                logger.debug(f"已上传块 {chunk_index}/{transfer_info['total_chunks']}")

            # 完成上传
            self.file_transfer_manager.complete_upload(file_id)
            return file_id

        except Exception as e:
            logger.error(f"上传文件出错: {str(e)}")
            self.file_transfer_manager.cancel_transfer(file_id)
            return None

    def download_file(self, file_id: str, file_name: str, file_size: int, 
                    total_chunks: int, output_dir: str = ".", 
                    compress: bool = True) -> bool:
        """下载文件"""
        # 开始下载
        file_path = self.file_transfer_manager.start_download(
            file_id, file_name, file_size, total_chunks, output_dir, compress
        )
        if not file_path:
            return False

        try:
            # 请求下载
            request = communication.grpc_service.gpu_share_pb2.FileTransferRequest(
                file_id=file_id,
                file_name=file_name
            )

            # 接收文件流
            for response in self.stub.DownloadFile(request):
                if not response.success:
                    logger.error(f"下载文件失败: {response.message}")
                    self.file_transfer_manager.cancel_transfer(file_id)
                    return False

                # 添加块
                if not self.file_transfer_manager.add_download_chunk(
                    file_id, response.chunk_index, response.chunk_data
                ):
                    logger.error(f"添加下载块 {response.chunk_index} 失败")
                    self.file_transfer_manager.cancel_transfer(file_id)
                    return False

                logger.debug(f"已下载块 {response.chunk_index}/{total_chunks}")

            # 检查下载是否完成
            transfer_info = self.file_transfer_manager.get_transfer_info(file_id)
            if transfer_info and transfer_info["status"] == "completed":
                return True

            return False

        except Exception as e:
            logger.error(f"下载文件出错: {str(e)}")
            self.file_transfer_manager.cancel_transfer(file_id)
            return False

class FileTransferService:
    """文件传输服务"""

    def __init__(self, file_transfer_manager: FileTransferManager):
        self.file_transfer_manager = file_transfer_manager

    def UploadFile(self, request_iterator, context):
        """处理文件上传请求"""
        try:
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
                yield communication.grpc_service.gpu_share_pb2.FileTransferResponse(
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
                    yield communication.grpc_service.gpu_share_pb2.FileTransferResponse(
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
                    yield communication.grpc_service.gpu_share_pb2.FileTransferResponse(
                        file_id=file_id,
                        success=False,
                        message=f"无法添加块 {request.chunk_index}"
                    )
                    return

            # 检查下载是否完成
            transfer_info = self.file_transfer_manager.get_transfer_info(file_id)
            if transfer_info and transfer_info["status"] == "completed":
                yield communication.grpc_service.gpu_share_pb2.FileTransferResponse(
                    file_id=file_id,
                    success=True,
                    message="文件上传成功"
                )
            else:
                yield communication.grpc_service.gpu_share_pb2.FileTransferResponse(
                    file_id=file_id,
                    success=False,
                    message="文件上传不完整"
                )

        except Exception as e:
            logger.error(f"处理文件上传请求出错: {str(e)}")
            yield communication.grpc_service.gpu_share_pb2.FileTransferResponse(
                file_id=file_id if 'file_id' in locals() else "",
                success=False,
                message=f"服务器错误: {str(e)}"
            )

    def DownloadFile(self, request, context):
        """处理文件下载请求"""
        try:
            file_id = request.file_id
            file_name = request.file_name

            # 获取传输信息
            transfer_info = self.file_transfer_manager.get_transfer_info(file_id)
            if not transfer_info:
                yield communication.grpc_service.gpu_share_pb2.FileTransferResponse(
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
                    yield communication.grpc_service.gpu_share_pb2.FileTransferResponse(
                        file_id=file_id,
                        success=False,
                        message=f"无法获取块 {chunk_index}"
                    )
                    return

                # 发送块
                yield communication.grpc_service.gpu_share_pb2.FileTransferRequest(
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

        except Exception as e:
            logger.error(f"处理文件下载请求出错: {str(e)}")
            yield communication.grpc_service.gpu_share_pb2.FileTransferResponse(
                file_id=file_id if 'file_id' in locals() else "",
                success=False,
                message=f"服务器错误: {str(e)}"
            )le_transfer_manager: FileTransferManager):
        self.file_transfer_manager = file_transfer_manager

    def UploadFile(self, request_iterator, context):
        """处理文件上传请求"""
        # 获取第一个请求获取文件信息
        first_request = next(request_iterator)
        file_id = first_request.file_id
        file_name = first_request.file_name
        file_size = first_request.file_size
        total_chunks = first_request.total_chunks
        is_compressed = first_request.is_compressed

        # 创建临时文件路径
        import tempfile
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, file_name)

        # 开始下载
        download_file_id = self.file_transfer_manager.start_download(
            file_id, file_name, file_size, total_chunks, temp_dir, is_compressed
        )

        if not download_file_id:
            yield communication.grpc_service.gpu_share_pb2.FileTransferResponse(
                file_id=file_id,
                success=False,
                message="无法开始文件下载"
            )
            return

        try:
            # 处理第一个块
            self.file_transfer_manager.add_download_chunk(
                file_id, first_request.chunk_index, first_request.chunk_data
            )

            # 处理剩余块
            for request in request_iterator:
                self.file_transfer_manager.add_download_chunk(
                    file_id, request.chunk_index, request.chunk_data
                )

            # 检查下载是否完成
            transfer_info = self.file_transfer_manager.get_transfer_info(file_id)
            if transfer_info and transfer_info["status"] == "completed":
                # 移动文件到最终位置
                import shutil
                final_path = os.path.join("uploads", file_name)
                os.makedirs("uploads", exist_ok=True)
                shutil.move(file_path, final_path)

                # 清理临时目录
                shutil.rmtree(temp_dir)

                yield communication.grpc_service.gpu_share_pb2.FileTransferResponse(
                    file_id=file_id,
                    success=True,
                    message="文件上传成功"
                )
            else:
                yield communication.grpc_service.gpu_share_pb2.FileTransferResponse(
                    file_id=file_id,
                    success=False,
                    message="文件上传不完整"
                )

        except Exception as e:
            logger.error(f"处理文件上传出错: {str(e)}")
            yield communication.grpc_service.gpu_share_pb2.FileTransferResponse(
                file_id=file_id,
                success=False,
                message=f"处理上传出错: {str(e)}"
            )
            # 清理临时目录
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except:
                pass

    def DownloadFile(self, request, context):
        """处理文件下载请求"""
        file_id = request.file_id
        file_name = request.file_name

        # 检查文件是否存在
        file_path = os.path.join("uploads", file_name)
        if not os.path.exists(file_path):
            yield communication.grpc_service.gpu_share_pb2.FileTransferResponse(
                file_id=file_id,
                success=False,
                message="文件不存在"
            )
            return

        # 开始上传
        upload_file_id = self.file_transfer_manager.start_upload(file_path, file_id)
        if not upload_file_id:
            yield communication.grpc_service.gpu_share_pb2.FileTransferResponse(
                file_id=file_id,
                success=False,
                message="无法开始文件上传"
            )
            return

        try:
            # 获取传输信息
            transfer_info = self.file_transfer_manager.get_transfer_info(upload_file_id)

            # 发送所有块
            for chunk_index in range(transfer_info["total_chunks"]):
                # 获取块数据
                chunk_data = self.file_transfer_manager.get_upload_chunk(upload_file_id, chunk_index)
                if not chunk_data:
                    yield communication.grpc_service.gpu_share_pb2.FileTransferResponse(
                        file_id=file_id,
                        success=False,
                        message=f"无法读取块 {chunk_index}"
                    )
                    return

                # 发送块
                yield communication.grpc_service.gpu_share_pb2.FileTransferRequest(
                    file_id=file_id,
                    file_name=file_name,
                    file_size=transfer_info["file_size"],
                    chunk_data=chunk_data,
                    chunk_index=chunk_index,
                    total_chunks=transfer_info["total_chunks"],
                    is_compressed=transfer_info["compress"]
                )

            # 完成上传
            self.file_transfer_manager.complete_upload(upload_file_id)

        except Exception as e:
            logger.error(f"处理文件下载出错: {str(e)}")
            yield communication.grpc_service.gpu_share_pb2.FileTransferResponse(
                file_id=file_id,
                success=False,
                message=f"处理下载出错: {str(e)}"
            )
