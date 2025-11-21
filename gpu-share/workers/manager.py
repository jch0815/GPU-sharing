#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Worker管理器
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class WorkerStatus(Enum):
    """Worker状态枚举"""
    ONLINE = "online"       # 在线
    OFFLINE = "offline"     # 离线
    BUSY = "busy"           # 忙碌
    ERROR = "error"         # 错误

@dataclass
class WorkerInfo:
    """Worker信息数据类"""
    worker_id: str
    hostname: str
    platform: str          # Windows, Linux, Docker, Android
    status: WorkerStatus
    last_heartbeat: float  # 最后心跳时间
    gpus: List[Dict]       # GPU列表
    metadata: Dict[str, Any]  # 其他元数据

class WorkerManager:
    """Worker管理器"""

    def __init__(self, heartbeat_timeout=60):
        self._workers: Dict[str, WorkerInfo] = {}  # worker_id -> WorkerInfo
        self._lock = threading.RLock()
        self._heartbeat_thread = None
        self._running = False
        self._heartbeat_timeout = heartbeat_timeout

    def start(self):
        """启动Worker管理器"""
        with self._lock:
            if self._running:
                return

            self._running = True
            self._heartbeat_thread = threading.Thread(target=self._heartbeat_check_loop, daemon=True)
            self._heartbeat_thread.start()
            logger.info("Worker管理器已启动")

    def stop(self):
        """停止Worker管理器"""
        with self._lock:
            if not self._running:
                return

            self._running = False
            if self._heartbeat_thread and self._heartbeat_thread.is_alive():
                self._heartbeat_thread.join(timeout=10)
            logger.info("Worker管理器已停止")

    def register_worker(self, worker_info: WorkerInfo) -> bool:
        """注册Worker"""
        with self._lock:
            worker_id = worker_info.worker_id
            if worker_id in self._workers:
                logger.warning(f"Worker {worker_id} 已存在，更新信息")

            worker_info.last_heartbeat = time.time()
            worker_info.status = WorkerStatus.ONLINE
            self._workers[worker_id] = worker_info
            logger.info(f"已注册Worker {worker_id} ({worker_info.platform})")
            return True

    def update_worker_heartbeat(self, worker_id: str) -> bool:
        """更新Worker心跳"""
        with self._lock:
            if worker_id not in self._workers:
                logger.warning(f"收到未知Worker的心跳: {worker_id}")
                return False

            self._workers[worker_id].last_heartbeat = time.time()

            # 如果Worker之前是离线状态，现在恢复在线
            if self._workers[worker_id].status == WorkerStatus.OFFLINE:
                self._workers[worker_id].status = WorkerStatus.ONLINE
                logger.info(f"Worker {worker_id} 已恢复在线")

            return True

    def update_worker_status(self, worker_id: str, status: WorkerStatus) -> bool:
        """更新Worker状态"""
        with self._lock:
            if worker_id not in self._workers:
                logger.warning(f"尝试更新未知Worker的状态: {worker_id}")
                return False

            old_status = self._workers[worker_id].status
            self._workers[worker_id].status = status

            if old_status != status:
                logger.info(f"Worker {worker_id} 状态从 {old_status.value} 更新为 {status.value}")

            return True

    def update_worker_gpus(self, worker_id: str, gpus: List[Dict]) -> bool:
        """更新Worker的GPU列表"""
        with self._lock:
            if worker_id not in self._workers:
                logger.warning(f"尝试更新未知Worker的GPU列表: {worker_id}")
                return False

            self._workers[worker_id].gpus = gpus
            logger.info(f"已更新Worker {worker_id} 的GPU列表，共 {len(gpus)} 个GPU")
            return True

    def unregister_worker(self, worker_id: str) -> bool:
        """注销Worker"""
        with self._lock:
            if worker_id not in self._workers:
                logger.warning(f"尝试注销不存在的Worker: {worker_id}")
                return False

            del self._workers[worker_id]
            logger.info(f"已注销Worker {worker_id}")
            return True

    def get_worker(self, worker_id: str) -> Optional[WorkerInfo]:
        """获取Worker信息"""
        with self._lock:
            return self._workers.get(worker_id)

    def get_all_workers(self) -> Dict[str, Dict]:
        """获取所有Worker信息"""
        with self._lock:
            result = {}
            for worker_id, worker_info in self._workers.items():
                result[worker_id] = {
                    'worker_id': worker_info.worker_id,
                    'hostname': worker_info.hostname,
                    'platform': worker_info.platform,
                    'status': worker_info.status.value,
                    'last_heartbeat': worker_info.last_heartbeat,
                    'gpus': worker_info.gpus,
                    'metadata': worker_info.metadata
                }
            return result

    def get_online_workers(self) -> List[WorkerInfo]:
        """获取在线Worker列表"""
        with self._lock:
            return [worker for worker in self._workers.values() 
                   if worker.status == WorkerStatus.ONLINE]

    def stop_all(self):
        """停止所有Worker"""
        with self._lock:
            logger.info("正在停止所有Worker...")
            for worker_id, worker_info in self._workers.items():
                if worker_info.status != WorkerStatus.OFFLINE:
                    # TODO: 向Worker发送停止信号
                    logger.info(f"已向Worker {worker_id} 发送停止信号")
            logger.info("所有Worker停止命令已发送")

    def _heartbeat_check_loop(self):
        """心跳检查循环"""
        while self._running:
            try:
                self._check_worker_heartbeats()
                time.sleep(10)  # 每10秒检查一次
            except Exception as e:
                logger.error(f"心跳检查循环出错: {str(e)}")
                time.sleep(5)  # 出错后短暂休眠

    def _check_worker_heartbeats(self):
        """检查Worker心跳"""
        current_time = time.time()

        with self._lock:
            for worker_id, worker_info in list(self._workers.items()):
                # 检查心跳超时
                if current_time - worker_info.last_heartbeat > self._heartbeat_timeout:
                    if worker_info.status != WorkerStatus.OFFLINE:
                        worker_info.status = WorkerStatus.OFFLINE
                        logger.warning(f"Worker {worker_id} 心跳超时，标记为离线")
