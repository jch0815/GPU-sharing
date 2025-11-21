#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
设备发现模块
"""

import json
import logging
import socket
import struct
import threading
import time
from typing import Dict, List, Optional, Callable

logger = logging.getLogger(__name__)

class DeviceDiscovery:
    """设备发现服务"""

    def __init__(self, server_host=None, server_port=None, device_type="master"):
        self.server_host = server_host or self._get_local_ip()
        self.server_port = server_port or 50051
        self.device_type = device_type  # "master" 或 "worker"

        # 广播设置
        self.broadcast_port = 50052
        self.broadcast_interval = 10  # 秒

        # 设备列表
        self.devices: Dict[str, Dict] = {}  # device_id -> device_info

        # 回调函数
        self.callbacks: List[Callable] = []

        # 运行状态
        self.running = False
        self.broadcast_thread = None
        self.listen_thread = None
        self.lock = threading.Lock()

    def start(self):
        """启动设备发现服务"""
        with self.lock:
            if self.running:
                return

            self.running = True

            if self.device_type == "master":
                # 主节点：监听设备发现请求
                self.listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
                self.listen_thread.start()
                logger.info("设备发现服务已启动（主节点模式）")
            else:
                # Worker节点：广播设备信息
                self.broadcast_thread = threading.Thread(target=self._broadcast_loop, daemon=True)
                self.broadcast_thread.start()
                logger.info("设备发现服务已启动（Worker节点模式）")

    def stop(self):
        """停止设备发现服务"""
        with self.lock:
            if not self.running:
                return

            self.running = False

            if self.broadcast_thread and self.broadcast_thread.is_alive():
                self.broadcast_thread.join(timeout=5)

            if self.listen_thread and self.listen_thread.is_alive():
                self.listen_thread.join(timeout=5)

            logger.info("设备发现服务已停止")

    def add_callback(self, callback: Callable):
        """添加设备发现回调函数"""
        with self.lock:
            self.callbacks.append(callback)

    def remove_callback(self, callback: Callable):
        """移除设备发现回调函数"""
        with self.lock:
            if callback in self.callbacks:
                self.callbacks.remove(callback)

    def get_devices(self) -> Dict[str, Dict]:
        """获取发现的设备列表"""
        with self.lock:
            return self.devices.copy()

    def _get_local_ip(self):
        """获取本地IP地址"""
        try:
            # 创建一个UDP套接字连接到公共DNS服务器
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except Exception:
            # 如果上述方法失败，返回回环地址
            return "127.0.0.1"

    def _broadcast_loop(self):
        """广播循环"""
        while self.running:
            try:
                # 构造广播消息
                message = {
                    "device_id": f"worker-{socket.gethostname()}",
                    "device_type": "worker",
                    "host": self.server_host,
                    "port": self.server_port,
                    "timestamp": time.time()
                }

                # 序列化消息
                message_json = json.dumps(message).encode('utf-8')

                # 创建UDP套接字
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

                # 发送广播
                sock.sendto(message_json, ('<broadcast>', self.broadcast_port))
                sock.close()

                logger.debug(f"已广播设备信息: {message['device_id']}")

                # 等待下一次广播
                time.sleep(self.broadcast_interval)

            except Exception as e:
                logger.error(f"设备广播出错: {str(e)}")
                time.sleep(5)  # 出错后短暂休眠

    def _listen_loop(self):
        """监听循环"""
        # 创建UDP套接字
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('', self.broadcast_port))

        while self.running:
            try:
                # 接收数据
                data, addr = sock.recvfrom(4096)

                # 解析消息
                try:
                    message = json.loads(data.decode('utf-8'))
                    self._process_discovery_message(message, addr[0])
                except Exception as e:
                    logger.error(f"解析设备发现消息出错: {str(e)}")

            except Exception as e:
                logger.error(f"设备发现监听出错: {str(e)}")
                time.sleep(1)  # 出错后短暂休眠

        sock.close()

    def _process_discovery_message(self, message: Dict, sender_ip: str):
        """处理设备发现消息"""
        device_id = message.get("device_id")
        device_type = message.get("device_type")

        if not device_id or not device_type:
            logger.warning("收到无效的设备发现消息")
            return

        # 忽略自己的广播
        if device_type == self.device_type:
            return

        # 更新设备信息
        with self.lock:
            # 更新或添加设备
            if device_id not in self.devices:
                logger.info(f"发现新设备: {device_id} ({device_type})")

            self.devices[device_id] = {
                **message,
                "last_seen": time.time(),
                "sender_ip": sender_ip
            }

            # 触发回调
            for callback in self.callbacks:
                try:
                    callback(device_id, self.devices[device_id])
                except Exception as e:
                    logger.error(f"设备发现回调出错: {str(e)}")

class AndroidDeviceDiscovery:
    """Android设备发现服务"""

    def __init__(self, adb_path="adb"):
        self.adb_path = adb_path
        self.devices: Dict[str, Dict] = {}  # device_id -> device_info
        self.callbacks: List[Callable] = []
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.check_interval = 10  # 秒

    def start(self):
        """启动Android设备发现服务"""
        with self.lock:
            if self.running:
                return

            self.running = True
            self.thread = threading.Thread(target=self._check_loop, daemon=True)
            self.thread.start()
            logger.info("Android设备发现服务已启动")

    def stop(self):
        """停止Android设备发现服务"""
        with self.lock:
            if not self.running:
                return

            self.running = False

            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=5)

            logger.info("Android设备发现服务已停止")

    def add_callback(self, callback: Callable):
        """添加设备发现回调函数"""
        with self.lock:
            self.callbacks.append(callback)

    def remove_callback(self, callback: Callable):
        """移除设备发现回调函数"""
        with self.lock:
            if callback in self.callbacks:
                self.callbacks.remove(callback)

    def get_devices(self) -> Dict[str, Dict]:
        """获取发现的Android设备列表"""
        with self.lock:
            return self.devices.copy()

    def _check_loop(self):
        """检查循环"""
        while self.running:
            try:
                # 获取已连接的设备列表
                result = subprocess.run(
                    [self.adb_path, "devices", "-l"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                if result.returncode != 0:
                    logger.error(f"获取Android设备列表失败: {result.stderr}")
                    time.sleep(self.check_interval)
                    continue

                # 解析设备列表
                current_devices = {}
                for line in result.stdout.strip().split('
'):
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) < 2:
                        continue

                    device_id = parts[0]
                    status = parts[1]

                    # 只处理已授权的设备
                    if status != "device":
                        continue

                    # 获取设备详细信息
                    device_info = self._get_device_info(device_id)
                    if device_info:
                        current_devices[device_id] = device_info

                # 比较设备列表，发现新设备和断开连接的设备
                with self.lock:
                    # 检查新设备
                    for device_id, device_info in current_devices.items():
                        if device_id not in self.devices:
                            logger.info(f"发现新的Android设备: {device_id}")

                            # 触发回调
                            for callback in self.callbacks:
                                try:
                                    callback(device_id, device_info)
                                except Exception as e:
                                    logger.error(f"Android设备发现回调出错: {str(e)}")

                    # 检查断开连接的设备
                    for device_id in list(self.devices.keys()):
                        if device_id not in current_devices:
                            logger.info(f"Android设备断开连接: {device_id}")

                    # 更新设备列表
                    self.devices = current_devices

                # 等待下一次检查
                time.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Android设备检查出错: {str(e)}")
                time.sleep(5)  # 出错后短暂休眠

    def _get_device_info(self, device_id: str) -> Optional[Dict]:
        """获取设备详细信息"""
        try:
            # 获取设备属性
            result = subprocess.run(
                [self.adb_path, "-s", device_id, "shell", "getprop"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if result.returncode != 0:
                logger.error(f"获取Android设备属性失败: {result.stderr}")
                return None

            # 解析设备属性
            properties = {}
            for line in result.stdout.strip().split('
'):
                if not line:
                    continue

                parts = line.split(': ', 1)
                if len(parts) == 2:
                    properties[parts[0]] = parts[1]

            # 获取GPU信息
            gpu_result = subprocess.run(
                [self.adb_path, "-s", device_id, "shell", "dumpsys", "gpu"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            gpu_info = ""
            if gpu_result.returncode == 0:
                # 尝试从输出中提取GPU信息
                for line in gpu_result.stdout.strip().split('
'):
                    if 'GL_RENDERER' in line or 'GPU' in line:
                        parts = line.split('=')
                        if len(parts) > 1:
                            gpu_info = parts[1].strip()
                            break

            # 获取端口转发状态
            port_result = subprocess.run(
                [self.adb_path, "-s", device_id, "forward", "--list"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            port_forwards = []
            if port_result.returncode == 0:
                for line in port_result.stdout.strip().split('
'):
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 3:
                        port_forwards.append({
                            "local": parts[0],
                            "remote": parts[1]
                        })

            # 返回设备信息
            return {
                "device_id": device_id,
                "model": properties.get("ro.product.model", "Unknown"),
                "manufacturer": properties.get("ro.product.manufacturer", "Unknown"),
                "android_version": properties.get("ro.build.version.release", "Unknown"),
                "api_level": properties.get("ro.build.version.sdk", "Unknown"),
                "gpu_info": gpu_info,
                "port_forwards": port_forwards,
                "last_seen": time.time()
            }

        except Exception as e:
            logger.error(f"获取Android设备信息出错: {str(e)}")
            return None

    def setup_port_forward(self, device_id: str, local_port: int, remote_port: int) -> bool:
        """设置端口转发"""
        try:
            result = subprocess.run(
                [self.adb_path, "-s", device_id, "forward", f"tcp:{local_port}", f"tcp:{remote_port}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            if result.returncode == 0:
                logger.info(f"已设置端口转发: {device_id} {local_port} -> {remote_port}")
                return True
            else:
                logger.error(f"设置端口转发失败: {result.stderr.decode('utf-8')}")
                return False

        except Exception as e:
            logger.error(f"设置端口转发出错: {str(e)}")
            return False

    def remove_port_forward(self, device_id: str, local_port: int) -> bool:
        """移除端口转发"""
        try:
            result = subprocess.run(
                [self.adb_path, "-s", device_id, "forward", "--remove", f"tcp:{local_port}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            if result.returncode == 0:
                logger.info(f"已移除端口转发: {device_id} {local_port}")
                return True
            else:
                logger.error(f"移除端口转发失败: {result.stderr.decode('utf-8')}")
                return False

        except Exception as e:
            logger.error(f"移除端口转发出错: {str(e)}")
            return False(
                [self.adb_path, "-s", device_id, "forward", "--list"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            port_forwards = []
            if port_result.returncode == 0:
                for line in port_result.stdout.strip().split('
'):
                    if line:
                        parts = line.split(' ')
                        if len(parts) >= 3:
                            port_forwards.append({
                                "local": parts[0],
                                "remote": parts[1]
                            })

            # 构造设备信息
            return {
                "device_id": device_id,
                "device_type": "android",
                "model": properties.get("ro.product.model", "Unknown"),
                "manufacturer": properties.get("ro.product.manufacturer", "Unknown"),
                "android_version": properties.get("ro.build.version.release", "Unknown"),
                "api_level": properties.get("ro.build.version.sdk", "Unknown"),
                "gpu_info": gpu_info,
                "port_forwards": port_forwards,
                "last_seen": time.time()
            }

        except Exception as e:
            logger.error(f"获取Android设备信息出错: {str(e)}")
            return None

    def setup_port_forward(self, device_id: str, local_port: int, remote_port: int) -> bool:
        """设置端口转发"""
        try:
            result = subprocess.run(
                [self.adb_path, "-s", device_id, "forward", f"tcp:{local_port}", f"tcp:{remote_port}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if result.returncode != 0:
                logger.error(f"设置Android设备端口转发失败: {result.stderr}")
                return False

            logger.info(f"已设置Android设备端口转发: {device_id} {local_port} -> {remote_port}")
            return True

        except Exception as e:
            logger.error(f"设置Android设备端口转发出错: {str(e)}")
            return False

    def remove_port_forward(self, device_id: str, local_port: int) -> bool:
        """移除端口转发"""
        try:
            result = subprocess.run(
                [self.adb_path, "-s", device_id, "forward", "--remove", f"tcp:{local_port}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if result.returncode != 0:
                logger.error(f"移除Android设备端口转发失败: {result.stderr}")
                return False

            logger.info(f"已移除Android设备端口转发: {device_id} {local_port}")
            return True

        except Exception as e:
            logger.error(f"移除Android设备端口转发出错: {str(e)}")
            return False
