#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
JWT认证模块
"""

import logging
import os
import time
from typing import Dict, Optional, Tuple

try:
    import jwt
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("PyJWT库未安装，无法使用JWT认证")
    jwt = None

logger = logging.getLogger(__name__)

class JWTAuth:
    """JWT认证管理器"""

    def __init__(self, secret_key: str, algorithm: str = 'HS256', 
                 token_expire: int = 3600):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.token_expire = token_expire

        # 存储已撤销的令牌
        self.revoked_tokens = set()

    def generate_token(self, payload: Dict) -> str:
        """生成JWT令牌"""
        if not jwt:
            logger.error("PyJWT库未安装，无法生成令牌")
            return ""

        # 添加过期时间
        payload['exp'] = time.time() + self.token_expire

        # 生成令牌
        token = jwt.encode(
            payload,
            self.secret_key,
            algorithm=self.algorithm
        )

        return token

    def verify_token(self, token: str) -> Optional[Dict]:
        """验证JWT令牌"""
        if not jwt:
            logger.error("PyJWT库未安装，无法验证令牌")
            return None

        try:
            # 解码令牌
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )

            # 检查令牌是否已撤销
            if payload.get('jti') in self.revoked_tokens:
                return None

            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def revoke_token(self, token: str) -> bool:
        """撤销JWT令牌"""
        if not jwt:
            logger.error("PyJWT库未安装，无法撤销令牌")
            return False

        try:
            # 解码令牌获取JTI
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )

            # 添加到已撤销列表
            if 'jti' in payload:
                self.revoked_tokens.add(payload['jti'])
                return True

            return False
        except jwt.InvalidTokenError:
            return False

class DeviceAuth:
    """设备认证管理器"""

    def __init__(self):
        # 存储已注册的设备
        self.registered_devices = {}

        # 存储设备白名单
        self.whitelist = set()

        # 存储设备令牌
        self.device_tokens = {}

    def register_device(self, device_id: str, device_info: Dict) -> Optional[str]:
        """注册设备"""
        # 检查设备是否在白名单中
        if self.whitelist and device_id not in self.whitelist:
            return None

        # 生成设备令牌
        token = self._generate_device_token(device_id)

        # 存储设备信息
        self.registered_devices[device_id] = {
            'info': device_info,
            'token': token,
            'registered_at': time.time()
        }

        # 存储设备令牌
        self.device_tokens[token] = device_id

        return token

    def unregister_device(self, device_id: str, token: str) -> bool:
        """注销设备"""
        # 检查令牌是否有效
        if token not in self.device_tokens or self.device_tokens[token] != device_id:
            return False

        # 移除设备信息
        if device_id in self.registered_devices:
            del self.registered_devices[device_id]

        # 移除设备令牌
        del self.device_tokens[token]

        return True

    def verify_device(self, device_id: str, token: str) -> bool:
        """验证设备"""
        # 检查令牌是否有效
        if token not in self.device_tokens or self.device_tokens[token] != device_id:
            return False

        # 检查设备是否已注册
        if device_id not in self.registered_devices:
            return False

        return True

    def add_to_whitelist(self, device_id: str) -> bool:
        """添加设备到白名单"""
        self.whitelist.add(device_id)
        return True

    def remove_from_whitelist(self, device_id: str) -> bool:
        """从白名单中移除设备"""
        if device_id in self.whitelist:
            self.whitelist.remove(device_id)
            return True
        return False

    def _generate_device_token(self, device_id: str) -> str:
        """生成设备令牌"""
        # 创建载荷
        payload = {
            'device_id': device_id,
            'iat': time.time(),
            'jti': f"device-{device_id}-{int(time.time())}"
        }

        # 生成令牌
        return jwt.encode(
            payload,
            os.urandom(32).hex(),  # 使用随机密钥
            algorithm='HS256'
        )

    def get_device_info(self, device_id: str) -> Optional[Dict]:
        """获取设备信息"""
        return self.registered_devices.get(device_id, {}).get('info')

class SecurityManager:
    """安全管理器"""

    def __init__(self, jwt_secret: str = None, jwt_algorithm: str = 'HS256'):
        # JWT认证管理器
        self.jwt_auth = JWTAuth(jwt_secret, jwt_algorithm) if jwt_secret else None

        # 设备认证管理器
        self.device_auth = DeviceAuth()

        # 通信加密密钥
        self.communication_key = os.urandom(32).hex()

    def generate_user_token(self, user_id: str, role: str = 'user') -> Optional[str]:
        """生成用户令牌"""
        if not self.jwt_auth:
            return None

        # 创建载荷
        payload = {
            'user_id': user_id,
            'role': role,
            'iat': time.time(),
            'jti': f"user-{user_id}-{int(time.time())}"
        }

        # 生成令牌
        return self.jwt_auth.generate_token(payload)

    def verify_user_token(self, token: str) -> Optional[Dict]:
        """验证用户令牌"""
        if not self.jwt_auth:
            return None

        return self.jwt_auth.verify_token(token)

    def generate_device_token(self, device_id: str, device_info: Dict) -> Optional[str]:
        """生成设备令牌"""
        return self.device_auth.register_device(device_id, device_info)

    def verify_device_token(self, device_id: str, token: str) -> bool:
        """验证设备令牌"""
        return self.device_auth.verify_device(device_id, token)

    def revoke_device_token(self, device_id: str, token: str) -> bool:
        """撤销设备令牌"""
        return self.device_auth.unregister_device(device_id, token)

    def encrypt_data(self, data: str) -> str:
        """加密数据"""
        try:
            from .crypto import encrypt_data
            return encrypt_data(data, self.communication_key)
        except ImportError:
            logger.warning("加密模块不可用，返回原始数据")
            return data

    def decrypt_data(self, encrypted_data: str) -> str:
        """解密数据"""
        try:
            from .crypto import decrypt_data
            return decrypt_data(encrypted_data, self.communication_key)
        except ImportError:
            logger.warning("解密模块不可用，返回原始数据")
            return encrypted_data
