#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
密码加密模块
"""

import hashlib
import secrets
import logging
import os
import time
from typing import Dict, Optional

try:
    import bcrypt
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("bcrypt库未安装，将使用简单的SHA-256哈希")
    bcrypt = None

logger = logging.getLogger(__name__)

def hash_password(password: str, use_bcrypt: bool = True) -> str:
    """哈希密码"""
    if use_bcrypt and bcrypt:
        # 使用bcrypt哈希
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    else:
        # 使用SHA-256哈希
        return hashlib.sha256(password.encode('utf-8')).hexdigest()

def verify_password(password: str, hashed_password: str, use_bcrypt: bool = True) -> bool:
    """验证密码"""
    if use_bcrypt and bcrypt:
        # 验证bcrypt哈希
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
    else:
        # 验证SHA-256哈希
        return hashlib.sha256(password.encode('utf-8')).hexdigest() == hashed_password

def generate_salt() -> str:
    """生成随机盐值"""
    return secrets.token_hex(16)

def generate_token(length: int = 32) -> str:
    """生成随机令牌"""
    return secrets.token_urlsafe(length)

def generate_api_key(length: int = 32) -> str:
    """生成API密钥"""
    return secrets.token_urlsafe(length)

def encrypt_data(data: str, key: str) -> str:
    """使用AES加密数据"""
    try:
        from Crypto.Cipher import AES
        from Crypto.Util.Padding import pad, unpad
        from Crypto.Random import get_random_bytes

        # 将密钥转换为16字节
        key_bytes = key.encode('utf-8')
        if len(key_bytes) < 16:
            key_bytes = key_bytes.ljust(16, b'\0')
        elif len(key_bytes) > 16:
            key_bytes = key_bytes[:16]

        # 生成随机IV
        iv = get_random_bytes(AES.block_size)

        # 创建加密器
        cipher = AES.new(key_bytes, AES.MODE_CBC, iv)

        # 填充数据
        padded_data = pad(data.encode('utf-8'), AES.block_size)

        # 加密数据
        encrypted_data = cipher.encrypt(padded_data)

        # 返回IV和加密数据
        return iv.hex() + encrypted_data.hex()
    except ImportError:
        logger.warning("PyCryptodome库未安装，无法加密数据")
        return data

def decrypt_data(encrypted_data: str, key: str) -> str:
    """使用AES解密数据"""
    try:
        from Crypto.Cipher import AES
        from Crypto.Util.Padding import unpad

        # 将密钥转换为16字节
        key_bytes = key.encode('utf-8')
        if len(key_bytes) < 16:
            key_bytes = key_bytes.ljust(16, b'\0')
        elif len(key_bytes) > 16:
            key_bytes = key_bytes[:16]

        # 提取IV和加密数据
        iv = bytes.fromhex(encrypted_data[:32])  # 16字节
        encrypted_bytes = bytes.fromhex(encrypted_data[32:])

        # 创建解密器
        cipher = AES.new(key_bytes, AES.MODE_CBC, iv)

        # 解密数据
        decrypted_padded = cipher.decrypt(encrypted_bytes)

        # 去除填充
        decrypted_data = unpad(decrypted_padded, AES.block_size)

        return decrypted_data.decode('utf-8')
    except ImportError:
        logger.warning("PyCryptodome库未安装，无法解密数据")
        return encrypted_data

class KeyManager:
    """密钥管理器"""

    def __init__(self, key_file: str = 'data/keys.json'):
        self.key_file = key_file
        self.keys = self._load_keys()

    def _load_keys(self) -> Dict[str, Dict]:
        """加载密钥"""
        if not os.path.exists(self.key_file):
            return {}

        try:
            import json
            with open(self.key_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载密钥失败: {str(e)}")
            return {}

    def _save_keys(self):
        """保存密钥"""
        try:
            import json
            os.makedirs(os.path.dirname(self.key_file), exist_ok=True)
            with open(self.key_file, 'w') as f:
                json.dump(self.keys, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"保存密钥失败: {str(e)}")
            return False

    def generate_key(self, name: str, key_type: str = 'api') -> str:
        """生成新密钥"""
        key = generate_api_key()

        # 添加到密钥列表
        self.keys[name] = {
            'key': key,
            'type': key_type,
            'created_at': time.time()
        }

        # 保存密钥
        if self._save_keys():
            logger.info(f"已生成新{key_type}密钥: {name}")
            return key

        return None

    def get_key(self, name: str) -> Optional[str]:
        """获取密钥"""
        if name in self.keys:
            return self.keys[name]['key']
        return None

    def revoke_key(self, name: str) -> bool:
        """撤销密钥"""
        if name not in self.keys:
            return False

        del self.keys[name]

        # 保存密钥
        if self._save_keys():
            logger.info(f"已撤销密钥: {name}")
            return True

        return False

    def rotate_keys(self, key_type: str = 'api') -> bool:
        """轮换所有指定类型的密钥"""
        keys_to_revoke = []

        for name, key_info in self.keys.items():
            if key_info['type'] == key_type:
                keys_to_revoke.append(name)

        # 生成新密钥
        for name in keys_to_revoke:
            new_key = generate_api_key()
            self.keys[name] = {
                'key': new_key,
                'type': key_type,
                'created_at': time.time()
            }
            logger.info(f"已轮换密钥: {name}")

        # 保存密钥
        return self._save_keys()

def generate_self_signed_cert(cert_file: str = 'data/server.crt', 
                         key_file: str = 'data/server.key',
                         country: str = 'CN',
                         state: str = 'Beijing',
                         locality: str = 'Beijing',
                         organization: str = 'GPU Share Platform',
                         common_name: str = 'gpu-share'):
    """生成自签名证书"""
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        import ipaddress

        # 生成私钥
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )

        # 创建证书主题
        subject = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, country),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, state),
            x509.NameAttribute(NameOID.LOCALITY_NAME, locality),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, organization),
            x509.NameAttribute(NameOID.COMMON_NAME, common_name)
        ])

        # 创建证书
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            subject
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.datetime.utcnow()
        ).not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=365)
        ).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName(common_name)
            ])
        ).sign(
            private_key,
            hashes.SHA256()
        )

        # 创建目录
        os.makedirs(os.path.dirname(cert_file), exist_ok=True)
        os.makedirs(os.path.dirname(key_file), exist_ok=True)

        # 写入证书和私钥
        with open(cert_file, 'wb') as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))

        with open(key_file, 'wb') as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL
            ))

        logger.info(f"已生成自签名证书: {cert_file}, {key_file}")
        return True
    except ImportError:
        logger.warning("cryptography库未安装，无法生成自签名证书")
        return False
    except Exception as e:
        logger.error(f"生成自签名证书失败: {str(e)}")
        return False
