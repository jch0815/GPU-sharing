#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
安全中间件
"""

import logging
from functools import wraps
from flask import request, jsonify, g

# 导入认证模块
try:
    from .auth import JWTAuth, DeviceAuth
    from .crypto import generate_api_key
except ImportError:
    JWTAuth = None
    DeviceAuth = None
    generate_api_key = None

logger = logging.getLogger(__name__)

# 初始化认证管理器
jwt_auth = JWTAuth() if JWTAuth else None
device_auth = DeviceAuth() if DeviceAuth else None
api_key = generate_api_key() if generate_api_key else None

def require_auth(f):
    """认证装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # 检查API密钥
        if not api_key:
            return jsonify({'error': 'API密钥未配置'}), 500

        # 检查请求头中的API密钥
        provided_key = request.headers.get('X-API-Key')
        if not provided_key or provided_key != api_key:
            return jsonify({'error': '无效的API密钥'}), 401

        return f(*args, **kwargs)
    return decorated_function

def require_jwt(f):
    """JWT认证装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not jwt_auth:
            return jsonify({'error': 'JWT认证未启用'}), 500

        # 检查请求头中的JWT令牌
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not token:
            return jsonify({'error': '缺少认证令牌'}), 401

        # 验证JWT令牌
        payload = jwt_auth.verify_token(token)
        if not payload:
            return jsonify({'error': '无效的认证令牌'}), 401

        # 将用户信息添加到请求上下文
        g.user = payload
        return f(*args, **kwargs)
    return decorated_function

def require_device_auth(f):
    """设备认证装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not device_auth:
            return jsonify({'error': '设备认证未启用'}), 500

        # 检查请求头中的设备令牌
        token = request.headers.get('X-Device-Token')
        if not token:
            return jsonify({'error': '缺少设备令牌'}), 401

        # 获取设备ID
        device_id = request.view_args.get('device_id')
        if not device_id:
            return jsonify({'error': '缺少设备ID'}), 400

        # 验证设备令牌
        if not device_auth.verify_device(device_id, token):
            return jsonify({'error': '无效的设备令牌'}), 401

        # 将设备信息添加到请求上下文
        g.device = device_auth.get_device_info(device_id)
        return f(*args, **kwargs)
    return decorated_function

def require_https(f):
    """HTTPS强制装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # 检查是否为HTTPS请求
        if not request.is_secure:
            return jsonify({'error': '必须使用HTTPS'}), 403

        return f(*args, **kwargs)
    return decorated_function

def encrypt_sensitive_data(data):
    """加密敏感数据"""
    try:
        from .crypto import encrypt_data
        return encrypt_data(data, api_key)
    except Exception as e:
        logger.error(f"加密数据失败: {str(e)}")
        return data

def decrypt_sensitive_data(encrypted_data):
    """解密敏感数据"""
    try:
        from .crypto import decrypt_data
        return decrypt_data(encrypted_data, api_key)
    except Exception as e:
        logger.error(f"解密数据失败: {str(e)}")
        return encrypted_data
