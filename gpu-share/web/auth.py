#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
用户认证模块
"""

import hashlib
import logging
import os
import secrets
from functools import wraps
from flask import session, request, redirect, url_for

logger = logging.getLogger(__name__)

# 硬编码的用户名和密码哈希
# 实际应用中应该使用数据库存储用户信息
DEFAULT_USERS = {
    'admin': bcrypt.hashpw(b'gpu-share-admin', bcrypt.gensalt()).decode('utf-8')
}

def login_required(f):
    """登录装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def verify_user(username, password):
    """验证用户名和密码"""
    # 检查用户是否存在
    if username not in DEFAULT_USERS:
        return False

    # 验证密码
    hashed_password = DEFAULT_USERS[username].encode('utf-8')
    provided_password = password.encode('utf-8')

    return bcrypt.checkpw(provided_password, hashed_password)

def generate_session_token():
    """生成会话令牌"""
    return secrets.token_hex(16)

def hash_password(password):
    """哈希密码"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def create_user(username, password):
    """创建新用户"""
    if username in DEFAULT_USERS:
        return False

    DEFAULT_USERS[username] = hash_password(password)
    return True

def update_password(username, old_password, new_password):
    """更新用户密码"""
    if username not in DEFAULT_USERS:
        return False

    # 验证旧密码
    if not verify_user(username, old_password):
        return False

    # 更新密码
    DEFAULT_USERS[username] = hash_password(new_password)
    return True

def delete_user(username, password):
    """删除用户"""
    if username not in DEFAULT_USERS:
        return False

    # 验证密码
    if not verify_user(username, password):
        return False

    # 删除用户
    del DEFAULT_USERS[username]
    return True

def get_users():
    """获取所有用户"""
    users = []
    for username in DEFAULT_USERS:
        users.append({
            'username': username,
            'is_admin': username == 'admin'
        })
    return users

def save_users():
    """保存用户信息到文件"""
    try:
        # 创建用户目录
        os.makedirs('data', exist_ok=True)

        # 保存用户信息
        with open('data/users.json', 'w') as f:
            import json
            json.dump({
                'users': [{
                    'username': username,
                    'password_hash': password_hash
                } for username, password_hash in DEFAULT_USERS.items()
                }]
            }, f, indent=2)

        logger.info("用户信息已保存")
        return True
    except Exception as e:
        logger.error(f"保存用户信息失败: {str(e)}")
        return False

def load_users():
    """从文件加载用户信息"""
    try:
        # 检查用户文件是否存在
        if not os.path.exists('data/users.json'):
            return False

        # 加载用户信息
        with open('data/users.json', 'r') as f:
            import json
            data = json.load(f)

            # 更新用户信息
            for user in data.get('users', []):
                DEFAULT_USERS[user['username']] = user['password_hash']

        logger.info("用户信息已加载")
        return True
    except Exception as e:
        logger.error(f"加载用户信息失败: {str(e)}")
        return False

def init_auth():
    """初始化认证系统"""
    # 创建数据目录
    os.makedirs('data', exist_ok=True)

    # 尝试加载用户信息
    if not load_users():
        # 如果加载失败，创建默认管理员用户
        logger.warning("无法加载用户信息，创建默认管理员用户")
        save_users()

    return True
