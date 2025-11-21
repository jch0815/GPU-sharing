#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Web应用入口
"""

import os
import sys
import logging
from flask import Flask

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

# 导入模块
from web.app import create_app
from core.scheduler import GPUScheduler
from workers.manager import WorkerManager
from web.auth import init_auth

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gpu-share-web.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """主函数"""
    # 初始化认证系统
    init_auth()

    # 创建调度器和Worker管理器
    scheduler = GPUScheduler()
    worker_manager = WorkerManager()

    # 创建Flask应用
    app = create_app(scheduler, worker_manager)

    # 启动应用
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'

    logger.info(f"启动Web应用: {host}:{port}")

    try:
        app.run(host=host, port=port, debug=debug)
    except KeyboardInterrupt:
        logger.info("收到终止信号，正在关闭Web应用...")
    except Exception as e:
        logger.error(f"Web应用运行出错: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
