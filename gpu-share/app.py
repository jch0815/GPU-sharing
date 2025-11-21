#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPU共享平台主入口文件
"""

import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入核心模块
from core.scheduler import GPUScheduler
from communication.server import CommunicationServer
from web.app import create_app
from workers.manager import WorkerManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gpu-share.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """主函数"""
    logger.info("启动GPU共享平台...")

    # 初始化GPU调度器
    scheduler = GPUScheduler()

    # 初始化通信服务器
    comm_server = CommunicationServer(scheduler)

    # 初始化Worker管理器
    worker_manager = WorkerManager()

    # 创建Web应用
    app = create_app(scheduler, worker_manager)

    # 启动服务
    try:
        # 在生产环境中，建议使用gunicorn或uWSGI
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        logger.info("接收到终止信号，正在关闭服务...")
    finally:
        # 清理资源
        comm_server.stop()
        worker_manager.stop_all()
        logger.info("服务已关闭")

if __name__ == '__main__':
    main()
