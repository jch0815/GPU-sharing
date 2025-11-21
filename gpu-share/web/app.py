#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Web应用模块
"""

import logging
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

logger = logging.getLogger(__name__)

def create_app(scheduler, worker_manager):
    """创建Flask应用"""
    app = Flask(__name__)
    CORS(app)  # 启用CORS支持

    # 存储调度器和Worker管理器的引用
    app.scheduler = scheduler
    app.worker_manager = worker_manager

    # 注册路由
    register_routes(app)

    return app

def register_routes(app):
    """注册路由"""

    @app.route('/')
    def index():
        """主页"""
        return render_template('index.html')

    @app.route('/api/gpus')
    def get_gpus():
        """获取所有GPU信息"""
        gpus = []
        for gpu in app.scheduler.get_all_gpus():
            gpus.append({
                'gpu_id': gpu.gpu_id,
                'worker_id': gpu.worker_id,
                'name': gpu.name,
                'memory_total': gpu.memory_total,
                'memory_used': gpu.memory_used,
                'utilization': gpu.utilization,
                'temperature': gpu.temperature,
                'power_usage': gpu.power_usage,
                'compute_capability': gpu.compute_capability,
                'is_available': gpu.is_available
            })
        return jsonify(gpus)

    @app.route('/api/tasks')
    def get_tasks():
        """获取所有任务信息"""
        tasks = []
        for task in app.scheduler._tasks.values():
            tasks.append({
                'task_id': task.task_id,
                'task_type': task.task_type,
                'status': task.status.value,
                'assigned_gpu_id': task.assigned_gpu_id,
                'worker_id': task.worker_id,
                'submit_time': task.submit_time,
                'start_time': task.start_time,
                'end_time': task.end_time,
                'requirements': task.requirements
            })
        return jsonify(tasks)

    @app.route('/api/workers')
    def get_workers():
        """获取所有Worker信息"""
        workers = app.worker_manager.get_all_workers()
        result = []
        for worker_id, worker_info in workers.items():
            result.append({
                'worker_id': worker_id,
                'hostname': worker_info.get('hostname', 'Unknown'),
                'platform': worker_info.get('platform', 'Unknown'),
                'status': worker_info.get('status', 'Unknown'),
                'last_heartbeat': worker_info.get('last_heartbeat', 0),
                'gpu_count': len(worker_info.get('gpus', []))
            })
        return jsonify(result)

    @app.route('/api/submit_task', methods=['POST'])
    def submit_task():
        """提交新任务"""
        data = request.json
        if not data:
            return jsonify({'error': '无效的请求数据'}), 400

        task_type = data.get('task_type', 'inference')
        requirements = data.get('requirements', {})

        # 生成任务ID
        import time
        task_id = f"task_{int(time.time())}"

        # 创建任务
        from core.scheduler import GPUTask
        task = GPUTask(
            task_id=task_id,
            task_type=task_type,
            requirements=requirements
        )

        # 提交任务
        app.scheduler.submit_task(task)

        return jsonify({'task_id': task_id, 'status': 'submitted'})

    @app.route('/api/cancel_task/<task_id>', methods=['POST'])
    def cancel_task(task_id):
        """取消任务"""
        success = app.scheduler.cancel_task(task_id)
        if success:
            return jsonify({'status': 'cancelled'})
        else:
            return jsonify({'error': '取消任务失败'}), 404

    @app.route('/api/task_result/<task_id>')
    def get_task_result(task_id):
        """获取任务结果"""
        result = app.scheduler.get_task_result(task_id)
        if result is not None:
            return jsonify(result)
        else:
            return jsonify({'error': '任务结果不存在或任务尚未完成'}), 404
