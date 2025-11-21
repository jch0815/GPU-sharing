#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简化版GPU共享平台 - 修复版
"""

import os
import sys
import json
import time
import logging
from flask import Flask, render_template_string, jsonify, request

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__)

# 模拟GPU数据
def get_mock_gpu_data():
    """获取模拟GPU数据"""
    return [
        {
            "id": "gpu-0",
            "name": "NVIDIA GeForce RTX 3080",
            "memory_total": 10240,
            "memory_used": 4096,
            "utilization": 45.2,
            "temperature": 68.5,
            "power_usage": 220.5,
            "compute_capability": "8.6",
            "is_available": True
        },
        {
            "id": "gpu-1",
            "name": "NVIDIA GeForce RTX 3070",
            "memory_total": 8192,
            "memory_used": 2048,
            "utilization": 30.1,
            "temperature": 65.0,
            "power_usage": 180.3,
            "compute_capability": "8.6",
            "is_available": True
        }
    ]

# 模拟任务数据
tasks = []

# HTML模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPU共享平台</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/vue@3.2.45/dist/vue.global.js"></script>
    <style>
        .gpu-card {
            transition: transform 0.3s;
        }
        .gpu-card:hover {
            transform: translateY(-5px);
        }
        .task-card {
            margin-bottom: 10px;
        }
        .status-pending { background-color: #fff3cd; }
        .status-running { background-color: #cfe2ff; }
        .status-completed { background-color: #d1e7dd; }
        .status-failed { background-color: #f8d7da; }
    </style>
</head>
<body>
    <div id="app" class="container-fluid">
        <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
            <div class="container-fluid">
                <a class="navbar-brand" href="#">GPU共享平台</a>
                <div class="navbar-nav ms-auto">
                    <span class="navbar-text">简化版演示</span>
                </div>
            </div>
        </nav>

        <div class="container mt-4">
            <!-- GPU状态卡片 -->
            <div class="row mb-4">
                <div class="col-12">
                    <h3>GPU状态</h3>
                </div>
                <div v-for="gpu in gpus" :key="gpu.id" class="col-md-6 mb-3">
                    <div class="card gpu-card h-100">
                        <div class="card-header">
                            <h5 class="card-title mb-0">{{ gpu.name }}</h5>
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-6">
                                    <p><strong>内存:</strong> {{ gpu.memory_used }}MB / {{ gpu.memory_total }}MB</p>
                                    <div class="progress mb-2">
                                        <div class="progress-bar" role="progressbar" 
                                            :style="{width: (gpu.memory_used / gpu.memory_total * 100) + '%'}">
                                        </div>
                                    </div>
                                </div>
                                <div class="col-6">
                                    <p><strong>利用率:</strong> {{ gpu.utilization }}%</p>
                                    <div class="progress mb-2">
                                        <div class="progress-bar bg-info" role="progressbar" 
                                            :style="{width: gpu.utilization + '%'}">
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div class="row">
                                <div class="col-6">
                                    <p><strong>温度:</strong> {{ gpu.temperature }}°C</p>
                                </div>
                                <div class="col-6">
                                    <p><strong>功耗:</strong> {{ gpu.power_usage }}W</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- 任务管理 -->
            <div class="row">
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-header">
                            <h5 class="card-title mb-0">提交任务</h5>
                        </div>
                        <div class="card-body">
                            <form @submit.prevent="submitTask">
                                <div class="mb-3">
                                    <label class="form-label">任务类型</label>
                                    <select class="form-select" v-model="newTask.type">
                                        <option value="matrix_multiplication">矩阵乘法</option>
                                        <option value="model_inference">模型推理</option>
                                        <option value="data_processing">数据处理</option>
                                    </select>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">参数</label>
                                    <textarea class="form-control" v-model="newTask.parameters" rows="3"></textarea>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">优先级</label>
                                    <select class="form-select" v-model="newTask.priority">
                                        <option value="0">低</option>
                                        <option value="1">普通</option>
                                        <option value="2">高</option>
                                        <option value="3">紧急</option>
                                    </select>
                                </div>
                                <button type="submit" class="btn btn-primary">提交任务</button>
                            </form>
                        </div>
                    </div>
                </div>
                <div class="col-md-8">
                    <div class="card">
                        <div class="card-header">
                            <h5 class="card-title mb-0">任务列表</h5>
                        </div>
                        <div class="card-body">
                            <div v-for="task in tasks" :key="task.id" 
                                class="card task-card" :class="'status-' + task.status">
                                <div class="card-body">
                                    <div class="d-flex justify-content-between">
                                        <h6 class="card-title mb-1">{{ task.type }} ({{ task.id }})</h6>
                                        <div>
                                            <button class="btn btn-sm btn-outline-danger" 
                                                @click="deleteTask(task.id)">删除</button>
                                        </div>
                                    </div>
                                    <p class="card-text">
                                        <small>参数: {{ task.parameters }}</small><br>
                                        <small>优先级: {{ task.priority }} | 状态: {{ task.status }}</small>
                                    </p>
                                </div>
                            </div>
                            <div v-if="tasks.length === 0" class="text-center text-muted">
                                暂无任务
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const { createApp } = Vue;

        createApp({
            data() {
                return {
                    gpus: [],
                    tasks: [],
                    newTask: {
                        type: 'matrix_multiplication',
                        parameters: '{"size": 1024, "iterations": 10}',
                        priority: 1
                    }
                };
            },
            methods: {
                async fetchGpus() {
                    try {
                        const response = await fetch('/api/gpus');
                        this.gpus = await response.json();
                    } catch (error) {
                        console.error('获取GPU信息失败:', error);
                    }
                },
                async fetchTasks() {
                    try {
                        const response = await fetch('/api/tasks');
                        this.tasks = await response.json();
                    } catch (error) {
                        console.error('获取任务列表失败:', error);
                    }
                },
                async submitTask() {
                    try {
                        const response = await fetch('/api/tasks', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify(this.newTask)
                        });

                        if (response.ok) {
                            this.fetchTasks();
                            // 重置表单
                            this.newTask.parameters = '{"size": 1024, "iterations": 10}';
                            this.newTask.priority = 1;
                        }
                    } catch (error) {
                        console.error('提交任务失败:', error);
                    }
                },
                async deleteTask(taskId) {
                    try {
                        const response = await fetch(`/api/tasks/${taskId}`, {
                            method: 'DELETE'
                        });

                        if (response.ok) {
                            this.fetchTasks();
                        }
                    } catch (error) {
                        console.error('删除任务失败:', error);
                    }
                }
            },
            mounted() {
                this.fetchGpus();
                this.fetchTasks();

                // 定期更新数据
                setInterval(() => {
                    this.fetchGpus();
                }, 5000);
            }
        }).mount('#app');
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """首页"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/gpus')
def get_gpus():
    """获取GPU信息"""
    return jsonify(get_mock_gpu_data())

@app.route('/api/tasks', methods=['GET', 'POST'])
def tasks_api():
    """任务API"""
    if request.method == 'GET':
        return jsonify(tasks)
    else:
        # 添加新任务
        task = request.get_json()
        task['id'] = f"task-{len(tasks) + 1}"
        task['status'] = 'pending'
        task['submit_time'] = time.time()
        tasks.append(task)
        return jsonify({'success': True, 'task_id': task['id']})

@app.route('/api/tasks/<task_id>', methods=['GET', 'DELETE'])
def task_detail(task_id):
    """任务详情API"""
    if request.method == 'GET':
        # 获取任务详情
        for task in tasks:
            if task['id'] == task_id:
                return jsonify(task)
        return jsonify({'error': '任务不存在'}), 404
    else:
        # 删除任务
        for i, task in enumerate(tasks):
            if task['id'] == task_id:
                tasks.pop(i)
                return jsonify({'success': True})
        return jsonify({'error': '任务不存在'}), 404

def main():
    """主函数"""
    print("启动简化版GPU共享平台...")
    print("访问地址: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    main()
        # 获取任务详情
        for task in tasks:
            if task['id'] == task_id:
                return jsonify(task)
        return jsonify({'error': '任务不存在'}), 404
    else:
        # 删除任务
        for i, task in enumerate(tasks):
            if task['id'] == task_id:
                tasks.pop(i)
                return jsonify({'success': True})
        return jsonify({'error': '任务不存在'}), 404

def main():
    """主函数"""
    print("启动简化版GPU共享平台...")
    print("访问地址: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    main()
