# GPU共享平台

## 项目概述

GPU共享平台是一个分布式GPU资源共享系统，旨在将不同设备上的GPU资源进行统一调度和管理，实现跨平台GPU算力的共享和利用。

## 项目结构

```
gpu-share/
├── core/          # 核心算力调度模块（Python）
├── communication/ # 跨设备通信协议（Python/C）
├── web/           # Web管理界面（HTML/Flask）
├── workers/       # 各平台Worker端代码
│   ├── windows/
│   ├── linux/
│   ├── docker/
│   └── android/   # Java客户端
├── tests/         # 测试代码
└── docs/          # 文档
```

## 技术栈

- **核心框架**: Python 3.8+
- **GPU支持**: CUDA (NVIDIA), OpenCL (跨平台)
- **通信协议**: gRPC, WebSocket
- **Web框架**: Flask
- **容器化**: Docker
- **移动端**: Android (Java)

## 功能特性

- 跨平台GPU资源检测和管理
- 分布式任务调度
- 实时资源监控
- Web管理界面
- 安全认证和权限控制

## 快速开始

### 环境要求

- Python 3.8+
- Docker (可选)
- Android SDK (用于安卓开发)

### 安装步骤

```bash
# 克隆仓库
git clone <repository-url>
cd gpu-share

# 创建虚拟环境
python -m venv gpu_env
source gpu_env/bin/activate  # Linux/Mac
gpu_env\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 启动服务
python app.py
```
