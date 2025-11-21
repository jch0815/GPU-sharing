# 环境配置指南

## 开发环境配置

### 主开发环境

1. **安装Python 3.8+**
   - 推荐使用Python 3.8或更高版本
   - 确保pip已安装并更新到最新版本

2. **创建虚拟环境**
   ```bash
   # 创建虚拟环境
   python -m venv gpu_env
   
   # 激活虚拟环境
   # Windows:
   gpu_env\Scripts\activate
   # Linux/Mac:
   source gpu_env/bin/activate
   ```

3. **安装基础依赖**
   ```bash
   pip install -r requirements.txt
   ```

### GPU库配置

#### CUDA (NVIDIA GPU)

1. 安装NVIDIA驱动程序
2. 安装CUDA Toolkit (推荐CUDA 11.x)
3. 验证CUDA安装:
   ```bash
   nvidia-smi
   ```

#### OpenCL (跨平台GPU)

1. 安装适合您平台的OpenCL运行时
   - NVIDIA: 已包含在CUDA中
   - AMD: 安装AMD ROCm或AMD APP SDK
   - Intel: 安装Intel OpenCL驱动

2. 验证OpenCL安装:
   ```python
   import pyopencl as cl
   platforms = cl.get_platforms()
   print("可用的OpenCL平台:")
   for p in platforms:
       print(p.name)
   ```

### 跨平台工具

#### Docker

1. 安装Docker Desktop (Windows/Mac) 或 Docker Engine (Linux)
2. 验证安装:
   ```bash
   docker --version
   docker-compose --version
   ```

#### Android SDK

1. 下载并安装Android Studio
2. 配置Android SDK路径
3. 安装ADB (Android Debug Bridge):
   ```bash
   # 验证ADB安装
   adb version
   ```

### 通信库配置

#### gRPC

1. 安装gRPC工具:
   ```bash
   pip install grpcio-tools
   ```

2. 生成protobuf代码:
   ```bash
   python -m grpc_tools.protoc --python_out=. --grpc_python_out=. --proto_path=. your_proto_file.proto
   ```

#### WebSocket

1. WebSocket库已包含在requirements.txt中
2. 测试WebSocket连接:
   ```python
   import asyncio
   import websockets
   
   async def test_websocket():
       uri = "ws://localhost:8765"
       async with websockets.connect(uri) as websocket:
           await websocket.send("Hello, WebSocket!")
           response = await websocket.recv()
           print(f"Received: {response}")
   ```

## IDE配置

### VS Code

1. 安装Python扩展
2. 配置Python解释器指向虚拟环境
3. 安装Docker扩展(可选)
4. 安装Remote - SSH扩展(用于远程开发)

### PyCharm

1. 创建新项目并选择现有解释器(虚拟环境)
2. 配置代码风格为PEP 8
3. 设置代码自动格式化工具

## 常见问题

### 1. CUDA安装问题

- 确保安装了与NVIDIA驱动兼容的CUDA版本
- 检查环境变量PATH是否包含CUDA路径

### 2. OpenCL检测不到设备

- 确保安装了正确的GPU驱动程序
- 检查OpenCL运行时是否正确安装

### 3. Android SDK配置问题

- 确保ANDROID_HOME环境变量设置正确
- 检查SDK平台工具是否在PATH中

### 4. Docker权限问题

- Linux用户可能需要将用户添加到docker组:
  ```bash
  sudo usermod -aG docker $USER
  ```
- 重启或注销后重新登录以应用更改