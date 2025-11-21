# 使用NVIDIA官方CUDA基础镜像
FROM nvidia/cuda:11.4.2-devel-ubuntu20.04

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y     python3     python3-pip     git     curl     wget     && rm -rf /var/lib/apt/lists/*

# 升级pip
RUN pip3 install --upgrade pip

# 复制项目文件
COPY . /app

# 安装Python依赖
RUN pip3 install -r requirements.txt

# 暴露端口
EXPOSE 5000 5050 50051

# 设置环境变量
ENV PYTHONPATH=/app
ENV CUDA_HOME=/usr/local/cuda

# 启动命令
CMD ["python", "run_web.py"]
