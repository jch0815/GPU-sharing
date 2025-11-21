@echo off
echo 启动GPU共享平台...

REM 设置Python路径
set PYTHONPATH=%CD%

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

REM 检查依赖是否安装
pip show flask >nul 2>&1
if errorlevel 1 (
    echo 正在安装依赖...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo 错误: 依赖安装失败
        pause
        exit /b 1
    )
)

REM 启动Web界面
echo 启动Web界面...
start "GPU共享平台 - Web界面" cmd /k "python run_web.py"

REM 启动gRPC服务
echo 启动gRPC服务...
start "GPU共享平台 - gRPC服务" cmd /k "python communication/grpc_service/server.py"

echo 启动完成，请访问 https://localhost:5050
pause
