#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
构建Windows Worker可执行文件
"""

import os
import sys
import shutil
import subprocess

def build_worker():
    """构建Worker可执行文件"""
    print("开始构建Windows Worker可执行文件...")

    # 检查PyInstaller是否安装
    try:
        subprocess.run(["pyinstaller", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("错误: PyInstaller未安装，正在安装...")
        subprocess.run(["pip", "install", "pyinstaller"], check=True)

    # 创建spec文件
    spec_content = """# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['workers/windows/worker.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'grpc._cython.cygrpc',
        'pynvml',
        'pycuda.driver',
        'pycuda.tools',
        'pycuda.autoinit'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='gpu-worker',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico'
)
"""

    # 写入spec文件
    with open('gpu-worker.spec', 'w') as f:
        f.write(spec_content)

    # 构建可执行文件
    print("正在构建可执行文件...")
    subprocess.run(["pyinstaller", "gpu-worker.spec"], check=True)

    # 创建发布目录
    release_dir = "release"
    if os.path.exists(release_dir):
        shutil.rmtree(release_dir)
    os.makedirs(release_dir)

    # 复制文件
    shutil.copytree("dist/gpu-worker", f"{release_dir}/gpu-worker")

    # 创建配置文件
    config_content = """{
    "worker_id": "windows-worker-1",
    "master_host": "localhost",
    "master_port": 50051,
    "platform": "windows"
}"""

    with open(f"{release_dir}/config.json", 'w') as f:
        f.write(config_content)

    # 创建启动脚本
    start_script = """@echo off
echo 启动GPU Worker...
gpu-worker.exe
pause
"""

    with open(f"{release_dir}/start.bat", 'w') as f:
        f.write(start_script)

    # 创建README
    readme_content = """# GPU共享平台 - Windows Worker

## 使用方法

1. 编辑config.json文件，设置主节点地址和端口
2. 双击start.bat启动Worker
3. 在Web界面中查看Worker状态

## 配置说明

- worker_id: Worker唯一标识
- master_host: 主节点IP地址
- master_port: 主节点gRPC端口
- platform: 平台类型（固定为windows）

## 故障排除

1. 如果GPU检测失败，请确保已安装NVIDIA驱动
2. 如果连接超时，请检查网络和防火墙设置
3. 查看日志文件获取详细错误信息
"""

    with open(f"{release_dir}/README.md", 'w') as f:
        f.write(readme_content)

    print(f"构建完成，发布文件位于: {release_dir}")

    # 打包发布文件
    import zipfile
    with zipfile.ZipFile("gpu-worker-windows.zip", "w") as zipf:
        for root, _, files in os.walk(release_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, release_dir)
                zipf.write(file_path, arcname)

    print("发布包已创建: gpu-worker-windows.zip")

if __name__ == "__main__":
    build_worker()
