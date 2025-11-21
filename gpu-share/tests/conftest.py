#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试配置
"""

import os
import sys
import pytest
import tempfile
import shutil

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# 测试数据目录
TEST_DATA_DIR = os.path.join(project_root, 'tests', 'data')

@pytest.fixture(scope="session")
def test_data_dir():
    """创建测试数据目录"""
    if not os.path.exists(TEST_DATA_DIR):
        os.makedirs(TEST_DATA_DIR, exist_ok=True)
    return TEST_DATA_DIR

@pytest.fixture
def temp_dir():
    """创建临时目录"""
    return tempfile.mkdtemp()

@pytest.fixture
def cleanup_temp_dir(temp_dir):
    """清理临时目录"""
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
