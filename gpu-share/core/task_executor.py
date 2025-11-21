#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
任务执行器
"""

import logging
import os
import tempfile
import time
import json
import numpy as np
from typing import Dict, Any, Optional, Callable

from core.task_scheduler import Task, TaskTypes

logger = logging.getLogger(__name__)

class BaseTaskHandler:
    """任务处理器基类"""

    def __init__(self, gpu_detector):
        self.gpu_detector = gpu_detector

    def execute(self, task: Task) -> Any:
        """执行任务"""
        raise NotImplementedError("子类必须实现execute方法")

class MatrixMultiplicationHandler(BaseTaskHandler):
    """矩阵乘法任务处理器"""

    def execute(self, task: Task) -> Dict:
        """执行矩阵乘法任务"""
        parameters = task.parameters
        size = parameters.get("size", 1024)
        dtype = parameters.get("dtype", "float32")
        iterations = parameters.get("iterations", 1)

        logger.info(f"执行矩阵乘法任务: 大小={size}x{size}, 数据类型={dtype}, 迭代次数={iterations}")

        # 获取GPU信息
        gpu_id = task.assigned_gpu_id
        if not gpu_id:
            raise ValueError("任务未分配GPU")

        gpu_info = self.gpu_detector.get_gpu_by_id(gpu_id)
        if not gpu_info:
            raise ValueError(f"找不到GPU: {gpu_id}")

        # 根据GPU类型选择执行方式
        if gpu_info.gpu_id.startswith("nvidia-"):
            return self._execute_with_cuda(task, size, dtype, iterations)
        elif gpu_info.gpu_id.startswith("opencl-"):
            return self._execute_with_opencl(task, size, dtype, iterations)
        else:
            # 使用NumPy在CPU上执行
            return self._execute_with_numpy(task, size, dtype, iterations)

    def _execute_with_cuda(self, task: Task, size: int, dtype: str, iterations: int) -> Dict:
        """使用CUDA执行矩阵乘法"""
        try:
            import cupy as cp

            # 创建随机矩阵
            np.random.seed(42)  # 确保可重复性
            a = cp.random.rand(size, size).astype(dtype)
            b = cp.random.rand(size, size).astype(dtype)

            # 预热GPU
            cp.dot(a, b)

            # 执行矩阵乘法并计时
            start_time = time.time()
            for _ in range(iterations):
                c = cp.dot(a, b)
            cp.cuda.Stream.null.synchronize()  # 确保所有计算完成
            elapsed_time = time.time() - start_time

            # 计算性能指标
            flops = 2 * size ** 3 * iterations  # 矩阵乘法的浮点运算次数
            gflops = flops / (elapsed_time * 1e9)  # GFLOPS

            # 返回结果
            return {
                "method": "CUDA",
                "size": size,
                "dtype": dtype,
                "iterations": iterations,
                "elapsed_time": elapsed_time,
                "gflops": gflops,
                "result_shape": c.shape,
                "gpu_id": task.assigned_gpu_id
            }

        except ImportError:
            logger.warning("CuPy不可用，回退到NumPy执行")
            return self._execute_with_numpy(task, size, dtype, iterations)
        except Exception as e:
            logger.error(f"CUDA矩阵乘法执行出错: {str(e)}")
            raise

    def _execute_with_opencl(self, task: Task, size: int, dtype: str, iterations: int) -> Dict:
        """使用OpenCL执行矩阵乘法"""
        try:
            import pyopencl as cl
            import pyopencl.array as cl_array

            # 获取GPU设备和上下文
            gpu_id = task.assigned_gpu_id
            if not gpu_id.startswith("opencl-"):
                raise ValueError("无效的OpenCL GPU ID")

            # 从ID中提取平台和设备索引
            parts = gpu_id.split('-')
            platform_idx = int(parts[1])
            device_idx = int(parts[2])

            # 创建上下文和命令队列
            ctx = cl.Context([cl.get_platforms()[platform_idx].get_devices()[device_idx]])
            queue = cl.CommandQueue(ctx)

            # 创建随机矩阵
            np.random.seed(42)  # 确保可重复性
            a_np = np.random.rand(size, size).astype(dtype)
            b_np = np.random.rand(size, size).astype(dtype)

            # 将数据传输到GPU
            a_g = cl_array.to_device(queue, a_np)
            b_g = cl_array.to_device(queue, b_np)
            c_g = cl_array.empty_like(a_g)

            # 创建并编译内核
            kernel_code = """
            __kernel void matmul(__global const float* A, __global const float* B, __global float* C, int N) {
                int row = get_global_id(0);
                int col = get_global_id(1);

                if (row < N && col < N) {
                    float sum = 0.0f;
                    for (int k = 0; k < N; k++) {
                        sum += A[row * N + k] * B[k * N + col];
                    }
                    C[row * N + col] = sum;
                }
            }
            """

            prg = cl.Program(ctx, kernel_code).build()

            # 执行矩阵乘法并计时
            start_time = time.time()
            for _ in range(iterations):
                prg.matmul(queue, (size, size), None, a_g.data, b_g.data, c_g.data, np.int32(size))
                queue.finish()  # 等待所有操作完成
            elapsed_time = time.time() - start_time

            # 计算性能指标
            flops = 2 * size ** 3 * iterations  # 矩阵乘法的浮点运算次数
            gflops = flops / (elapsed_time * 1e9)  # GFLOPS

            # 返回结果
            return {
                "method": "OpenCL",
                "size": size,
                "dtype": dtype,
                "iterations": iterations,
                "elapsed_time": elapsed_time,
                "gflops": gflops,
                "result_shape": (size, size),
                "gpu_id": task.assigned_gpu_id
            }

        except ImportError:
            logger.warning("PyOpenCL不可用，回退到NumPy执行")
            return self._execute_with_numpy(task, size, dtype, iterations)
        except Exception as e:
            logger.error(f"OpenCL矩阵乘法执行出错: {str(e)}")
            raise

    def _execute_with_numpy(self, task: Task, size: int, dtype: str, iterations: int) -> Dict:
        """使用NumPy执行矩阵乘法"""
        # 创建随机矩阵
        np.random.seed(42)  # 确保可重复性
        a = np.random.rand(size, size).astype(dtype)
        b = np.random.rand(size, size).astype(dtype)

        # 执行矩阵乘法并计时
        start_time = time.time()
        for _ in range(iterations):
            c = np.dot(a, b)
        elapsed_time = time.time() - start_time

        # 计算性能指标
        flops = 2 * size ** 3 * iterations  # 矩阵乘法的浮点运算次数
        gflops = flops / (elapsed_time * 1e9)  # GFLOPS

        # 返回结果
        return {
            "method": "NumPy",
            "size": size,
            "dtype": dtype,
            "iterations": iterations,
            "elapsed_time": elapsed_time,
            "gflops": gflops,
            "result_shape": c.shape,
            "gpu_id": task.assigned_gpu_id
        }

class ModelInferenceHandler(BaseTaskHandler):
    """模型推理任务处理器"""

    def execute(self, task: Task) -> Dict:
        """执行模型推理任务"""
        parameters = task.parameters
        model_path = parameters.get("model_path")
        input_data = parameters.get("input_data")
        batch_size = parameters.get("batch_size", 1)
        iterations = parameters.get("iterations", 1)

        logger.info(f"执行模型推理任务: 模型={model_path}, 批大小={batch_size}, 迭代次数={iterations}")

        # 获取GPU信息
        gpu_id = task.assigned_gpu_id
        if not gpu_id:
            raise ValueError("任务未分配GPU")

        gpu_info = self.gpu_detector.get_gpu_by_id(gpu_id)
        if not gpu_info:
            raise ValueError(f"找不到GPU: {gpu_id}")

        # 根据GPU类型选择执行方式
        if gpu_info.gpu_id.startswith("nvidia-"):
            return self._execute_with_cuda(task, model_path, input_data, batch_size, iterations)
        elif gpu_info.gpu_id.startswith("opencl-"):
            return self._execute_with_opencl(task, model_path, input_data, batch_size, iterations)
        else:
            # 使用CPU执行
            return self._execute_with_cpu(task, model_path, input_data, batch_size, iterations)

    def _execute_with_cuda(self, task: Task, model_path: str, input_data: Any, 
                          batch_size: int, iterations: int) -> Dict:
        """使用CUDA执行模型推理"""
        try:
            import torch
            import torchvision.transforms as transforms

            # 检查CUDA可用性
            if not torch.cuda.is_available():
                logger.warning("CUDA不可用，回退到CPU执行")
                return self._execute_with_cpu(task, model_path, input_data, batch_size, iterations)

            # 设置设备
            device = torch.device("cuda")

            # 加载模型
            if model_path.endswith('.pth'):
                model = torch.load(model_path)
                model = model.to(device)
                model.eval()
            else:
                # 如果是其他格式，需要相应的加载逻辑
                raise ValueError(f"不支持的模型格式: {model_path}")

            # 准备输入数据
            if isinstance(input_data, str):
                # 如果是文件路径，加载数据
                if input_data.endswith('.jpg') or input_data.endswith('.png'):
                    from PIL import Image
                    image = Image.open(input_data)
                    preprocess = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
                    input_tensor = preprocess(image)
                    input_batch = input_tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1)
                else:
                    raise ValueError(f"不支持的输入数据格式: {input_data}")
            else:
                # 如果是直接提供的数据
                input_batch = torch.tensor(input_data)

            # 将数据移至GPU
            if torch.cuda.is_available():
                input_batch = input_batch.to(device)

            # 预热GPU
            with torch.no_grad():
                _ = model(input_batch)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            # 执行推理并计时
            with torch.no_grad():
                start_time = time.time()
                for _ in range(iterations):
                    output = model(input_batch)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                elapsed_time = time.time() - start_time

            # 计算性能指标
            avg_time_per_batch = elapsed_time / iterations
            throughput = batch_size / avg_time_per_batch  # 样本/秒

            # 返回结果
            return {
                "method": "CUDA",
                "model_path": model_path,
                "batch_size": batch_size,
                "iterations": iterations,
                "elapsed_time": elapsed_time,
                "avg_time_per_batch": avg_time_per_batch,
                "throughput": throughput,
                "output_shape": list(output.shape),
                "gpu_id": task.assigned_gpu_id
            }

        except ImportError:
            logger.warning("PyTorch不可用，回退到CPU执行")
            return self._execute_with_cpu(task, model_path, input_data, batch_size, iterations)
        except Exception as e:
            logger.error(f"CUDA模型推理执行出错: {str(e)}")
            raise

    def _execute_with_opencl(self, task: Task, model_path: str, input_data: Any, 
                            batch_size: int, iterations: int) -> Dict:
        """使用OpenCL执行模型推理"""
        # 这里可以实现使用OpenCL执行模型推理的逻辑
        # 由于OpenCL没有像CUDA那样成熟的深度学习框架支持，这里简化实现
        logger.warning("OpenCL模型推理支持有限，回退到CPU执行")
        return self._execute_with_cpu(task, model_path, input_data, batch_size, iterations)

    def _execute_with_cpu(self, task: Task, model_path: str, input_data: Any, 
                         batch_size: int, iterations: int) -> Dict:
        """使用CPU执行模型推理"""
        try:
            import torch
            import torchvision.transforms as transforms

            # 设置设备
            device = torch.device("cpu")

            # 加载模型
            if model_path.endswith('.pth'):
                model = torch.load(model_path, map_location=device)
                model.eval()
            else:
                # 如果是其他格式，需要相应的加载逻辑
                raise ValueError(f"不支持的模型格式: {model_path}")

            # 准备输入数据
            if isinstance(input_data, str):
                # 如果是文件路径，加载数据
                if input_data.endswith('.jpg') or input_data.endswith('.png'):
                    from PIL import Image
                    image = Image.open(input_data)
                    preprocess = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
                    input_tensor = preprocess(image)
                    input_batch = input_tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1)
                else:
                    raise ValueError(f"不支持的输入数据格式: {input_data}")
            else:
                # 如果是直接提供的数据
                input_batch = torch.tensor(input_data)

            # 执行推理并计时
            with torch.no_grad():
                start_time = time.time()
                for _ in range(iterations):
                    output = model(input_batch)
                elapsed_time = time.time() - start_time

            # 计算性能指标
            avg_time_per_batch = elapsed_time / iterations
            throughput = batch_size / avg_time_per_batch  # 样本/秒

            # 返回结果
            return {
                "method": "CPU",
                "model_path": model_path,
                "batch_size": batch_size,
                "iterations": iterations,
                "elapsed_time": elapsed_time,
                "avg_time_per_batch": avg_time_per_batch,
                "throughput": throughput,
                "output_shape": list(output.shape),
                "gpu_id": task.assigned_gpu_id
            }

        except Exception as e:
            logger.error(f"CPU模型推理执行出错: {str(e)}")
            raise

class DataProcessingHandler(BaseTaskHandler):
    """数据处理任务处理器"""

    def execute(self, task: Task) -> Dict:
        """执行数据处理任务"""
        parameters = task.parameters
        operation = parameters.get("operation", "filter")
        data_size = parameters.get("data_size", 1000000)
        data_type = parameters.get("data_type", "float32")
        iterations = parameters.get("iterations", 1)

        logger.info(f"执行数据处理任务: 操作={operation}, 数据大小={data_size}, 数据类型={data_type}, 迭代次数={iterations}")

        # 获取GPU信息
        gpu_id = task.assigned_gpu_id
        if not gpu_id:
            raise ValueError("任务未分配GPU")

        gpu_info = self.gpu_detector.get_gpu_by_id(gpu_id)
        if not gpu_info:
            raise ValueError(f"找不到GPU: {gpu_id}")

        # 根据GPU类型选择执行方式
        if gpu_info.gpu_id.startswith("nvidia-"):
            return self._execute_with_cuda(task, operation, data_size, data_type, iterations)
        elif gpu_id.startswith("opencl-"):
            return self._execute_with_opencl(task, operation, data_size, data_type, iterations)
        else:
            # 使用NumPy在CPU上执行
            return self._execute_with_numpy(task, operation, data_size, data_type, iterations)

    def _execute_with_cuda(self, task: Task, operation: str, data_size: int, 
                          data_type: str, iterations: int) -> Dict:
        """使用CUDA执行数据处理"""
        try:
            import cupy as cp

            # 生成随机数据
            np.random.seed(42)  # 确保可重复性
            data = cp.random.rand(data_size).astype(data_type)

            # 根据操作类型执行相应的处理
            start_time = time.time()
            for _ in range(iterations):
                if operation == "filter":
                    # 示例：过滤大于0.5的值
                    result = cp.where(data > 0.5, data, 0)
                elif operation == "transform":
                    # 示例：应用正弦变换
                    result = cp.sin(data)
                elif operation == "reduce":
                    # 示例：求和
                    result = cp.sum(data)
                else:
                    raise ValueError(f"不支持的操作: {operation}")

            # 确保所有操作完成
            cp.cuda.Stream.null.synchronize()
            elapsed_time = time.time() - start_time

            # 计算性能指标
            throughput = data_size * iterations / (elapsed_time * 1e6)  # 百万元素/秒

            # 返回结果
            return {
                "method": "CUDA",
                "operation": operation,
                "data_size": data_size,
                "data_type": data_type,
                "iterations": iterations,
                "elapsed_time": elapsed_time,
                "throughput": throughput,
                "result_type": type(result).__name__,
                "gpu_id": task.assigned_gpu_id
            }

        except ImportError:
            logger.warning("CuPy不可用，回退到NumPy执行")
            return self._execute_with_numpy(task, operation, data_size, data_type, iterations)
        except Exception as e:
            logger.error(f"CUDA数据处理执行出错: {str(e)}")
            raise

    def _execute_with_opencl(self, task: Task, operation: str, data_size: int, 
                            data_type: str, iterations: int) -> Dict:
        """使用OpenCL执行数据处理"""
        try:
            import pyopencl as cl
            import pyopencl.array as cl_array

            # 获取GPU设备和上下文
            gpu_id = task.assigned_gpu_id
            if not gpu_id.startswith("opencl-"):
                raise ValueError("无效的OpenCL GPU ID")

            # 从ID中提取平台和设备索引
            parts = gpu_id.split('-')
            platform_idx = int(parts[1])
            device_idx = int(parts[2])

            # 创建上下文和命令队列
            ctx = cl.Context([cl.get_platforms()[platform_idx].get_devices()[device_idx]])
            queue = cl.CommandQueue(ctx)

            # 生成随机数据
            np.random.seed(42)  # 确保可重复性
            data_np = np.random.rand(data_size).astype(data_type)

            # 将数据传输到GPU
            data_g = cl_array.to_device(queue, data_np)

            # 根据操作类型执行相应的处理
            start_time = time.time()
            for _ in range(iterations):
                if operation == "filter":
                    # 示例：过滤大于0.5的值
                    result_g = cl_array.zeros_like(data_g)
                    kernel_code = """
                    __kernel void filter(__global const float* data, __global float* result, int size) {
                        int idx = get_global_id(0);
                        if (idx < size) {
                            if (data[idx] > 0.5f) {
                                result[idx] = data[idx];
                            } else {
                                result[idx] = 0.0f;
                            }
                        }
                    }
                    """
                    prg = cl.Program(ctx, kernel_code).build()
                    prg.filter(queue, (data_size,), None, data_g.data, result_g.data, np.int32(data_size))
                    result = result_g.get()
                elif operation == "transform":
                    # 示例：应用正弦变换
                    result_g = cl_array.empty_like(data_g)
                    kernel_code = """
                    __kernel void transform(__global const float* data, __global float* result, int size) {
                        int idx = get_global_id(0);
                        if (idx < size) {
                            result[idx] = sin(data[idx]);
                        }
                    }
                    """
                    prg = cl.Program(ctx, kernel_code).build()
                    prg.transform(queue, (data_size,), None, data_g.data, result_g.data, np.int32(data_size))
                    result = result_g.get()
                elif operation == "reduce":
                    # 示例：求和
                    result = cl_array.sum(data_g).get()
                else:
                    raise ValueError(f"不支持的操作: {operation}")

                queue.finish()  # 等待所有操作完成

            elapsed_time = time.time() - start_time

            # 计算性能指标
            throughput = data_size * iterations / (elapsed_time * 1e6)  # 百万元素/秒

            # 返回结果
            return {
                "method": "OpenCL",
                "operation": operation,
                "data_size": data_size,
                "data_type": data_type,
                "iterations": iterations,
                "elapsed_time": elapsed_time,
                "throughput": throughput,
                "result_type": type(result).__name__,
                "gpu_id": task.assigned_gpu_id
            }

        except ImportError:
            logger.warning("PyOpenCL不可用，回退到NumPy执行")
            return self._execute_with_numpy(task, operation, data_size, data_type, iterations)
        except Exception as e:
            logger.error(f"OpenCL数据处理执行出错: {str(e)}")
            raise

    def _execute_with_numpy(self, task: Task, operation: str, data_size: int, 
                           data_type: str, iterations: int) -> Dict:
        """使用NumPy执行数据处理"""
        # 生成随机数据
        np.random.seed(42)  # 确保可重复性
        data = np.random.rand(data_size).astype(data_type)

        # 根据操作类型执行相应的处理
        start_time = time.time()
        for _ in range(iterations):
            if operation == "filter":
                # 示例：过滤大于0.5的值
                result = np.where(data > 0.5, data, 0)
            elif operation == "transform":
                # 示例：应用正弦变换
                result = np.sin(data)
            elif operation == "reduce":
                # 示例：求和
                result = np.sum(data)
            else:
                raise ValueError(f"不支持的操作: {operation}")

        elapsed_time = time.time() - start_time

        # 计算性能指标
        throughput = data_size * iterations / (elapsed_time * 1e6)  # 百万元素/秒

        # 返回结果
        return {
            "method": "NumPy",
            "operation": operation,
            "data_size": data_size,
            "data_type": data_type,
            "iterations": iterations,
            "elapsed_time": elapsed_time,
            "throughput": throughput,
            "result_type": type(result).__name__,
            "gpu_id": task.assigned_gpu_id
        }

class CustomKernelHandler(BaseTaskHandler):
    """自定义内核任务处理器"""

    def execute(self, task: Task) -> Dict:
        """执行自定义内核任务"""
        parameters = task.parameters
        kernel_code = parameters.get("kernel_code")
        kernel_name = parameters.get("kernel_name", "custom_kernel")
        input_data = parameters.get("input_data")
        output_size = parameters.get("output_size")
        iterations = parameters.get("iterations", 1)

        logger.info(f"执行自定义内核任务: 内核={kernel_name}, 迭代次数={iterations}")

        # 获取GPU信息
        gpu_id = task.assigned_gpu_id
        if not gpu_id:
            raise ValueError("任务未分配GPU")

        gpu_info = self.gpu_detector.get_gpu_by_id(gpu_id)
        if not gpu_info:
            raise ValueError(f"找不到GPU: {gpu_id}")

        # 根据GPU类型选择执行方式
        if gpu_id.startswith("nvidia-"):
            return self._execute_with_cuda(task, kernel_code, kernel_name, input_data, output_size, iterations)
        elif gpu_id.startswith("opencl-"):
            return self._execute_with_opencl(task, kernel_code, kernel_name, input_data, output_size, iterations)
        else:
            raise ValueError("自定义内核任务需要GPU支持")

    def _execute_with_cuda(self, task: Task, kernel_code: str, kernel_name: str, 
                          input_data: Any, output_size: int, iterations: int) -> Dict:
        """使用CUDA执行自定义内核"""
        try:
            import cupy as cp
            from cupy.cuda.compiler import compile_with_cache

            # 编译CUDA内核
            module = compile_with_cache(kernel_code)
            kernel = module.get_function(kernel_name)

            # 准备输入数据
            if isinstance(input_data, str):
                # 如果是文件路径，加载数据
                input_array = np.load(input_data)
            else:
                # 如果是直接提供的数据
                input_array = np.array(input_data)

            # 将数据传输到GPU
            input_gpu = cp.array(input_array)

            # 准备输出缓冲区
            if output_size:
                output_gpu = cp.zeros(output_size, dtype=input_array.dtype)
            else:
                output_gpu = cp.zeros_like(input_gpu)

            # 确定网格和块大小
            block_size = (256,)  # 每个块的线程数
            grid_size = ( (output_gpu.size + block_size[0] - 1) // block_size[0], )  # 块的数量

            # 执行内核并计时
            start_time = time.time()
            for _ in range(iterations):
                kernel(grid_size, block_size, (input_gpu, output_gpu))
                cp.cuda.Stream.null.synchronize()  # 确保所有计算完成
            elapsed_time = time.time() - start_time

            # 将结果传回CPU
            result = output_gpu.get()

            # 计算性能指标
            throughput = output_gpu.size * iterations / (elapsed_time * 1e6)  # 百万元素/秒

            # 返回结果
            return {
                "method": "CUDA",
                "kernel_name": kernel_name,
                "input_size": input_gpu.size,
                "output_size": output_gpu.size,
                "iterations": iterations,
                "elapsed_time": elapsed_time,
                "throughput": throughput,
                "result_shape": result.shape,
                "gpu_id": task.assigned_gpu_id
            }

        except Exception as e:
            logger.error(f"CUDA自定义内核执行出错: {str(e)}")
            raise

    def _execute_with_opencl(self, task: Task, kernel_code: str, kernel_name: str, 
                            input_data: Any, output_size: int, iterations: int) -> Dict:
        """使用OpenCL执行自定义内核"""
        try:
            import pyopencl as cl
            import pyopencl.array as cl_array

            # 获取GPU设备和上下文
            gpu_id = task.assigned_gpu_id
            if not gpu_id.startswith("opencl-"):
                raise ValueError("无效的OpenCL GPU ID")

            # 从ID中提取平台和设备索引
            parts = gpu_id.split('-')
            platform_idx = int(parts[1])
            device_idx = int(parts[2])

            # 创建上下文和命令队列
            ctx = cl.Context([cl.get_platforms()[platform_idx].get_devices()[device_idx]])
            queue = cl.CommandQueue(ctx)

            # 编译OpenCL内核
            prg = cl.Program(ctx, kernel_code).build()

            # 准备输入数据
            if isinstance(input_data, str):
                # 如果是文件路径，加载数据
                input_array = np.load(input_data)
            else:
                # 如果是直接提供的数据
                input_array = np.array(input_data)

            # 将数据传输到GPU
            input_gpu = cl_array.to_device(queue, input_array)

            # 准备输出缓冲区
            if output_size:
                output_gpu = cl_array.zeros(queue, output_size, dtype=input_array.dtype)
            else:
                output_gpu = cl_array.zeros_like(input_gpu)

            # 确定全局和局部工作大小
            local_size = (256,)  # 每个工作组的线程数
            global_size = ( (output_gpu.size + local_size[0] - 1) // local_size[0] * local_size[0], )  # 总线程数

            # 获取内核函数
            kernel = getattr(prg, kernel_name)

            # 执行内核并计时
            start_time = time.time()
            for _ in range(iterations):
                kernel(queue, global_size, local_size, input_gpu.data, output_gpu.data)
                queue.finish()  # 等待所有操作完成
            elapsed_time = time.time() - start_time

            # 将结果传回CPU
            result = output_gpu.get()

            # 计算性能指标
            throughput = output_gpu.size * iterations / (elapsed_time * 1e6)  # 百万元素/秒

            # 返回结果
            return {
                "method": "OpenCL",
                "kernel_name": kernel_name,
                "input_size": input_gpu.size,
                "output_size": output_gpu.size,
                "iterations": iterations,
                "elapsed_time": elapsed_time,
                "throughput": throughput,
                "result_shape": result.shape,
                "gpu_id": task.assigned_gpu_id
            }

        except Exception as e:
            logger.error(f"OpenCL自定义内核执行出错: {str(e)}")
            raise

# 任务执行器工厂
class TaskExecutorFactory:
    """任务执行器工厂"""

    @staticmethod
    def create_handler(task_type: str, gpu_detector) -> BaseTaskHandler:
        """创建任务处理器"""
        if task_type == TaskTypes.MATRIX_MULTIPLICATION:
            return MatrixMultiplicationHandler(gpu_detector)
        elif task_type == TaskTypes.MODEL_INFERENCE:
            return ModelInferenceHandler(gpu_detector)
        elif task_type == TaskTypes.DATA_PROCESSING:
            return DataProcessingHandler(gpu_detector)
        elif task_type == TaskTypes.CUSTOM_KERNEL:
            return CustomKernelHandler(gpu_detector)
        else:
            raise ValueError(f"不支持的任务类型: {task_type}")有限，回退到CPU执行")
        return self._execute_with_cpu(task, model_path, input_data, batch_size, iterations)

    def _execute_with_cpu(self, task: Task, model_path: str, input_data: Any, 
                         batch_size: int, iterations: int) -> Dict:
        """使用CPU执行模型推理"""
        try:
            import torch
            import torchvision.transforms as transforms

            # 设置设备
            device = torch.device("cpu")

            # 加载模型
            if model_path.endswith('.pth'):
                model = torch.load(model_path, map_location=device)
                model.eval()
            else:
                # 如果是其他格式，需要相应的加载逻辑
                raise ValueError(f"不支持的模型格式: {model_path}")

            # 准备输入数据
            if isinstance(input_data, str):
                # 如果是文件路径，加载数据
                if input_data.endswith('.jpg') or input_data.endswith('.png'):
                    from PIL import Image
                    image = Image.open(input_data)
                    preprocess = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
                    input_tensor = preprocess(image)
                    input_batch = input_tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1)
                else:
                    raise ValueError(f"不支持的输入数据格式: {input_data}")
            else:
                # 如果是直接提供的数据
                input_batch = torch.tensor(input_data)

            # 执行推理并计时
            with torch.no_grad():
                start_time = time.time()
                for _ in range(iterations):
                    output = model(input_batch)
                elapsed_time = time.time() - start_time

            # 计算性能指标
            avg_time_per_batch = elapsed_time / iterations
            throughput = batch_size / avg_time_per_batch  # 样本/秒

            # 返回结果
            return {
                "method": "CPU",
                "model_path": model_path,
                "batch_size": batch_size,
                "iterations": iterations,
                "elapsed_time": elapsed_time,
                "avg_time_per_batch": avg_time_per_batch,
                "throughput": throughput,
                "output_shape": list(output.shape),
                "gpu_id": task.assigned_gpu_id
            }

        except Exception as e:
            logger.error(f"CPU模型推理执行出错: {str(e)}")
            raise

class DataProcessingHandler(BaseTaskHandler):
    """数据处理任务处理器"""

    def execute(self, task: Task) -> Dict:
        """执行数据处理任务"""
        parameters = task.parameters
        operation = parameters.get("operation", "filter")
        data_size = parameters.get("data_size", 1000000)
        iterations = parameters.get("iterations", 1)

        logger.info(f"执行数据处理任务: 操作={operation}, 数据大小={data_size}, 迭代次数={iterations}")

        # 获取GPU信息
        gpu_id = task.assigned_gpu_id
        if not gpu_id:
            raise ValueError("任务未分配GPU")

        gpu_info = self.gpu_detector.get_gpu_by_id(gpu_id)
        if not gpu_info:
            raise ValueError(f"找不到GPU: {gpu_id}")

        # 根据GPU类型选择执行方式
        if gpu_info.gpu_id.startswith("nvidia-"):
            return self._execute_with_cuda(task, operation, data_size, iterations)
        elif gpu_id.startswith("opencl-"):
            return self._execute_with_opencl(task, operation, data_size, iterations)
        else:
            # 使用NumPy在CPU上执行
            return self._execute_with_numpy(task, operation, data_size, iterations)

    def _execute_with_cuda(self, task: Task, operation: str, data_size: int, iterations: int) -> Dict:
        """使用CUDA执行数据处理"""
        try:
            import cupy as cp

            # 创建随机数据
            np.random.seed(42)  # 确保可重复性
            data = cp.random.rand(data_size).astype('float32')

            # 执行数据处理并计时
            start_time = time.time()
            for _ in range(iterations):
                if operation == "filter":
                    # 简单的过滤操作：保留大于0.5的值
                    result = cp.where(data > 0.5, data, 0)
                elif operation == "transform":
                    # 简单的变换操作：平方
                    result = cp.square(data)
                elif operation == "reduce":
                    # 简单的归约操作：求和
                    result = cp.sum(data)
                else:
                    raise ValueError(f"不支持的数据处理操作: {operation}")

            # 确保所有计算完成
            cp.cuda.Stream.null.synchronize()
            elapsed_time = time.time() - start_time

            # 计算性能指标
            throughput = data_size * iterations / elapsed_time  # 元素/秒

            # 返回结果
            return {
                "method": "CUDA",
                "operation": operation,
                "data_size": data_size,
                "iterations": iterations,
                "elapsed_time": elapsed_time,
                "throughput": throughput,
                "result_type": type(result).__name__,
                "gpu_id": task.assigned_gpu_id
            }

        except ImportError:
            logger.warning("CuPy不可用，回退到NumPy执行")
            return self._execute_with_numpy(task, operation, data_size, iterations)
        except Exception as e:
            logger.error(f"CUDA数据处理执行出错: {str(e)}")
            raise

    def _execute_with_opencl(self, task: Task, operation: str, data_size: int, iterations: int) -> Dict:
        """使用OpenCL执行数据处理"""
        try:
            import pyopencl as cl
            import pyopencl.array as cl_array

            # 获取GPU设备和上下文
            gpu_id = task.assigned_gpu_id
            if not gpu_id.startswith("opencl-"):
                raise ValueError("无效的OpenCL GPU ID")

            # 从ID中提取平台和设备索引
            parts = gpu_id.split('-')
            platform_idx = int(parts[1])
            device_idx = int(parts[2])

            # 创建上下文和命令队列
            ctx = cl.Context([cl.get_platforms()[platform_idx].get_devices()[device_idx]])
            queue = cl.CommandQueue(ctx)

            # 创建随机数据
            np.random.seed(42)  # 确保可重复性
            data_np = np.random.rand(data_size).astype('float32')

            # 将数据传输到GPU
            data_g = cl_array.to_device(queue, data_np)

            # 执行数据处理并计时
            start_time = time.time()
            for _ in range(iterations):
                if operation == "filter":
                    # 简单的过滤操作：保留大于0.5的值
                    result_g = cl_array.zeros_like(data_g)
                    kernel_code = """
                    __kernel void filter(__global const float* input, __global float* output, int size) {
                        int idx = get_global_id(0);
                        if (idx < size) {
                            if (input[idx] > 0.5f) {
                                output[idx] = input[idx];
                            } else {
                                output[idx] = 0.0f;
                            }
                        }
                    }
                    """
                    prg = cl.Program(ctx, kernel_code).build()
                    prg.filter(queue, (data_size,), None, data_g.data, result_g.data, np.int32(data_size))
                    result = result_g.get()
                elif operation == "transform":
                    # 简单的变换操作：平方
                    result_g = cl_array.zeros_like(data_g)
                    kernel_code = """
                    __kernel void transform(__global const float* input, __global float* output, int size) {
                        int idx = get_global_id(0);
                        if (idx < size) {
                            output[idx] = input[idx] * input[idx];
                        }
                    }
                    """
                    prg = cl.Program(ctx, kernel_code).build()
                    prg.transform(queue, (data_size,), None, data_g.data, result_g.data, np.int32(data_size))
                    result = result_g.get()
                elif operation == "reduce":
                    # 简单的归约操作：求和
                    result = cl_array.sum(data_g).get()
                else:
                    raise ValueError(f"不支持的数据处理操作: {operation}")

                queue.finish()  # 等待所有操作完成

            elapsed_time = time.time() - start_time

            # 计算性能指标
            throughput = data_size * iterations / elapsed_time  # 元素/秒

            # 返回结果
            return {
                "method": "OpenCL",
                "operation": operation,
                "data_size": data_size,
                "iterations": iterations,
                "elapsed_time": elapsed_time,
                "throughput": throughput,
                "result_type": type(result).__name__,
                "gpu_id": task.assigned_gpu_id
            }

        except ImportError:
            logger.warning("PyOpenCL不可用，回退到NumPy执行")
            return self._execute_with_numpy(task, operation, data_size, iterations)
        except Exception as e:
            logger.error(f"OpenCL数据处理执行出错: {str(e)}")
            raise

    def _execute_with_numpy(self, task: Task, operation: str, data_size: int, iterations: int) -> Dict:
        """使用NumPy执行数据处理"""
        # 创建随机数据
        np.random.seed(42)  # 确保可重复性
        data = np.random.rand(data_size).astype('float32')

        # 执行数据处理并计时
        start_time = time.time()
        for _ in range(iterations):
            if operation == "filter":
                # 简单的过滤操作：保留大于0.5的值
                result = np.where(data > 0.5, data, 0)
            elif operation == "transform":
                # 简单的变换操作：平方
                result = np.square(data)
            elif operation == "reduce":
                # 简单的归约操作：求和
                result = np.sum(data)
            else:
                raise ValueError(f"不支持的数据处理操作: {operation}")

        elapsed_time = time.time() - start_time

        # 计算性能指标
        throughput = data_size * iterations / elapsed_time  # 元素/秒

        # 返回结果
        return {
            "method": "NumPy",
            "operation": operation,
            "data_size": data_size,
            "iterations": iterations,
            "elapsed_time": elapsed_time,
            "throughput": throughput,
            "result_type": type(result).__name__,
            "gpu_id": task.assigned_gpu_id
        }

class CustomKernelHandler(BaseTaskHandler):
    """自定义内核任务处理器"""

    def execute(self, task: Task) -> Dict:
        """执行自定义内核任务"""
        parameters = task.parameters
        kernel_code = parameters.get("kernel_code")
        kernel_name = parameters.get("kernel_name", "custom_kernel")
        input_data = parameters.get("input_data")
        output_size = parameters.get("output_size")
        iterations = parameters.get("iterations", 1)

        logger.info(f"执行自定义内核任务: 内核={kernel_name}, 迭代次数={iterations}")

        # 获取GPU信息
        gpu_id = task.assigned_gpu_id
        if not gpu_id:
            raise ValueError("任务未分配GPU")

        gpu_info = self.gpu_detector.get_gpu_by_id(gpu_id)
        if not gpu_info:
            raise ValueError(f"找不到GPU: {gpu_id}")

        # 根据GPU类型选择执行方式
        if gpu_info.gpu_id.startswith("nvidia-"):
            return self._execute_with_cuda(task, kernel_code, kernel_name, input_data, output_size, iterations)
        elif gpu_id.startswith("opencl-"):
            return self._execute_with_opencl(task, kernel_code, kernel_name, input_data, output_size, iterations)
        else:
            raise ValueError("自定义内核任务需要GPU支持")

    def _execute_with_cuda(self, task: Task, kernel_code: str, kernel_name: str, 
                          input_data: Any, output_size: int, iterations: int) -> Dict:
        """使用CUDA执行自定义内核"""
        try:
            import cupy as cp
            from cupy.cuda.compiler import compile_with_cache

            # 创建临时文件保存内核代码
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
                f.write(kernel_code)
                kernel_path = f.name

            try:
                # 编译CUDA内核
                module = compile_with_cache(kernel_code)
                kernel = module.get_function(kernel_name)

                # 准备输入数据
                if isinstance(input_data, str):
                    # 如果是文件路径，加载数据
                    input_array = np.load(input_data)
                else:
                    # 如果是直接提供的数据
                    input_array = np.array(input_data)

                # 将数据传输到GPU
                input_gpu = cp.asarray(input_array)

                # 准备输出数据
                if output_size:
                    output_gpu = cp.zeros(output_size, dtype=input_gpu.dtype)
                else:
                    output_gpu = cp.zeros_like(input_gpu)

                # 确定网格和块大小
                block_size = (256,)
                grid_size = ((output_gpu.size + block_size[0] - 1) // block_size[0],)

                # 预热GPU
                kernel(grid_size, block_size, (input_gpu, output_gpu))
                cp.cuda.Stream.null.synchronize()

                # 执行内核并计时
                start_time = time.time()
                for _ in range(iterations):
                    kernel(grid_size, block_size, (input_gpu, output_gpu))
                    cp.cuda.Stream.null.synchronize()  # 确保所有计算完成
                elapsed_time = time.time() - start_time

                # 将结果传回CPU
                output_result = cp.asnumpy(output_gpu)

                # 计算性能指标
                elements_processed = output_gpu.size * iterations
                throughput = elements_processed / elapsed_time  # 元素/秒

                # 返回结果
                return {
                    "method": "CUDA",
                    "kernel_name": kernel_name,
                    "input_shape": input_array.shape,
                    "output_shape": output_result.shape,
                    "iterations": iterations,
                    "elapsed_time": elapsed_time,
                    "throughput": throughput,
                    "gpu_id": task.assigned_gpu_id
                }

            finally:
                # 删除临时文件
                os.unlink(kernel_path)

        except ImportError:
            logger.error("CuPy不可用，无法执行CUDA自定义内核")
            raise
        except Exception as e:
            logger.error(f"CUDA自定义内核执行出错: {str(e)}")
            raise

    def _execute_with_opencl(self, task: Task, kernel_code: str, kernel_name: str, 
                           input_data: Any, output_size: int, iterations: int) -> Dict:
        """使用OpenCL执行自定义内核"""
        try:
            import pyopencl as cl
            import pyopencl.array as cl_array

            # 获取GPU设备和上下文
            gpu_id = task.assigned_gpu_id
            if not gpu_id.startswith("opencl-"):
                raise ValueError("无效的OpenCL GPU ID")

            # 从ID中提取平台和设备索引
            parts = gpu_id.split('-')
            platform_idx = int(parts[1])
            device_idx = int(parts[2])

            # 创建上下文和命令队列
            ctx = cl.Context([cl.get_platforms()[platform_idx].get_devices()[device_idx]])
            queue = cl.CommandQueue(ctx)

            # 准备输入数据
            if isinstance(input_data, str):
                # 如果是文件路径，加载数据
                input_array = np.load(input_data)
            else:
                # 如果是直接提供的数据
                input_array = np.array(input_data)

            # 将数据传输到GPU
            input_gpu = cl_array.to_device(queue, input_array)

            # 准备输出数据
            if output_size:
                output_gpu = cl_array.zeros(queue, output_size, dtype=input_gpu.dtype)
            else:
                output_gpu = cl_array.zeros_like(input_gpu)

            # 编译OpenCL内核
            prg = cl.Program(ctx, kernel_code).build()
            kernel = getattr(prg, kernel_name)

            # 确定全局和局部工作大小
            local_size = (256,)
            global_size = ((output_gpu.size + local_size[0] - 1) // local_size[0] * local_size[0],)

            # 预热GPU
            kernel(queue, global_size, local_size, input_gpu.data, output_gpu.data)
            queue.finish()

            # 执行内核并计时
            start_time = time.time()
            for _ in range(iterations):
                kernel(queue, global_size, local_size, input_gpu.data, output_gpu.data)
                queue.finish()  # 等待所有操作完成
            elapsed_time = time.time() - start_time

            # 将结果传回CPU
            output_result = output_gpu.get()

            # 计算性能指标
            elements_processed = output_gpu.size * iterations
            throughput = elements_processed / elapsed_time  # 元素/秒

            # 返回结果
            return {
                "method": "OpenCL",
                "kernel_name": kernel_name,
                "input_shape": input_array.shape,
                "output_shape": output_result.shape,
                "iterations": iterations,
                "elapsed_time": elapsed_time,
                "throughput": throughput,
                "gpu_id": task.assigned_gpu_id
            }

        except ImportError:
            logger.error("PyOpenCL不可用，无法执行OpenCL自定义内核")
            raise
        except Exception as e:
            logger.error(f"OpenCL自定义内核执行出错: {str(e)}")
            raise

# 任务执行器工厂
class TaskExecutorFactory:
    """任务执行器工厂"""

    @staticmethod
    def create_handler(task_type: str, gpu_detector) -> BaseTaskHandler:
        """创建任务处理器"""
        if task_type == TaskTypes.MATRIX_MULTIPLICATION:
            return MatrixMultiplicationHandler(gpu_detector)
        elif task_type == TaskTypes.MODEL_INFERENCE:
            return ModelInferenceHandler(gpu_detector)
        elif task_type == TaskTypes.DATA_PROCESSING:
            return DataProcessingHandler(gpu_detector)
        elif task_type == TaskTypes.CUSTOM_KERNEL:
            return CustomKernelHandler(gpu_detector)
        else:
            raise ValueError(f"不支持的任务类型: {task_type}")
