#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能优化工具模块
提供内存管理、GPU检测等工具函数
"""

import gc
import os
import cv2
import numpy as np
from typing import Optional, Tuple

def force_memory_cleanup():
    """
    强制内存清理
    在处理大文件或长时间运行后调用
    """
    gc.collect()
    try:
        # 如果使用CUDA，清理GPU内存
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            # OpenCV CUDA内存管理
            pass  # OpenCV的CUDA会自动管理
    except:
        pass


def get_optimal_chunk_size(file_size_bytes: int) -> int:
    """
    根据文件大小返回最优的chunk大小
    
    参数:
        file_size_bytes: 文件大小（字节）
    
    返回:
        最优chunk大小（字节）
    """
    if file_size_bytes < 100 * 1024 * 1024:  # < 100MB
        return 8 * 1024 * 1024  # 8MB
    elif file_size_bytes < 500 * 1024 * 1024:  # < 500MB
        return 16 * 1024 * 1024  # 16MB
    else:  # >= 500MB
        return 32 * 1024 * 1024  # 32MB


def get_gpu_info() -> dict:
    """
    获取GPU信息
    
    返回:
        包含GPU状态的字典
    """
    info = {
        'cuda_available': False,
        'cuda_devices': 0,
        'current_device': None,
        'device_name': None,
    }
    
    try:
        info['cuda_devices'] = cv2.cuda.getCudaEnabledDeviceCount()
        if info['cuda_devices'] > 0:
            info['cuda_available'] = True
            info['current_device'] = cv2.cuda.getDevice()
    except:
        pass
    
    return info


def optimize_video_writer_params(fps: float, quality: str = 'balanced') -> Tuple[str, dict]:
    """
    返回优化的视频编码器参数
    
    参数:
        fps: 帧率
        quality: 'fast' (最快), 'balanced' (平衡), 'quality' (高质量)
    
    返回:
        (codec, params) 元组
    """
    # 尝试使用的编码器（按速度排序）
    if quality == 'fast':
        # 最快模式：牺牲一些质量换取速度
        codecs = [
            ('avc1', {}),  # H.264（最快）
            ('X264', {}),
            ('mp4v', {}),
        ]
    elif quality == 'quality':
        # 高质量模式：稍慢但质量更好
        codecs = [
            ('avc1', {}),
            ('H264', {}),
            ('mp4v', {}),
        ]
    else:  # balanced
        # 平衡模式（默认）
        codecs = [
            ('avc1', {}),
            ('H264', {}),
            ('X264', {}),
            ('mp4v', {}),
        ]
    
    return codecs


def check_opencv_optimizations() -> dict:
    """
    检查OpenCV是否启用了优化
    
    返回:
        优化状态字典
    """
    info = {
        'num_threads': cv2.getNumThreads(),
        'use_optimized': cv2.useOptimized(),
        'build_info': {},
    }
    
    # 解析构建信息
    build = cv2.getBuildInformation()
    for line in build.split('\n'):
        if 'CUDA' in line or 'OPENCL' in line or 'TBB' in line or 'IPP' in line:
            info['build_info'][line.strip()] = True
    
    return info


def auto_set_opencv_threads(num_videos: int = 1):
    """
    自动设置OpenCV线程数
    
    参数:
        num_videos: 同时处理的视频数量
    """
    import os
    cpu_count = os.cpu_count() or 4
    
    # 如果只处理一个视频，使用所有CPU核心
    if num_videos == 1:
        optimal_threads = max(1, cpu_count - 1)  # 留一个核心给系统
    else:
        # 多视频并行时，每个视频分配较少线程
        optimal_threads = max(1, (cpu_count - 1) // num_videos)
    
    cv2.setNumThreads(optimal_threads)
    return optimal_threads


def enable_opencv_optimizations():
    """
    启用OpenCV优化
    在程序启动时调用一次
    """
    # 启用优化代码
    cv2.setUseOptimized(True)
    
    # 如果有多线程支持，自动设置线程数
    auto_set_opencv_threads()
    
    print(f"✓ OpenCV优化已启用")
    print(f"  - 线程数: {cv2.getNumThreads()}")
    print(f"  - 优化代码: {cv2.useOptimized()}")


if __name__ == '__main__':
    """测试性能工具"""
    print("=" * 50)
    print("性能优化工具测试")
    print("=" * 50)
    
    # GPU信息
    gpu_info = get_gpu_info()
    print("\n🎮 GPU信息:")
    print(f"  CUDA可用: {gpu_info['cuda_available']}")
    print(f"  CUDA设备数: {gpu_info['cuda_devices']}")
    if gpu_info['device_name']:
        print(f"  设备名称: {gpu_info['device_name']}")
    
    # OpenCV优化
    print("\n⚙️  OpenCV优化状态:")
    enable_opencv_optimizations()
    
    opt_info = check_opencv_optimizations()
    print(f"  构建信息: {len(opt_info['build_info'])} 项优化特性")
    
    print("\n✓ 测试完成")

