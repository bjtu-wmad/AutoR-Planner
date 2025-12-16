#!/usr/bin/env python3
"""
Epona自定义测试数据生成脚本
自动生成模拟的驾驶场景数据用于测试
"""

import os
import argparse
import numpy as np
import cv2
from pathlib import Path


def create_synthetic_frame(frame_idx, scenario='straight', size=(512, 1024)):
    """
    创建合成测试图像
    
    Args:
        frame_idx: 帧索引
        scenario: 场景类型 ('straight', 'left_turn', 'right_turn')
        size: 图像大小 (height, width)
    """
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 天空背景
    img[:h//2, :] = [135, 206, 235]  # 天蓝色
    
    # 地面
    img[h//2:, :] = [100, 100, 100]  # 灰色
    
    # 绘制道路
    road_left = w//4
    road_right = 3*w//4
    
    if scenario == 'left_turn':
        # 左转:道路逐渐向左偏移
        offset = int(frame_idx * w / 40)
        road_left -= offset
        road_right -= offset
    elif scenario == 'right_turn':
        # 右转:道路逐渐向右偏移 
        offset = int(frame_idx * w / 40)
        road_left += offset
        road_right += offset
    
    cv2.rectangle(img, (max(0, road_left), h//2), 
                  (min(w, road_right), h), (50, 50, 50), -1)
    
    # 绘制车道线(动画效果)
    lane_spacing = 50
    lane_offset = (frame_idx * 10) % (lane_spacing * 2)
    
    for i in range(-lane_offset, h, lane_spacing):
        y1 = h//2 + i
        y2 = y1 + lane_spacing//2
        if y1 < h and y2 < h:
            cv2.line(img, (w//2-2, y1), (w//2+2, min(y2, h-1)), 
                    (255, 255, 255), 3)
    
    # 绘制左右路边线
    cv2.line(img, (max(0, road_left), h//2), 
            (max(0, road_left), h), (255, 255, 0), 3)
    cv2.line(img, (min(w, road_right), h//2), 
            (min(w, road_right), h), (255, 255, 0), 3)
    
    # 添加帧编号和场景信息
    scenario_text = scenario.replace('_', ' ').title()
    cv2.putText(img, f"Frame {frame_idx:02d} - {scenario_text}", 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return img


def generate_trajectory_straight(num_frames):
    """生成直线行驶轨迹"""
    poses = []
    yaws = []
    
    for i in range(num_frames):
        # 匀速直行,每帧前进1米
        poses.append([1.0, 0.0])
        yaws.append([0.0])
    
    return np.array([poses]), np.array([yaws])


def generate_trajectory_left_turn(num_frames):
    """生成左转弯轨迹"""
    poses = []
    yaws = []
    
    for i in range(num_frames):
        t = i / num_frames
        
        # 逐渐减速并向左偏移
        dx = 1.0 - 0.4 * t          # 前进速度减缓
        dy = -0.08 * t * t          # 向左偏移(y为负)
        yaw_angle = 2.0 + 4.0 * t   # 左转角度逐渐增加
        
        poses.append([dx, dy])
        yaws.append([yaw_angle])
    
    return np.array([poses]), np.array([yaws])


def generate_trajectory_right_turn(num_frames):
    """生成右转弯轨迹"""
    poses = []
    yaws = []
    
    for i in range(num_frames):
        t = i / num_frames
        
        # 逐渐减速并向右偏移
        dx = 1.0 - 0.3 * t          # 前进速度减缓
        dy = 0.08 * t * t           # 向右偏移(y为正)
        yaw_angle = -2.0 - 3.0 * t  # 右转角度逐渐增加(负值)
        
        poses.append([dx, dy])
        yaws.append([yaw_angle])
    
    return np.array([poses]), np.array([yaws])


def generate_video_data(output_dir, video_name, scenario='straight', 
                       num_frames=10, image_size=(512, 1024)):
    """
    生成一个完整的测试视频数据
    
    Args:
        output_dir: 输出目录
        video_name: 视频名称
        scenario: 场景类型 ('straight', 'left_turn', 'right_turn')
        num_frames: 帧数
        image_size: 图像大小
    """
    video_dir = Path(output_dir) / video_name
    video_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"生成视频: {video_name} ({scenario}, {num_frames}帧)")
    
    # 1. 生成图像
    for i in range(num_frames):
        img = create_synthetic_frame(i, scenario, image_size)
        img_path = video_dir / f"{i:06d}.png"
        cv2.imwrite(str(img_path), img)
    
    # 2. 生成轨迹数据
    if scenario == 'straight':
        pose, yaw = generate_trajectory_straight(num_frames + 1)
    elif scenario == 'left_turn':
        pose, yaw = generate_trajectory_left_turn(num_frames + 1)
    elif scenario == 'right_turn':
        pose, yaw = generate_trajectory_right_turn(num_frames + 1)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
        raise ValueError(f"Unknown scenario: {scenario}")
    
    # 3. 保存numpy文件
    np.save(video_dir / 'pose.npy', pose)
    np.save(video_dir / 'yaw.npy', yaw)
    
    # 4. 验证数据
    print(f"  ✅ 生成 {num_frames} 帧图像")
    print(f"  ✅ pose shape: {pose.shape}")
    print(f"  ✅ yaw shape: {yaw.shape}")
    print(f"  📁 保存到: {video_dir}")
    
    return video_dir


def main():
    parser = argparse.ArgumentParser(description='生成Epona测试数据')
    parser.add_argument('--output_dir', type=str, default='data',
                       help='输出目录 (default: data)')
    parser.add_argument('--num_videos', type=int, default=3,
                       help='生成视频数量 (default: 3)')
    parser.add_argument('--num_frames', type=int, default=10,
                       help='每个视频的帧数 (default: 10)')
    parser.add_argument('--image_size', type=str, default='512x1024',
                       help='图像大小 HxW (default: 512x1024)')
    
    args = parser.parse_args()
    
    # 解析图像大小
    h, w = map(int, args.image_size.split('x'))
    image_size = (h, w)
    
    print("="*60)
    print("Epona 自定义测试数据生成器")
    print("="*60)
    print(f"输出目录: {args.output_dir}")
    print(f"视频数量: {args.num_videos}")
    print(f"每视频帧数: {args.num_frames}")
    print(f"图像大小: {h}x{w}")
    print("="*60 + "\n")
    
    # 场景列表
    scenarios = ['straight', 'left_turn', 'right_turn']
    
    # 生成多个视频
    for i in range(args.num_videos):
        scenario = scenarios[i % len(scenarios)]
        video_name = f"video-{i+1:02d}"
        
        generate_video_data(
            output_dir=args.output_dir,
            video_name=video_name,
            scenario=scenario,
            num_frames=args.num_frames,
            image_size=image_size
        )
        print()
    
    print("="*60)
    print(f"✅ 成功生成 {args.num_videos} 个测试视频!")
    print("="*60)
    print("\n下一步:")
    print(f"1. 查看生成的数据: ls -lh {args.output_dir}/*/")
    print("2. 运行测试脚本:")
    print("   python scripts/test/test_demo.py ")
    print("       --exp_name 'demo_test' ")
    print("       --resume_path 'pretrained/epona_nuplan.pkl' ")
    print("       --config 'configs/dit_config_dcae_nuplan_cached.py'")
    print()


if __name__ == '__main__':
    main()
