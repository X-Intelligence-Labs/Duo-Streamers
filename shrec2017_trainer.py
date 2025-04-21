import os
import copy
import itertools
import logging
import glob
from tqdm import tqdm
from functools import partial
from datetime import datetime
from random import randint, shuffle

import numpy as np
from numpy import genfromtxt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR

import argparse
import logging
import tempfile
import uuid
from argparse import Namespace
from typing import Literal, Optional, Tuple, Union

from torcheval.metrics import MulticlassAccuracy, MulticlassConfusionMatrix, MulticlassF1Score
from torch.utils.tensorboard import SummaryWriter

# 设置日志记录
_logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# 数据集路径和配置
MAIN_FOLDER = "./"  # 当前目录作为主目录

# 14种手势类别
GESTURE_NAME = [
    "Grab", "Tap", "Expand", "Pinch", "Rotation CW", "Rotation CCW",
    "Swipe Right", "Swipe Left", "Swipe Up", "Swipe Down", "Swipe X",
    "Swipe V", "Swipe +", "Shake"
]

# 完整的28个类别（14种手势 × 2种手指配置）
FULL_GESTURE_NAME = []
for gesture in GESTURE_NAME:
    FULL_GESTURE_NAME.append(f"{gesture} (单指)")
    FULL_GESTURE_NAME.append(f"{gesture} (整手)")

# 定义关节名称
# SHREC2017的骨架结构包含22个关节点，每个关节点有3D坐标
JOINT_NAME = [
    "palm",
    "thumb_base", "thumb_middle", "thumb_end",
    "index_base", "index_middle", "index_middle2", "index_end",
    "middle_base", "middle_middle", "middle_middle2", "middle_end",
    "ring_base", "ring_middle", "ring_middle2", "ring_end",
    "pinky_base", "pinky_middle", "pinky_middle2", "pinky_end",
    "wrist1", "wrist2"
]

# 默认使用完整的28个类别，可通过命令行参数修改
SELECTED_GESTURE_NAME = FULL_GESTURE_NAME
SELECTED_JOINT_NAME = JOINT_NAME

# 数据维度
JOINTS_PER_FRAME = 22  # 每帧的关节数
COORDS_PER_JOINT = 3   # 每个关节的坐标维度 (x,y,z)
INPUT_DIM = len(SELECTED_JOINT_NAME) * COORDS_PER_JOINT  # 输入维度=22*3=66
OUTPUT_DIM = len(SELECTED_GESTURE_NAME)  # 输出维度=28（类别数）
MAX_LENGTH = 100  # 最大序列长度
BATCH_SIZE = 8   # 批次大小

# 读取SHREC2017数据集的函数
def read_skeleton_file(file_path):
    """
    读取单个骨架文件（skeletons_world.txt）
    """
    try:
        data = genfromtxt(file_path, delimiter=' ')
        return data
    except:
        _logger.error(f"无法读取文件: {file_path}")
        return None

def load_gesture_data(gesture_id, finger_type, subject_id, trial_id):
    """
    加载特定手势、手指类型、受试者和试验的骨架数据
    """
    path = os.path.join(MAIN_FOLDER, f"gesture_{gesture_id}", f"finger_{finger_type}", 
                        f"subject_{subject_id}", f"essai_{trial_id}", "skeletons_world.txt")
    return read_skeleton_file(path)

def read_labels_file(file_path):
    """
    读取标签文件（train_gestures.txt 或 test_gestures.txt）
    返回：包含所有样本信息的字典列表
    """
    labels = []
    label_counts = {}  # 用于统计标签分布
    
    _logger.info(f"正在读取标签文件: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split()
                if len(parts) >= 7:
                    try:
                        label = int(parts[5])
                        # 统计标签出现次数
                        if label not in label_counts:
                            label_counts[label] = 0
                        label_counts[label] += 1
                        
                        sample = {
                            'gesture_id': int(parts[0]),
                            'finger_type': int(parts[1]),
                            'subject_id': int(parts[2]),
                            'trial_id': int(parts[3]),
                            'gesture_class': int(parts[4]),
                            'label': label,
                            'num_frames': int(parts[6])
                        }
                        labels.append(sample)
                    except ValueError as e:
                        _logger.error(f"解析错误 (行 {line_num}): {e} - {line.strip()}")
                else:
                    _logger.warning(f"跳过格式不正确的行 {line_num}: {line.strip()}")
    except Exception as e:
        _logger.error(f"读取文件 {file_path} 时出错: {e}")
    
    # 输出标签分布信息
    _logger.info(f"标签分布: {sorted(label_counts.items())}")
    _logger.info(f"总共读取了 {len(labels)} 个样本")
    
    return labels

def load_gesture_dataset(labels_file, max_samples=None, mode='full'):
    """
    加载数据集
    labels_file: 标签文件路径（train_gestures.txt 或 test_gestures.txt）
    max_samples: 限制加载的样本数量（用于测试）
    mode: 'single_finger'表示只使用单指手势(finger_1)，'full'表示使用所有手势
    """
    labels = read_labels_file(labels_file)
    if max_samples:
        labels = labels[:max_samples]
    
    gesture_data = []
    gesture_labels = []
    skipped_samples = 0
    
    for i, sample in enumerate(tqdm(labels, desc="加载数据集")):
        # 如果模式是'single_finger'，只加载finger_1的样本
        if mode == 'single_finger' and sample['finger_type'] != 1:
            skipped_samples += 1
            continue
            
        data = load_gesture_data(
            sample['gesture_id'],
            sample['finger_type'],
            sample['subject_id'],
            sample['trial_id']
        )
        
        if data is not None:
            gesture_data.append(data)
            
            # 计算标签：(gesture_id - 1) * 2 + (finger_type - 1)
            # 这将把14种手势 × 2种手指配置映射到0-27的范围
            if mode == 'full':
                adjusted_label = (sample['gesture_id'] - 1) * 2 + (sample['finger_type'] - 1)
                gesture_labels.append(adjusted_label)
            else:  # single_finger模式
                # 单指模式下，标签范围是0-13
                gesture_labels.append(sample['gesture_id'] - 1)
    
    if skipped_samples > 0:
        _logger.info(f"根据选择的模式 '{mode}'，跳过了 {skipped_samples} 个样本")
    
    _logger.info(f"成功加载 {len(gesture_data)} 个样本")
    
    return gesture_data, gesture_labels

def normalize_skeleton(skeleton_data):
    """
    归一化骨架数据
    """
    # 简单的减均值除以标准差归一化
    mean = np.mean(skeleton_data, axis=0)
    std = np.std(skeleton_data, axis=0)
    std[std == 0] = 1  # 避免除以零
    return (skeleton_data - mean) / std

# 数据预处理函数
def pad_data(data, input_dim=INPUT_DIM, max_length=MAX_LENGTH):
    """
    将骨架序列填充或截断到指定长度
    """
    data_padded = np.zeros([len(data), max_length, input_dim])
    for i in range(len(data)):
        if len(data[i]) <= max_length:
            data_padded[i, :len(data[i])] = data[i][:, :input_dim]  # 只取前input_dim个元素
        else:
            data_padded[i] = data[i][:max_length, :input_dim]
    
    return data_padded

# 数据增强
def translate_joints(joints, translation_range):
    """平移增强"""
    translation = np.random.uniform(-translation_range, translation_range, size=(1, 3))
    translated_joints = joints + translation.reshape(1, 1, 3)
    return translated_joints

def rotate_joints(joints, rotation_range):
    """旋转增强"""
    angle = np.random.uniform(-rotation_range, rotation_range)
    axis = np.random.normal(size=3)
    axis /= np.linalg.norm(axis)
    r = R.from_rotvec(angle * axis)
    rotated_joints = r.apply(joints.reshape(-1, 3)).reshape(joints.shape)
    return rotated_joints

def scale_joints(joints, scale_range):
    """缩放增强"""
    scale = np.random.uniform(1 - scale_range, 1 + scale_range)
    return joints * scale

def add_noise(joints, noise_std):
    """添加噪声"""
    return joints + np.random.normal(0, noise_std, size=joints.shape)

def time_warp(sequence, speed_factor):
    """时间扭曲增强"""
    num_frames, num_features = sequence.shape
    new_num_frames = int(num_frames / speed_factor)
    
    if new_num_frames <= 1:  # 避免过度扭曲
        return sequence
    
    # 插值索引
    original_indices = np.arange(num_frames)
    new_indices = np.linspace(0, num_frames - 1, new_num_frames)
    
    # 对每个特征进行插值
    warped_sequence = np.zeros((new_num_frames, num_features))
    for i in range(num_features):
        interp_function = interp1d(original_indices, sequence[:, i], kind='linear', bounds_error=False, fill_value="extrapolate")
        warped_sequence[:, i] = interp_function(new_indices)
    
    # 填充或截断到指定长度
    data_padded = np.zeros((MAX_LENGTH, num_features))
    data_padded[:min(warped_sequence.shape[0], MAX_LENGTH)] = warped_sequence[:min(warped_sequence.shape[0], MAX_LENGTH)]
    
    return data_padded.astype(np.float32)

def augment_skeleton(skeleton, translation_range=0.1, rotation_range=np.pi/8, scale_range=0.1, noise_std=0.01):
    """组合多种增强方法"""
    # 将骨架重塑为帧数 x 关节数 x 3
    num_frames, num_features = skeleton.shape
    joints_per_frame = num_features // 3
    skeleton_reshaped = skeleton.reshape(num_frames, joints_per_frame, 3)
    
    # 应用增强
    skeleton_augmented = translate_joints(skeleton_reshaped, translation_range)
    skeleton_augmented = rotate_joints(skeleton_augmented, rotation_range)
    skeleton_augmented = scale_joints(skeleton_augmented, scale_range)
    skeleton_augmented = add_noise(skeleton_augmented, noise_std)
    
    # 重塑回原始形状
    return skeleton_augmented.reshape(num_frames, num_features).astype(np.float32)

# 自定义数据集类
class SHREC2017Dataset(Dataset):
    def __init__(self, data, labels, augment=False, time_warp_prob=0.7):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.augment = augment
        self.time_warp_prob = time_warp_prob

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx].numpy()
        
        if self.augment:
            # 应用增强
            if np.random.random() < 0.9:  # 70%的概率应用骨架增强
                d = augment_skeleton(d)
            
            # 应用时间扭曲
            if np.random.random() < self.time_warp_prob:
                speed_factor = np.random.uniform(0.8, 1.2)  # 速度因子
                d = time_warp(d, speed_factor)
        
        return torch.tensor(d, dtype=torch.float32), self.labels[idx]

# 模型定义
class StreamingSightMu(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=512, dropout=0.3, num_layers=6):
        super(StreamingSightMu, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_dim = input_dim
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, hidden_size)
        
        # 关节注意力机制 - 对骨架的不同关节点赋予不同权重
        self.joint_attention = nn.Sequential(
            nn.Linear(input_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 22),  # 假设有22个关节点
            nn.Softmax(dim=-1)
        )
        
        # 门控递归单元 - 类似于GRU的门控机制
        # 更新门
        self.update_gates = nn.ModuleList([
            nn.Linear(hidden_size + hidden_size, hidden_size)
            for _ in range(num_layers)
        ])
        
        # 重置门
        self.reset_gates = nn.ModuleList([
            nn.Linear(hidden_size + hidden_size, hidden_size)
            for _ in range(num_layers)
        ])
        
        # 候选隐藏状态生成
        self.candidate_layers = nn.ModuleList([
            nn.Linear(hidden_size + hidden_size, hidden_size)
            for _ in range(num_layers)
        ])
        
        # Dropout层
        self.dropouts = nn.ModuleList([
            nn.Dropout(p=dropout)
            for _ in range(num_layers)
        ])
        
        # 残差连接
        self.residual_projections = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size)
            for _ in range(num_layers - 1)
        ])
        
        # 输出层
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        self.output_activation = nn.ReLU()
        self.output_dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, output_dim)
    
    def forward(self, x, h):
        """
        x: 输入张量，形状为 [batch_size, seq_length=1, input_size]
        h: 外部隐藏状态，形状为 [num_layers, batch_size, hidden_size]
        """
        batch_size, seq_length, input_size = x.size()
        x = x.squeeze(1)  # [batch_size, input_size]
        
        # 应用关节注意力 - 找出重要的关节点
        joint_weights = self.joint_attention(x)  # [batch_size, 22]
        # 将关节点权重应用到输入特征上
        # 假设输入x的组织方式是[batch_size, 22*3]，每个关节有3个坐标
        x_reshaped = x.view(batch_size, 22, 3)
        x_weighted = x_reshaped * joint_weights.unsqueeze(-1)
        x = x_weighted.reshape(batch_size, input_size)
        
        # 输入投影
        x = self.input_projection(x)
        
        # 通过多层门控更新
        new_h = []
        
        for i in range(self.num_layers):
            # 获取当前层的隐藏状态
            curr_h = h[i]
            
            # 连接输入和隐藏状态
            combined = torch.cat([x, curr_h], dim=-1)
            
            # 计算更新门和重置门
            update_gate = torch.sigmoid(self.update_gates[i](combined))
            reset_gate = torch.sigmoid(self.reset_gates[i](combined))
            
            # 计算候选隐藏状态
            reset_hidden = reset_gate * curr_h
            candidate_input = torch.cat([x, reset_hidden], dim=-1)
            candidate_hidden = torch.tanh(self.candidate_layers[i](candidate_input))
            
            # 更新隐藏状态
            new_hidden = (1 - update_gate) * curr_h + update_gate * candidate_hidden
            
            # 应用dropout
            new_hidden = self.dropouts[i](new_hidden)
            
            # 保存新隐藏状态
            new_h.append(new_hidden.detach())
            
            # 为下一层准备输入，添加残差连接
            if i < self.num_layers - 1:
                x = x + self.residual_projections[i](new_hidden)
            else:
                x = new_hidden
        
        # 最终输出层
        x = self.output_projection(x)
        x = self.output_activation(x)
        x = self.output_dropout(x)
        output = self.fc(x)
        
        # 返回输出和更新后的隐藏状态
        return F.log_softmax(output, dim=-1), new_h

class StreamingSightBi(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=256, dropout=0.2, num_layers=1):
        super(StreamingSightBi, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout
        )
        
        # 输出层
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(hidden_size, output_dim)
    
    def forward(self, x, hPt, cPt):
        # x: 输入张量，形状为 [batch_size, seq_length, input_size]
        # hPt, cPt: LSTM的隐藏状态和单元状态
        
        # 通过LSTM
        x, (hPt, cPt) = self.lstm(x, (hPt, cPt))
        
        # 展平并通过全连接层
        x = self.flatten(x)
        x = F.relu(x)
        x = self.fc(x)
        
        return F.log_softmax(x, dim=-1), (hPt, cPt)

class StreamingSightMuOriginal(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=1024, dropout=0.2, num_layers=3):
        super(StreamingSightMuOriginal, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 递归全连接层
        self.rnnsfc1 = nn.Linear(input_dim + hidden_size, hidden_size)
        self.dropout1 = nn.Dropout(p=dropout)
        self.rnnsfc2 = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(p=dropout)
        self.rnnsfc3 = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.dropout3 = nn.Dropout(p=dropout)
        
        # 输出层
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(hidden_size, output_dim)
    
    def forward(self, x, h):
        # x: 输入张量，形状为 [batch_size, seq_length, input_size]
        # h: 外部隐藏状态，形状为 [3, batch_size, hidden_size]
        
        batch_size, seq_length, input_size = x.size()
        x = x.squeeze(1)  # 假设seq_length=1
        
        # 第一个全连接层，输入为当前输入和隐藏状态的拼接
        xh1 = torch.cat([x, h[0]], dim=-1)
        nh0 = F.relu(self.rnnsfc1(xh1))
        nh0 = self.dropout1(nh0)
        
        # 第二个全连接层
        xh2 = torch.cat([nh0, h[1]], dim=-1)
        nh1 = F.relu(self.rnnsfc2(xh2))
        nh1 = self.dropout2(nh1)
        
        # 第三个全连接层
        xh3 = torch.cat([nh1, h[2]], dim=-1)
        nh2 = F.relu(self.rnnsfc3(xh3))
        nh2 = self.dropout3(nh2)
        
        # 展平并通过全连接层
        x = self.flatten(nh2)
        x = F.relu(x)
        x = self.fc(x)
        
        # 生成新的隐藏状态
        hnew = [nh0.detach(), nh1.detach(), nh2.detach()]
        
        # 返回输出和更新后的隐藏状态
        return F.log_softmax(x, dim=-1), hnew

class TCRN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=1024, dropout=0.2, num_layers=3, frames_per_segment=3):
        super(TCRN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.frames_per_segment = frames_per_segment  # 每次处理的帧数
        
        # 时间卷积部分 - 处理连续的frames_per_segment帧
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_size, kernel_size=frames_per_segment, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        
        # 递归全连接层 - 与StreamingSightMuOriginal结构一致
        self.rnnsfc1 = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.dropout1 = nn.Dropout(p=dropout)
        self.rnnsfc2 = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(p=dropout)
        self.rnnsfc3 = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.dropout3 = nn.Dropout(p=dropout)
        
        # 输出层
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(hidden_size, output_dim)
    
    def forward(self, x, h):
        """
        x: 输入张量，形状为 [batch_size, frames_per_segment, input_size]
        h: 外部隐藏状态，形状为 [3, batch_size, hidden_size]
        """
        batch_size, seq_length, input_size = x.size()
        
        # 确保输入的帧数正确
        assert seq_length == self.frames_per_segment, f"输入帧数应为{self.frames_per_segment}，但得到{seq_length}"
        
        # 重塑张量以适应Conv1d的输入需求 [batch, channels, seq_len]
        x = x.permute(0, 2, 1)  # [batch_size, input_size, frames_per_segment]
        
        # 应用时间卷积
        # 输出形状: [batch_size, hidden_size, 1]
        x = self.temporal_conv(x)
        
        # 压缩最后一个维度
        x = x.squeeze(-1)  # [batch_size, hidden_size]
        
        # 第一个全连接层，输入为卷积输出和隐藏状态的拼接
        xh1 = torch.cat([x, h[0]], dim=-1)
        nh0 = F.relu(self.rnnsfc1(xh1))
        nh0 = self.dropout1(nh0)
        
        # 第二个全连接层
        xh2 = torch.cat([nh0, h[1]], dim=-1)
        nh1 = F.relu(self.rnnsfc2(xh2))
        nh1 = self.dropout2(nh1)
        
        # 第三个全连接层
        xh3 = torch.cat([nh1, h[2]], dim=-1)
        nh2 = F.relu(self.rnnsfc3(xh3))
        nh2 = self.dropout3(nh2)
        
        # 展平并通过全连接层
        x = self.flatten(nh2)
        x = F.relu(x)
        x = self.fc(x)
        
        # 生成新的隐藏状态
        hnew = [nh0.detach(), nh1.detach(), nh2.detach()]
        
        # 返回输出和更新后的隐藏状态
        return F.log_softmax(x, dim=-1), hnew

class MSTCRN(nn.Module):
    """
    多尺度时间卷积递归网络 (Multi-Scale Temporal Convolutional Recurrent Network)
    
    特点：
    1. 引入分层外置状态：快速层(Fast)和慢速层(Slow)
    2. 快速层每帧更新，捕获短期依赖
    3. 慢速层间隔多帧更新，保存长期信息
    """
    def __init__(self, input_dim, output_dim, hidden_size=1024, dropout=0.2, 
                 num_layers=3, frames_per_segment=3, slow_update_rate=10):
        super(MSTCRN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.frames_per_segment = frames_per_segment  # 每次处理的帧数
        self.slow_update_rate = slow_update_rate     # 慢速层更新频率
        
        # 时间卷积部分 - 处理连续的frames_per_segment帧
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_size, kernel_size=frames_per_segment, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        
        # 快速层递归全连接 - 每一步都更新
        self.fast_rnnsfc1 = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.fast_dropout1 = nn.Dropout(p=dropout)
        self.fast_rnnsfc2 = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.fast_dropout2 = nn.Dropout(p=dropout)
        self.fast_rnnsfc3 = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.fast_dropout3 = nn.Dropout(p=dropout)
        
        # 慢速层更新网络 - 每slow_update_rate步更新一次
        self.slow_update1 = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.slow_dropout1 = nn.Dropout(p=dropout)
        self.slow_update2 = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.slow_dropout2 = nn.Dropout(p=dropout)
        self.slow_update3 = nn.Linear(hidden_size + hidden_size, hidden_size)
        self.slow_dropout3 = nn.Dropout(p=dropout)
        
        # 输出层 - 结合快速和慢速状态
        self.output_fusion = nn.Linear(hidden_size * 2, hidden_size)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(hidden_size, output_dim)
    
    def forward(self, x, state_dict):
        """
        x: 输入张量，形状为 [batch_size, frames_per_segment, input_size]
        state_dict: 包含外部状态的字典，结构为：
            {
                'fast': [h1, h2, h3],  # 快速层状态
                'slow': [h1, h2, h3],  # 慢速层状态
                'step_count': int       # 当前步数，用于确定是否更新慢速层
            }
        """
        batch_size, seq_length, input_size = x.size()
        
        # 确保输入的帧数正确
        assert seq_length == self.frames_per_segment, f"输入帧数应为{self.frames_per_segment}，但得到{seq_length}"
        
        # 重塑张量以适应Conv1d的输入需求 [batch, channels, seq_len]
        x = x.permute(0, 2, 1)  # [batch_size, input_size, frames_per_segment]
        
        # 应用时间卷积获取z_t
        z_t = self.temporal_conv(x)
        z_t = z_t.squeeze(-1)  # [batch_size, hidden_size]
        
        # 获取当前状态
        fast_h = state_dict['fast']  # 快速层隐藏状态
        slow_h = state_dict['slow']  # 慢速层隐藏状态
        step_count = state_dict['step_count']  # 当前步数
        
        # 更新快速层 - 每步都更新
        # 第一层
        fast_xh1 = torch.cat([z_t, fast_h[0]], dim=-1)
        fast_nh0 = F.relu(self.fast_rnnsfc1(fast_xh1))
        fast_nh0 = self.fast_dropout1(fast_nh0)
        
        # 第二层
        fast_xh2 = torch.cat([fast_nh0, fast_h[1]], dim=-1)
        fast_nh1 = F.relu(self.fast_rnnsfc2(fast_xh2))
        fast_nh1 = self.fast_dropout2(fast_nh1)
        
        # 第三层
        fast_xh3 = torch.cat([fast_nh1, fast_h[2]], dim=-1)
        fast_nh2 = F.relu(self.fast_rnnsfc3(fast_xh3))
        fast_nh2 = self.fast_dropout3(fast_nh2)
        
        # 更新慢速层 - 仅在step_count是slow_update_rate的倍数时更新
        if step_count % self.slow_update_rate == 0:
            # 第一层 - 慢速层利用快速层状态更新
            slow_xh1 = torch.cat([fast_nh0, slow_h[0]], dim=-1)
            slow_nh0 = F.relu(self.slow_update1(slow_xh1))
            slow_nh0 = self.slow_dropout1(slow_nh0)
            
            # 第二层
            slow_xh2 = torch.cat([fast_nh1, slow_h[1]], dim=-1)
            slow_nh1 = F.relu(self.slow_update2(slow_xh2))
            slow_nh1 = self.slow_dropout2(slow_nh1)
            
            # 第三层
            slow_xh3 = torch.cat([fast_nh2, slow_h[2]], dim=-1)
            slow_nh2 = F.relu(self.slow_update3(slow_xh3))
            slow_nh2 = self.slow_dropout3(slow_nh2)
        else:
            # 不更新，保持原状态
            slow_nh0 = slow_h[0]
            slow_nh1 = slow_h[1]
            slow_nh2 = slow_h[2]
        
        # 结合快速层和慢速层的最终状态做预测
        combined_state = torch.cat([fast_nh2, slow_nh2], dim=-1)
        fused_state = F.relu(self.output_fusion(combined_state))
        
        # 通过输出层
        x = self.flatten(fused_state)
        x = F.relu(x)
        x = self.fc(x)
        
        # 生成新的状态字典
        new_state_dict = {
            'fast': [fast_nh0.detach(), fast_nh1.detach(), fast_nh2.detach()],
            'slow': [slow_nh0.detach(), slow_nh1.detach(), slow_nh2.detach()],
            'step_count': step_count + 1
        }
        
        # 返回输出和更新后的状态字典
        return F.log_softmax(x, dim=-1), new_state_dict

def generate_position_weights(seq_length):
    """
    生成基于位置的权重向量，中间位置权重高，边缘位置权重低
    """
    # 使用高斯分布生成权重
    mid_point = seq_length / 2
    # 标准差控制权重分布的"宽度"
    sigma = seq_length / 4
    
    weights = np.zeros(seq_length)
    for i in range(seq_length):
        weights[i] = np.exp(-0.5 * ((i - mid_point) / sigma) ** 2)
    
    # 归一化权重使其和为1
    weights = weights / np.sum(weights)
    
    return weights

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        try:
            # 验证标签范围
            max_label = len(SELECTED_GESTURE_NAME) - 1  # 减1因为我们使用0-27而不是1-28
            if torch.max(target).item() > max_label or torch.min(target).item() < 0:
                _logger.error(f"批次 {batch_idx} 中发现无效标签: 最小={torch.min(target).item()}, 最大={torch.max(target).item()}, 有效范围=[0,{max_label}]")
                continue
                
            data, target = data.to(device), target.to(device)
            batch_size, max_length, input_dim = data.size()
            
            # 生成基于位置的权重
            position_weights = generate_position_weights(max_length)
            position_weights = torch.tensor(position_weights, dtype=torch.float32).to(device)
            
            # 初始化外部记忆
            if isinstance(model, StreamingSightMu):
                # 初始化隐藏状态
                hPt = [torch.zeros(batch_size, model.hidden_size).to(device) for _ in range(model.num_layers)]
                loss = 0
                frame_outputs = []
                
                # 对每一帧进行处理，根据位置加权
                for i in range(max_length):
                    frames = data[:, i, :].unsqueeze(1)
                    output, hPt = model(frames, hPt)
                    
                    # 加权损失
                    frame_loss = criterion(output, target) * position_weights[i]
                    loss += frame_loss
                    
                    # 存储每帧的输出
                    frame_outputs.append(output)
                
                # 加权组合所有帧的输出
                outputs = torch.zeros_like(frame_outputs[0])
                for i, output in enumerate(frame_outputs):
                    outputs += output * position_weights[i]
                
            elif isinstance(model, StreamingSightBi):
                hPt = torch.zeros(model.num_layers, batch_size, model.hidden_size).to(device)
                cPt = torch.zeros(model.num_layers, batch_size, model.hidden_size).to(device)
                
                # 一次性处理整个序列
                outputs, (hPt, cPt) = model(data, hPt, cPt)
                loss = criterion(outputs, target)
                
            elif isinstance(model, StreamingSightMuOriginal):
                # 初始化隐藏状态
                hPt = [torch.zeros(batch_size, model.hidden_size).to(device) for _ in range(model.num_layers)]
                loss = 0
                frame_outputs = []
                
                # 对每一帧进行处理，根据位置加权
                for i in range(max_length):
                    frames = data[:, i, :].unsqueeze(1)
                    output, hPt = model(frames, hPt)
                    
                    # 加权损失
                    frame_loss = criterion(output, target) * position_weights[i]
                    loss += frame_loss
                    
                    # 存储每帧的输出
                    frame_outputs.append(output)
                
                # 加权组合所有帧的输出
                outputs = torch.zeros_like(frame_outputs[0])
                for i, output in enumerate(frame_outputs):
                    outputs += output * position_weights[i]
                    
            elif isinstance(model, TCRN):
                # 初始化隐藏状态
                hPt = [torch.zeros(batch_size, model.hidden_size).to(device) for _ in range(model.num_layers)]
                loss = 0
                frame_outputs = []
                
                # 计算可以完整分割为3帧段的数量
                frames_per_segment = model.frames_per_segment
                complete_segments = max_length // frames_per_segment
                
                # 对每3帧进行处理
                for i in range(complete_segments):
                    start_idx = i * frames_per_segment
                    end_idx = start_idx + frames_per_segment
                    segment = data[:, start_idx:end_idx, :]
                    
                    # 处理当前段
                    output, hPt = model(segment, hPt)
                    
                    # 使用段的中间位置的权重
                    middle_idx = start_idx + frames_per_segment // 2
                    segment_weight = position_weights[middle_idx]
                    
                    # 加权损失
                    segment_loss = criterion(output, target) * segment_weight
                    loss += segment_loss
                    
                    # 存储输出，对该段中的每一帧都使用相同的输出
                    for j in range(start_idx, end_idx):
                        # 调整权重以适应每帧存储，但总和保持不变
                        frame_weight = position_weights[j] / sum(position_weights[start_idx:end_idx])
                        frame_outputs.append((output, frame_weight))
                
                # 处理剩余的帧 (不足一个完整段的部分)
                remaining_frames = max_length % frames_per_segment
                if remaining_frames > 0:
                    start_idx = complete_segments * frames_per_segment
                    
                    # 使用填充创建完整段
                    padded_segment = torch.zeros(batch_size, frames_per_segment, input_dim).to(device)
                    padded_segment[:, :remaining_frames, :] = data[:, start_idx:, :]
                    
                    # 处理填充后的段
                    output, hPt = model(padded_segment, hPt)
                    
                    # 只对真实帧使用权重
                    for j in range(remaining_frames):
                        idx = start_idx + j
                        frame_loss = criterion(output, target) * position_weights[idx] / remaining_frames
                        loss += frame_loss
                        
                        # 存储输出
                        frame_outputs.append((output, position_weights[idx]))
                
                # 加权组合所有帧的输出
                outputs = torch.zeros_like(frame_outputs[0][0])
                total_weight = 0
                for output, weight in frame_outputs:
                    outputs += output * weight
                    total_weight += weight
                
                # 归一化输出
                if total_weight > 0:
                    outputs = outputs / total_weight
            
            elif isinstance(model, MSTCRN):
                # 初始化多尺度状态
                fast_h = [torch.zeros(batch_size, model.hidden_size).to(device) for _ in range(model.num_layers)]
                slow_h = [torch.zeros(batch_size, model.hidden_size).to(device) for _ in range(model.num_layers)]
                state_dict = {'fast': fast_h, 'slow': slow_h, 'step_count': 0}
                
                loss = 0
                frame_outputs = []
                
                # 计算可以完整分割为frames_per_segment帧段的数量
                frames_per_segment = model.frames_per_segment
                complete_segments = max_length // frames_per_segment
                
                # 对每段进行处理
                for i in range(complete_segments):
                    start_idx = i * frames_per_segment
                    end_idx = start_idx + frames_per_segment
                    segment = data[:, start_idx:end_idx, :]
                    
                    # 处理当前段
                    output, state_dict = model(segment, state_dict)
                    
                    # 使用段的中间位置的权重
                    middle_idx = start_idx + frames_per_segment // 2
                    segment_weight = position_weights[middle_idx]
                    
                    # 加权损失
                    segment_loss = criterion(output, target) * segment_weight
                    loss += segment_loss
                    
                    # 存储输出，对该段中的每一帧都使用相同的输出
                    for j in range(start_idx, end_idx):
                        # 调整权重以适应每帧存储，但总和保持不变
                        frame_weight = position_weights[j] / sum(position_weights[start_idx:end_idx])
                        frame_outputs.append((output, frame_weight))
                
                # 处理剩余的帧 (不足一个完整段的部分)
                remaining_frames = max_length % frames_per_segment
                if remaining_frames > 0:
                    start_idx = complete_segments * frames_per_segment
                    
                    # 使用填充创建完整段
                    padded_segment = torch.zeros(batch_size, frames_per_segment, input_dim).to(device)
                    padded_segment[:, :remaining_frames, :] = data[:, start_idx:, :]
                    
                    # 处理填充后的段
                    output, state_dict = model(padded_segment, state_dict)
                    
                    # 只对真实帧使用权重
                    for j in range(remaining_frames):
                        idx = start_idx + j
                        frame_loss = criterion(output, target) * position_weights[idx] / remaining_frames
                        loss += frame_loss
                        
                        # 存储输出
                        frame_outputs.append((output, position_weights[idx]))
                
                # 加权组合所有帧的输出
                outputs = torch.zeros_like(frame_outputs[0][0])
                total_weight = 0
                for output, weight in frame_outputs:
                    outputs += output * weight
                    total_weight += weight
                
                # 归一化输出
                if total_weight > 0:
                    outputs = outputs / total_weight
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # 计算准确率
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            #if batch_idx % 10 == 0:
                #_logger.info(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                              #f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        except Exception as e:
            _logger.error(f"训练批次 {batch_idx} 出错: {str(e)}")
            import traceback
            _logger.error(traceback.format_exc())
            continue
    
    train_loss /= max(1, len(train_loader))  # 避免除以零
    accuracy = 100. * correct / max(1, total)  # 避免除以零
    
    _logger.info(f'Train set: Average loss: {train_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)')
    
    return train_loss, accuracy

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader, desc="Testing")):
            try:
                # 验证标签范围
                max_label = len(SELECTED_GESTURE_NAME) - 1  # 减1因为我们使用0-27而不是1-28
                if torch.max(target).item() > max_label or torch.min(target).item() < 0:
                    _logger.error(f"测试批次 {batch_idx} 中发现无效标签: 最小={torch.min(target).item()}, 最大={torch.max(target).item()}, 有效范围=[0,{max_label}]")
                    continue
                    
                data, target = data.to(device), target.to(device)
                batch_size, max_length, input_dim = data.size()
                
                # 生成基于位置的权重
                position_weights = generate_position_weights(max_length)
                position_weights = torch.tensor(position_weights, dtype=torch.float32).to(device)
                
                # 初始化外部记忆
                if isinstance(model, StreamingSightMu):
                    # 初始化隐藏状态
                    hPt = [torch.zeros(batch_size, model.hidden_size).to(device) for _ in range(model.num_layers)]
                    test_batch_loss = 0
                    frame_outputs = []
                    
                    # 对每一帧进行处理，根据位置加权
                    for i in range(max_length):
                        frames = data[:, i, :].unsqueeze(1)
                        output, hPt = model(frames, hPt)
                        
                        # 加权损失
                        frame_loss = criterion(output, target) * position_weights[i]
                        test_batch_loss += frame_loss
                        
                        # 存储每帧的输出
                        frame_outputs.append(output)
                    
                    # 加权组合所有帧的输出
                    outputs = torch.zeros_like(frame_outputs[0])
                    for i, output in enumerate(frame_outputs):
                        outputs += output * position_weights[i]
                    
                    test_loss += test_batch_loss.item()
                    
                elif isinstance(model, StreamingSightBi):
                    hPt = torch.zeros(model.num_layers, batch_size, model.hidden_size).to(device)
                    cPt = torch.zeros(model.num_layers, batch_size, model.hidden_size).to(device)
                    
                    # 一次性处理整个序列
                    outputs, _ = model(data, hPt, cPt)
                    test_loss += criterion(outputs, target).item()
                    
                elif isinstance(model, StreamingSightMuOriginal):
                    # 初始化隐藏状态
                    hPt = [torch.zeros(batch_size, model.hidden_size).to(device) for _ in range(model.num_layers)]
                    test_batch_loss = 0
                    frame_outputs = []
                    
                    # 对每一帧进行处理，根据位置加权
                    for i in range(max_length):
                        frames = data[:, i, :].unsqueeze(1)
                        output, hPt = model(frames, hPt)
                        
                        # 加权损失
                        frame_loss = criterion(output, target) * position_weights[i]
                        test_batch_loss += frame_loss
                        
                        # 存储每帧的输出
                        frame_outputs.append(output)
                    
                    # 加权组合所有帧的输出
                    outputs = torch.zeros_like(frame_outputs[0])
                    for i, output in enumerate(frame_outputs):
                        outputs += output * position_weights[i]
                    
                    test_loss += test_batch_loss.item()
                    
                elif isinstance(model, TCRN):
                    # 初始化隐藏状态
                    hPt = [torch.zeros(batch_size, model.hidden_size).to(device) for _ in range(model.num_layers)]
                    test_batch_loss = 0
                    frame_outputs = []
                    
                    # 计算可以完整分割为3帧段的数量
                    frames_per_segment = model.frames_per_segment
                    complete_segments = max_length // frames_per_segment
                    
                    # 对每3帧进行处理
                    for i in range(complete_segments):
                        start_idx = i * frames_per_segment
                        end_idx = start_idx + frames_per_segment
                        segment = data[:, start_idx:end_idx, :]
                        
                        # 处理当前段
                        output, hPt = model(segment, hPt)
                        
                        # 使用段的中间位置的权重
                        middle_idx = start_idx + frames_per_segment // 2
                        segment_weight = position_weights[middle_idx]
                        
                        # 加权损失
                        segment_loss = criterion(output, target) * segment_weight
                        test_batch_loss += segment_loss
                        
                        # 存储输出，对该段中的每一帧都使用相同的输出
                        for j in range(start_idx, end_idx):
                            # 调整权重以适应每帧存储，但总和保持不变
                            frame_weight = position_weights[j] / sum(position_weights[start_idx:end_idx])
                            frame_outputs.append((output, frame_weight))
                    
                    # 处理剩余的帧 (不足一个完整段的部分)
                    remaining_frames = max_length % frames_per_segment
                    if remaining_frames > 0:
                        start_idx = complete_segments * frames_per_segment
                        
                        # 使用填充创建完整段
                        padded_segment = torch.zeros(batch_size, frames_per_segment, input_dim).to(device)
                        padded_segment[:, :remaining_frames, :] = data[:, start_idx:, :]
                        
                        # 处理填充后的段
                        output, hPt = model(padded_segment, hPt)
                        
                        # 只对真实帧使用权重
                        for j in range(remaining_frames):
                            idx = start_idx + j
                            frame_loss = criterion(output, target) * position_weights[idx] / remaining_frames
                            test_batch_loss += frame_loss
                            
                            # 存储输出
                            frame_outputs.append((output, position_weights[idx]))
                    
                    # 加权组合所有帧的输出
                    outputs = torch.zeros_like(frame_outputs[0][0])
                    total_weight = 0
                    for output, weight in frame_outputs:
                        outputs += output * weight
                        total_weight += weight
                    
                    # 归一化输出
                    if total_weight > 0:
                        outputs = outputs / total_weight
                        
                    test_loss += test_batch_loss.item()
                
                elif isinstance(model, MSTCRN):
                    # 初始化多尺度状态
                    fast_h = [torch.zeros(batch_size, model.hidden_size).to(device) for _ in range(model.num_layers)]
                    slow_h = [torch.zeros(batch_size, model.hidden_size).to(device) for _ in range(model.num_layers)]
                    state_dict = {'fast': fast_h, 'slow': slow_h, 'step_count': 0}
                    
                    test_batch_loss = 0
                    frame_outputs = []
                    
                    # 计算可以完整分割为frames_per_segment帧段的数量
                    frames_per_segment = model.frames_per_segment
                    complete_segments = max_length // frames_per_segment
                    
                    # 对每段进行处理
                    for i in range(complete_segments):
                        start_idx = i * frames_per_segment
                        end_idx = start_idx + frames_per_segment
                        segment = data[:, start_idx:end_idx, :]
                        
                        # 处理当前段
                        output, state_dict = model(segment, state_dict)
                        
                        # 使用段的中间位置的权重
                        middle_idx = start_idx + frames_per_segment // 2
                        segment_weight = position_weights[middle_idx]
                        
                        # 加权损失
                        segment_loss = criterion(output, target) * segment_weight
                        test_batch_loss += segment_loss
                        
                        # 存储输出
                        for j in range(start_idx, end_idx):
                            frame_weight = position_weights[j] / sum(position_weights[start_idx:end_idx])
                            frame_outputs.append((output, frame_weight))
                    
                    # 处理剩余的帧 (不足一个完整段的部分)
                    remaining_frames = max_length % frames_per_segment
                    if remaining_frames > 0:
                        start_idx = complete_segments * frames_per_segment
                        
                        # 使用填充创建完整段
                        padded_segment = torch.zeros(batch_size, frames_per_segment, input_dim).to(device)
                        padded_segment[:, :remaining_frames, :] = data[:, start_idx:, :]
                        
                        # 处理填充后的段
                        output, state_dict = model(padded_segment, state_dict)
                        
                        # 只对真实帧使用权重
                        for j in range(remaining_frames):
                            idx = start_idx + j
                            frame_loss = criterion(output, target) * position_weights[idx] / remaining_frames
                            test_batch_loss += frame_loss
                            
                            # 存储输出
                            frame_outputs.append((output, position_weights[idx]))
                    
                    # 加权组合所有帧的输出
                    outputs = torch.zeros_like(frame_outputs[0][0])
                    total_weight = 0
                    for output, weight in frame_outputs:
                        outputs += output * weight
                        total_weight += weight
                    
                    # 归一化输出
                    if total_weight > 0:
                        outputs = outputs / total_weight
                    
                    test_loss += test_batch_loss.item()
                
                # 计算准确率
                pred = outputs.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
            except Exception as e:
                _logger.error(f"测试批次 {batch_idx} 出错: {str(e)}")
                import traceback
                _logger.error(traceback.format_exc())
                continue
    
    test_loss /= max(1, len(test_loader))  # 避免除以零
    accuracy = 100. * correct / max(1, total)  # 避免除以零
    
    _logger.info(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)')
    
    return test_loss, accuracy

def save_model(model, optimizer, epoch, accuracy, model_type, path="./models"):
    if not os.path.exists(path):
        os.makedirs(path)
        
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy
    }
    
    filename = os.path.join(path, f"{model_type}_checkpoint_epoch{epoch}_acc{accuracy:.2f}.pt")
    torch.save(checkpoint, filename)
    _logger.info(f"Model saved to {filename}")

def compute_confusion_matrix(model, device, test_loader):
    """
    计算测试集上的混淆矩阵
    """
    model.eval()
    confmat = MulticlassConfusionMatrix(num_classes=len(SELECTED_GESTURE_NAME))
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            try:
                data, target = data.to(device), target.to(device)
                batch_size, max_length, input_dim = data.size()
                
                # 生成基于位置的权重
                position_weights = generate_position_weights(max_length)
                position_weights = torch.tensor(position_weights, dtype=torch.float32).to(device)
                
                # 初始化外部记忆
                if isinstance(model, StreamingSightMu):
                    # 初始化隐藏状态
                    hPt = [torch.zeros(batch_size, model.hidden_size).to(device) for _ in range(model.num_layers)]
                    frame_outputs = []
                    
                    # 对每一帧进行处理，根据位置加权
                    for i in range(max_length):
                        frames = data[:, i, :].unsqueeze(1)
                        output, hPt = model(frames, hPt)
                        frame_outputs.append(output)
                    
                    # 加权组合所有帧的输出
                    outputs = torch.zeros_like(frame_outputs[0])
                    for i, output in enumerate(frame_outputs):
                        outputs += output * position_weights[i]
                    
                elif isinstance(model, StreamingSightBi):
                    hPt = torch.zeros(model.num_layers, batch_size, model.hidden_size).to(device)
                    cPt = torch.zeros(model.num_layers, batch_size, model.hidden_size).to(device)
                    outputs, _ = model(data, hPt, cPt)
                    
                elif isinstance(model, StreamingSightMuOriginal):
                    # 初始化隐藏状态
                    hPt = [torch.zeros(batch_size, model.hidden_size).to(device) for _ in range(model.num_layers)]
                    frame_outputs = []
                    
                    # 对每一帧进行处理，根据位置加权
                    for i in range(max_length):
                        frames = data[:, i, :].unsqueeze(1)
                        output, hPt = model(frames, hPt)
                        frame_outputs.append(output)
                    
                    # 加权组合所有帧的输出
                    outputs = torch.zeros_like(frame_outputs[0])
                    for i, output in enumerate(frame_outputs):
                        outputs += output * position_weights[i]
                
                elif isinstance(model, TCRN):
                    # 初始化隐藏状态
                    hPt = [torch.zeros(batch_size, model.hidden_size).to(device) for _ in range(model.num_layers)]
                    frame_outputs = []
                    
                    # 计算可以完整分割为3帧段的数量
                    frames_per_segment = model.frames_per_segment
                    complete_segments = max_length // frames_per_segment
                    
                    # 对每3帧进行处理
                    for i in range(complete_segments):
                        start_idx = i * frames_per_segment
                        end_idx = start_idx + frames_per_segment
                        segment = data[:, start_idx:end_idx, :]
                        
                        # 处理当前段
                        output, hPt = model(segment, hPt)
                        
                        # 存储输出，对该段中的每一帧都使用相同的输出
                        for j in range(start_idx, end_idx):
                            frame_weight = position_weights[j] / sum(position_weights[start_idx:end_idx])
                            frame_outputs.append((output, frame_weight))
                    
                    # 处理剩余的帧 (不足一个完整段的部分)
                    remaining_frames = max_length % frames_per_segment
                    if remaining_frames > 0:
                        start_idx = complete_segments * frames_per_segment
                        
                        # 使用填充创建完整段
                        padded_segment = torch.zeros(batch_size, frames_per_segment, input_dim).to(device)
                        padded_segment[:, :remaining_frames, :] = data[:, start_idx:, :]
                        
                        # 处理填充后的段
                        output, hPt = model(padded_segment, hPt)
                        
                        # 只对真实帧使用权重
                        for j in range(remaining_frames):
                            idx = start_idx + j
                            frame_outputs.append((output, position_weights[idx]))
                    
                    # 加权组合所有帧的输出
                    outputs = torch.zeros_like(frame_outputs[0][0])
                    total_weight = 0
                    for output, weight in frame_outputs:
                        outputs += output * weight
                        total_weight += weight
                    
                    # 归一化输出
                    if total_weight > 0:
                        outputs = outputs / total_weight
                
                elif isinstance(model, MSTCRN):
                    # 初始化多尺度状态
                    fast_h = [torch.zeros(batch_size, model.hidden_size).to(device) for _ in range(model.num_layers)]
                    slow_h = [torch.zeros(batch_size, model.hidden_size).to(device) for _ in range(model.num_layers)]
                    state_dict = {'fast': fast_h, 'slow': slow_h, 'step_count': 0}
                    
                    frame_outputs = []
                    
                    # 计算可以完整分割为frames_per_segment帧段的数量
                    frames_per_segment = model.frames_per_segment
                    complete_segments = max_length // frames_per_segment
                    
                    # 对每段进行处理
                    for i in range(complete_segments):
                        start_idx = i * frames_per_segment
                        end_idx = start_idx + frames_per_segment
                        segment = data[:, start_idx:end_idx, :]
                        
                        # 处理当前段
                        output, state_dict = model(segment, state_dict)
                        
                        # 存储输出
                        for j in range(start_idx, end_idx):
                            frame_weight = position_weights[j] / sum(position_weights[start_idx:end_idx])
                            frame_outputs.append((output, frame_weight))
                    
                    # 处理剩余的帧 (不足一个完整段的部分)
                    remaining_frames = max_length % frames_per_segment
                    if remaining_frames > 0:
                        start_idx = complete_segments * frames_per_segment
                        
                        # 使用填充创建完整段
                        padded_segment = torch.zeros(batch_size, frames_per_segment, input_dim).to(device)
                        padded_segment[:, :remaining_frames, :] = data[:, start_idx:, :]
                        
                        # 处理填充后的段
                        output, state_dict = model(padded_segment, state_dict)
                        
                        # 只对真实帧使用权重
                        for j in range(remaining_frames):
                            idx = start_idx + j
                            frame_outputs.append((output, position_weights[idx]))
                    
                    # 加权组合所有帧的输出
                    outputs = torch.zeros_like(frame_outputs[0][0])
                    total_weight = 0
                    for output, weight in frame_outputs:
                        outputs += output * weight
                        total_weight += weight
                    
                    # 归一化输出
                    if total_weight > 0:
                        outputs = outputs / total_weight
                
                pred = outputs.argmax(dim=1)
                confmat.update(pred, target)
                
            except Exception as e:
                _logger.error(f"计算混淆矩阵时出错: {str(e)}")
                continue
    
    return confmat.compute().cpu().numpy()

def plot_confusion_matrix(cm, class_names):
    """
    将混淆矩阵绘制为热力图
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 计算准确率
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm_norm, annot=False, fmt='.2f', xticklabels=class_names, yticklabels=class_names, cmap='Blues', ax=ax)
    
    # 设置标题和标签
    ax.set_xlabel('预测类别')
    ax.set_ylabel('真实类别')
    ax.set_title('归一化混淆矩阵')
    
    # 旋转x轴标签以防止重叠
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig

def main():
    # 参数设置
    parser = argparse.ArgumentParser(description='SHREC2017 Hand Gesture Recognition')
    parser.add_argument('--model-type', type=str, default='tcrn',
                        choices=['mu', 'bi', 'mu_original', 'tcrn', 'mstcrn'],
                        help='模型类型: mu (StreamingSightMu)、bi (StreamingSightBi)、mu_original (StreamingSightMuOriginal)、tcrn (TCRN) 或 mstcrn (MS-TCRN) (default: tcrn)')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='训练批次大小 (default: 8)')
    parser.add_argument('--test-batch-size', type=int, default=BATCH_SIZE,
                        help='测试批次大小 (default: 8)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数 (default: 100)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='学习率 (default: 0.0001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='禁用CUDA训练')
    parser.add_argument('--seed', type=int, default=1,
                        help='随机种子 (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='训练时每隔多少批次记录一次日志 (default: 10)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='是否保存模型')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='限制加载的样本数量（用于测试）')
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'single_finger'],
                        help='训练模式: full (所有手势) 或 single_finger (仅单指手势) (default: full)')
    parser.add_argument('--scheduler', type=str, default='reduce_on_plateau',
                        choices=['step', 'reduce_on_plateau', 'cosine', 'none'],
                        help='学习率调度器类型 (default: reduce_on_plateau)')
    parser.add_argument('--step-size', type=int, default=30,
                        help='StepLR中的步长 (default: 30)')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='学习率衰减因子 (default: 0.1)')
    parser.add_argument('--patience', type=int, default=10,
                        help='ReduceLROnPlateau的耐心值 (default: 10)')
    parser.add_argument('--min-lr', type=float, default=1e-5,
                        help='最小学习率 (default: 1e-5)')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='预热阶段的轮数 (default: 5)')
    parser.add_argument('--tensorboard', action='store_true', default=False,
                        help='是否使用TensorBoard记录训练过程')
    parser.add_argument('--resume', type=str, default=None,
                        help='从checkpoint继续训练的文件路径')
    parser.add_argument('--start-epoch', type=int, default=1,
                        help='如果不从checkpoint继续训练，指定起始轮数 (default: 1)')
    parser.add_argument('--additional-epochs', type=int, default=100,
                        help='从checkpoint继续训练时，额外训练的轮数 (default: 100)')
    parser.add_argument('--frames-per-segment', type=int, default=3,
                        help='TCRN模型每次处理的帧数 (default: 3)')
    parser.add_argument('--hidden-size', type=int, default=768,
                        help='模型隐藏层大小 (default: 768)')
    parser.add_argument('--slow-update-rate', type=int, default=3,
                        help='MS-TCRN模型慢速层更新频率 (default: 3)')
    parser.add_argument('--weight-decay', type=float, default=0.00001,
                        help='权重衰减系数用于L2正则化 (default: 0.00001)')
    args = parser.parse_args()
    
    # 根据选择的模式设置类别
    global SELECTED_GESTURE_NAME, OUTPUT_DIM
    if args.mode == 'single_finger':
        SELECTED_GESTURE_NAME = GESTURE_NAME
        _logger.info("使用单指手势模式 (14个类别)")
    else:
        SELECTED_GESTURE_NAME = FULL_GESTURE_NAME
        _logger.info("使用完整手势模式 (28个类别)")
    
    OUTPUT_DIM = len(SELECTED_GESTURE_NAME)
    _logger.info(f"输出维度: {OUTPUT_DIM}")
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # 设置TensorBoard
    if args.tensorboard:
        log_dir = os.path.join("runs", f"{args.model_type}_{args.mode}_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        writer = SummaryWriter(log_dir=log_dir)
        _logger.info(f"TensorBoard日志保存在: {log_dir}")
    else:
        writer = None
    
    # 加载数据
    _logger.info("加载训练数据...")
    train_gesture_data, train_gesture_labels = load_gesture_dataset("train_gestures.txt", args.max_samples, args.mode)
    
    _logger.info("加载测试数据...")
    test_gesture_data, test_gesture_labels = load_gesture_dataset("test_gestures.txt", args.max_samples, args.mode)
    
    # 预处理数据
    _logger.info("预处理训练数据...")
    # 对每个骨架序列应用归一化
    normalized_train_data = [normalize_skeleton(sequence) for sequence in train_gesture_data]
    # 填充/截断序列
    train_data = pad_data(normalized_train_data)
    train_label = np.array(train_gesture_labels)
    
    _logger.info("预处理测试数据...")
    normalized_test_data = [normalize_skeleton(sequence) for sequence in test_gesture_data]
    test_data = pad_data(normalized_test_data)
    test_label = np.array(test_gesture_labels)
    
    _logger.info(f"训练数据形状: {train_data.shape}")
    _logger.info(f"测试数据形状: {test_data.shape}")
    
    # 创建数据集和数据加载器
    train_dataset = SHREC2017Dataset(train_data, train_label, augment=True)
    test_dataset = SHREC2017Dataset(test_data, test_label, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
    
    # 创建模型
    input_dim = INPUT_DIM
    output_dim = OUTPUT_DIM
    hidden_size = args.hidden_size
    
    if args.model_type == 'mu':
        model = StreamingSightMu(input_dim, output_dim, hidden_size=hidden_size).to(device)
        _logger.info("使用 StreamingSightMu 模型")
    elif args.model_type == 'bi':
        model = StreamingSightBi(input_dim, output_dim, hidden_size=hidden_size).to(device)
        _logger.info("使用 StreamingSightBi 模型")
    elif args.model_type == 'tcrn':
        model = TCRN(input_dim, output_dim, hidden_size=hidden_size, frames_per_segment=args.frames_per_segment).to(device)
        _logger.info(f"使用 TCRN 模型，每次处理 {args.frames_per_segment} 帧")
    elif args.model_type == 'mstcrn':
        model = MSTCRN(input_dim, output_dim, hidden_size=hidden_size, 
                      frames_per_segment=args.frames_per_segment,
                      slow_update_rate=args.slow_update_rate).to(device)
        _logger.info(f"使用 MS-TCRN 模型，每次处理 {args.frames_per_segment} 帧，慢速层每 {args.slow_update_rate} 步更新一次")
    else:
        model = StreamingSightMuOriginal(input_dim, output_dim, hidden_size=hidden_size).to(device)
        _logger.info("使用 StreamingSightMuOriginal 模型")
    
    # 输出模型结构信息
    _logger.info(f"模型输入维度: {input_dim}")
    _logger.info(f"模型输出维度: {output_dim}")
    _logger.info(f"模型隐藏层大小: {hidden_size}")
    
    # 优化器和损失函数
    if args.model_type in ['tcrn', 'mstcrn']:
        # 为TCRN和MS-TCRN模型添加权重衰减
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        _logger.info(f"使用Adam优化器，学习率={args.lr}，权重衰减={args.weight_decay}")
    else:
        # 其他模型使用原有的优化器配置
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        _logger.info(f"使用Adam优化器，学习率={args.lr}，无权重衰减")
    
    criterion = nn.CrossEntropyLoss()
    
    # 学习率调度器
    if args.scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        _logger.info(f"使用StepLR调度器，步长={args.step_size}，gamma={args.gamma}")
    elif args.scheduler == 'reduce_on_plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.gamma, 
                                    patience=args.patience, min_lr=args.min_lr)
        _logger.info(f"使用ReduceLROnPlateau调度器，耐心值={args.patience}，gamma={args.gamma}")
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
        _logger.info(f"使用CosineAnnealingLR调度器，T_max={args.epochs}，最小学习率={args.min_lr}")
    else:
        scheduler = None
        _logger.info("不使用学习率调度器")
    
    # 初始化训练参数
    start_epoch = args.start_epoch
    best_accuracy = 0.0
    
    # 从checkpoint继续训练
    if args.resume:
        if os.path.isfile(args.resume):
            _logger.info(f"从checkpoint {args.resume} 加载模型")
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 只有当存在相应的键时才加载
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
            if 'accuracy' in checkpoint:
                best_accuracy = checkpoint['accuracy']
                
            _logger.info(f"成功加载checkpoint：轮数={start_epoch-1}，最佳准确率={best_accuracy:.2f}%")
            
            # 设置总训练轮数
            total_epochs = start_epoch + args.additional_epochs - 1
            _logger.info(f"将从第{start_epoch}轮开始额外训练{args.additional_epochs}轮，总共{total_epochs}轮")
        else:
            _logger.warning(f"找不到checkpoint: {args.resume}")
            total_epochs = args.epochs
    else:
        total_epochs = args.epochs
    
    # 学习率预热
    warmup_epochs = args.warmup_epochs
    if warmup_epochs > 0 and start_epoch == 1:  # 只有在从头开始训练时才应用预热
        _logger.info(f"使用学习率预热，预热轮数={warmup_epochs}")
    
    # 训练模型
    _logger.info("开始训练...")
    
    for epoch in range(start_epoch, total_epochs + 1):
        # 学习率预热
        if epoch <= warmup_epochs and scheduler is not None and start_epoch == 1:
            # 线性预热学习率
            warmup_factor = min(1.0, epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * warmup_factor
            _logger.info(f"预热阶段 {epoch}/{warmup_epochs}，学习率={optimizer.param_groups[0]['lr']:.6f}")
        
        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion, epoch)
        test_loss, test_acc = test(model, device, test_loader, criterion)
        
        # 记录到TensorBoard
        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/test', test_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/test', test_acc, epoch)
            writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
            
            # 每10个epoch记录一次混淆矩阵
            if epoch % 10 == 0:
                try:
                    # 计算混淆矩阵
                    confusion_matrix = compute_confusion_matrix(model, device, test_loader)
                    # 添加到TensorBoard
                    fig = plot_confusion_matrix(confusion_matrix, class_names=SELECTED_GESTURE_NAME)
                    writer.add_figure(f'Confusion Matrix/epoch_{epoch}', fig, epoch)
                except Exception as e:
                    _logger.error(f"记录混淆矩阵时出错: {str(e)}")
        
        # 更新学习率
        if scheduler is not None and (epoch > warmup_epochs or start_epoch > 1):
            if args.scheduler == 'reduce_on_plateau':
                scheduler.step(test_loss)
            else:
                scheduler.step()
            
            # 记录当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            _logger.info(f"当前学习率: {current_lr:.6f}")
        
        # 保存最佳模型
        if test_acc > best_accuracy and args.save_model:
            best_accuracy = test_acc
            save_model(model, optimizer, epoch, test_acc, f"{args.model_type}_{args.mode}")
    
    # 关闭TensorBoard写入器
    if writer is not None:
        writer.close()
    
    _logger.info(f"训练完成! 最佳准确率: {best_accuracy:.2f}%")

if __name__ == "__main__":
    main() 