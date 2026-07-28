"""
Streamlit App: Micro-expression assisted rPPG heart-rate interval estimation

功能：
    上传一段人脸视频，系统自动按时间区间估计心率，并输出：
    1. 每个时间区间的心率估计表
    2. 心率变化曲线
    3. 各 ROI 权重与面部动作强度

核心技术路线：
    视频上传
      ↓
    OpenCV 读取视频帧
      ↓
    MediaPipe Face Mesh 人脸关键点定位
      ↓
    提取额头、左脸颊、右脸颊等等 ROI
      ↓
    提取 RGB 时间序列
      ↓
    计算面部局部运动强度，近似表示微表情/面部动作干扰
      ↓
    POS / Green rPPG 信号提取
      ↓
    Welch 频谱分析估计心率
      ↓
    根据面部动作强度动态调整 ROI 权重
      ↓
    输出区间心率


rppg信号处理：
    原始ROI RGB信号
    ↓
    插值填补缺失值
    ↓
    POS/绿色通道算法提取rPPG信号
    ↓
    去趋势 + 标准化（detrend_and_normalize）
    ↓
    safe_bandpass_filter（保留心率频段）
    ↓
    Welch频谱分析（estimate_hr_from_signal）
    ↓
    心率估计结果


说明：
    1. 本代码中的“微表情模块”不是情绪分类，也不是判断心理状态。
    2. 它只利用面部关键点运动强度，辅助判断某些 ROI 是否受表情/头动影响。
    3. rPPG 才是心率估计主方法，微表情/面部动作只用于 ROI 加权。
"""

from __future__ import annotations
import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import signal

# 导入MediaPipe人脸网格模型：关键点检测
try:
    # 优先导入独立的FaceMesh模块
    from mediapipe.solutions.face_mesh import FaceMesh
except ImportError:
    # 兼容旧版本MediaPipe导入方式
    import mediapipe as mp
    FaceMesh = mp.solutions.face_mesh.FaceMesh


# 数据结构

# 定义帧级结果的数据类：存储单帧的时间、ROI的RGB值、关键点、ROI有效性
@dataclass
class FrameResult:
    # 帧对应的时间（秒）
    time_sec: float
    # 每个ROI的RGB均值数组 
    rgb_by_roi: Dict[str, np.ndarray]
    # 该帧的人脸关键点像素坐标（None表示未检测到）
    landmarks: Optional[np.ndarray]
    # 每个ROI是否有效（True=RGB提取成功）
    roi_valid: Dict[str, bool]

# 定义区间级结果的数据类：存储每个时间窗口的心率及辅助信息
@dataclass
class IntervalResult:
    # 区间起始时间
    start_sec: float
    # 区间结束时间
    end_sec: float
    # 区间中心时间
    center_sec: float
    # 融合后的心率估计值（bpm）
    hr_bpm: Optional[float]
    # 额头ROI单独估计的心率
    hr_forehead: Optional[float]
    # 左脸颊
    hr_left_cheek: Optional[float]
    # 右脸颊
    hr_right_cheek: Optional[float]
    # 额头ROI的权重（由运动强度决定）
    weight_forehead: float
    # 左脸颊
    weight_left_cheek: float
    # 右脸颊
    weight_right_cheek: float
    # 额头区域的运动强度（微表情/动作）
    motion_forehead: float
    # 左脸颊
    motion_left_cheek: float
    # 右脸颊
    motion_right_cheek: float
    # 该区间是否有效（True=心率估计成功）
    valid: bool

# MediaPipe Face Mesh landmark groups
# MediaPipe Face Mesh关键点编号（实用近似）：用于ROI划分和运动强度计算

# 额头关键点索引
FOREHEAD_POINTS = [10, 67, 69, 104, 108, 109, 151, 297, 299, 333, 337, 338]
# 左脸颊
LEFT_CHEEK_POINTS = [117, 118, 119, 123, 126, 147, 187, 203, 205, 206, 216]
# 右脸颊
RIGHT_CHEEK_POINTS = [346, 347, 348, 352, 355, 376, 411, 423, 425, 426, 436]
# 嘴巴（用于辅助计算脸颊运动）
MOUTH_POINTS = [13, 14, 78, 80, 81, 82, 87, 88, 95, 178, 308, 310, 311, 312, 317, 318, 324, 402]
# 左眼
LEFT_EYE_POINTS = [33, 133, 159, 145, 160, 144]
# 右眼
RIGHT_EYE_POINTS = [362, 263, 386, 374, 385, 380]

# ROI分组字典：键=ROI名称，值=关键点索引列表
ROI_GROUPS = {
    "forehead": FOREHEAD_POINTS,
    "left_cheek": LEFT_CHEEK_POINTS,
    "right_cheek": RIGHT_CHEEK_POINTS,
}

# ============================================================
# 基础工具函数
# ============================================================

def normalized_landmarks_to_pixels(landmarks, width: int, height: int) -> np.ndarray:
    """将 MediaPipe 归一化关键点转换为像素坐标。"""
    # 初始化关键点像素坐标列表
    pts = []
    # 遍历每个归一化关键点（x/y范围0-1）
    for lm in landmarks.landmark:
        # 转换为像素坐标：x*宽度，y*高度
        pts.append([lm.x * width, lm.y * height])
    # 转换为float32类型的numpy数组返回
    return np.asarray(pts, dtype=np.float32)

def polygon_mask_from_points(frame_shape: Tuple[int, int, int], points: np.ndarray) -> np.ndarray:
    """根据 ROI 点位生成凸包 mask
       凸包更贴合面部的自然轮廓，能精确覆盖目标皮肤区域；有效排除眼睛、鼻子、嘴巴等非皮肤区域，减少噪声
    """
    # 获取帧的高度和宽度
    h, w = frame_shape[:2]
    # 创建全黑的mask（与帧同尺寸）
    mask = np.zeros((h, w), dtype=np.uint8)
    # 关键点数量不足3个时，返回空mask
    if points.shape[0] < 3:
        return mask
    # 计算关键点的凸包（凸多边形）
    hull = cv2.convexHull(points.astype(np.int32))
    # 填充凸包区域为白色（255）
    cv2.fillConvexPoly(mask, hull, 255)
    # 返回ROI的mask（白色区域为ROI）
    return mask

def mean_rgb_in_mask(frame_bgr: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    """计算 ROI mask 内的平均 RGB"""
    # mask中无有效像素（全黑），返回None
    if mask.sum() == 0:
        return None
    # 将BGR（OpenCV默认）转换为RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # 提取mask内的所有RGB像素
    pixels = frame_rgb[mask > 0]
    # 无有效像素时返回None
    if pixels.size == 0:
        return None
    # 计算像素的RGB均值，返回float32数组
    return pixels.mean(axis=0).astype(np.float32)

def detrend_and_normalize(x: np.ndarray) -> np.ndarray:
    """去趋势 + 标准化"""
    # 转换为float64类型（提高计算精度）
    x = np.asarray(x, dtype=np.float64)
    # 去除信号的线性趋势（消除基线漂移）
    x = signal.detrend(x)
    # 计算信号标准差
    std = np.std(x)
    # 标准差过小（信号无波动），返回全零数组
    if std < 1e-8:
        return x * 0.0
    # 标准化：(x-均值)/标准差
    return (x - np.mean(x)) / std

def safe_bandpass_filter(
    x: np.ndarray,
    fs: float,
    low_hz: float = 0.75,
    high_hz: float = 3.0,
) -> Optional[np.ndarray]:
    """带通滤波（原始 rPPG 信号中包含大量与心率无关的噪声）
       保留常见心率频段（0.75-3Hz 对应 45-180bpm）
    """
    # 转换为float64类型
    x = np.asarray(x, dtype=np.float64)
    # 心率是周期性信号，要准确估计其频率，至少需要3-4个完整的周期
    # 信号长度不足（至少8个点或4秒数据），返回None
    if len(x) < max(8, int(fs * 4)):
        return None
    # 计算奈奎斯特频率（采样频率的1/2）
    nyq = fs / 2.0
    # 计算高通/低通的归一化频率（0-1之间）
    high = min(high_hz / nyq, 0.99)
    low = max(low_hz / nyq, 0.001)
    # 高低通频率无效（low>=high），返回None
    if low >= high:
        return None
    try:
        # 设计3阶巴特沃斯带通滤波器
        # 通带内的频率响应尽可能平坦，没有纹波
        b, a = signal.butter(3, [low, high], btype="bandpass")
        # 零相位双向滤波（filtfilt）：避免相位偏移
        return signal.filtfilt(b, a, x)
    except Exception:
        # 滤波失败时返回None
        return None

def estimate_hr_from_signal(
    x: np.ndarray,
    fs: float,
    min_bpm: float = 45.0,
    max_bpm: float = 180.0,
) -> Optional[float]:
    """
    从一段 rPPG 信号中估计心率
    
    通过频谱分析将时域信号转换为频域，
    找到能量最高的频率分量（主频），该频率即为脉搏的跳动频率，乘以 60 即可得到每分钟心率
    
    方法：
        1. 去趋势和标准化
        2. 带通滤波
        3. Welch 频谱分析
        4. 找主频
        5. bpm = 主频 Hz × 60
    """
    # 转换为float64类型
    x = np.asarray(x, dtype=np.float64)
    # 信号长度不足（至少4秒数据），返回None
    if len(x) < int(fs * 4):
        return None
    
    # 步骤1：去趋势+标准化
    x = detrend_and_normalize(x)
    # 步骤2：带通滤波（保留心率对应频段）
    filtered = safe_bandpass_filter(
        x,
        fs,
        low_hz=min_bpm / 60.0,  # 最低心率转换为Hz
        high_hz=max_bpm / 60.0, # 最高心率转换为Hz
    )

    # 滤波后信号无效（None或无波动），返回None
    if filtered is None or np.std(filtered) < 1e-8:
        return None
    
    # 步骤3：Welch频谱分析（计算功率谱密度PSD）
    # 分段处理、加窗、平均
    # 设定每段的长度：不超过信号长度，且至少32点/8秒（8秒分段的频率分辨率0.125Hz，对应7.5bpm）数据
    nperseg = min(len(filtered), max(32, int(fs * 8)))
    # 计算频率轴和功率谱
    freqs, power = signal.welch(filtered, fs=fs, nperseg=nperseg)
    # 筛选出心率对应频率区间的频率/功率
    mask = (freqs >= min_bpm / 60.0) & (freqs <= max_bpm / 60.0)
    if not np.any(mask):
        return None
    # 提取有效频率和功率
    freqs_roi = freqs[mask]
    power_roi = power[mask]
    # 无有效功率值，返回None
    if len(power_roi) == 0 or np.all(power_roi <= 0):
        return None
    
    # 步骤4：找到最大功率对应的频率（主频）
    peak_freq = freqs_roi[np.argmax(power_roi)]
    
    # 步骤5：转换为心率（bpm）并返回
    return float(peak_freq * 60.0)


# ============================================================
# rPPG 信号提取方法
# ============================================================

def green_channel_signal(rgb: np.ndarray) -> np.ndarray:
    """Green Channel baseline：最简单的rPPG方法，提取绿色通道信号。"""
    # 返回RGB数组的第二列（G通道）
    return rgb[:, 1]

def pos_signal(rgb: np.ndarray) -> np.ndarray:
    """
    POS-inspired rPPG signal extraction.
    POS 思路：
        面部皮肤的 RGB 像素值变化由静态分量（皮肤本身的色调和环境光照强度决定）和动态分量组成（由心脏搏动引起的皮肤血液容积周期性变化导致）
        在归一化 RGB 空间中构建与皮肤色调正交的投影，增强脉搏信号。
    """
    # 转换为float64类型
    rgb = np.asarray(rgb, dtype=np.float64)
    # 校验输入形状：必须是[T, 3]（T帧，每帧RGB）
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError("rgb must have shape [T, 3]")
    
    # RGB 信号归一化
    mean_rgb = np.mean(rgb, axis=0) # 计算RGB序列的均值
    mean_rgb = np.where(mean_rgb == 0, 1e-6, mean_rgb) # np.where 代替直接赋值，避免除以0错误
    c = rgb / mean_rgb - 1.0   # 归一化RGB信号：(当前RGB/均值) - 1
    
    # 计算两个正交分量（脉搏信号的变化主要集中在G-B和G+B-2R两个方向上）
    s1 = c[:, 1] - c[:, 2]  # G - B
    s2 = c[:, 1] + c[:, 2] - 2.0 * c[:, 0]  # G + B - 2R
    
    # 平衡两个分量的能量
    std_s2 = np.std(s2)# 计算s2的标准差
    alpha = 0.0 if std_s2 < 1e-8 else np.std(s1) / std_s2# 计算平衡系数alpha（避免除以0）
    
    # 融合分量得到POS rPPG信号
    h = s1 + alpha * s2
    
    # 返回float64类型的信号
    return h.astype(np.float64)

def rgb_to_rppg_signal(rgb: np.ndarray, method: str = "pos") -> np.ndarray:
    """根据指定方法将RGB序列转换为rPPG信号。"""
    if method == "green":
        # 绿色通道方法
        return green_channel_signal(rgb)
    if method == "pos":
        # POS方法
        return pos_signal(rgb)
    # 不支持的方法抛出异常
    raise ValueError(f"Unsupported rPPG method: {method}")


# ============================================================
# 微表情 / 面部动作辅助模块
# ============================================================

def compute_group_motion(
    landmarks_seq: List[Optional[np.ndarray]],
    point_indices: List[int],  # 关键点索引列表
) -> float:
    """
    计算某个面部区域的关键点运动强度。
    面部局部运动（包括微表情、说话、眨眼、轻微头动）会破坏对应区域的 rPPG 信号质量，
    因此需要根据运动强度动态降低受干扰区域的权重，提升信号干净区域的贡献
    用途：
        作为微表情/面部动作的近似特征。
    解释：
        运动强度越大，说明这个区域越可能受表情、说话、头动影响。
        对应 ROI 的权重应该降低。
    """
    # 过滤出有效关键帧（非None）
    valid = [lm for lm in landmarks_seq if lm is not None]
    # 有效帧不足2个，返回极大值（表示运动强度极高）
    if len(valid) < 2:
        return 1e6
    # 初始化运动强度列表
    motions = []
    
    # 通过相邻帧关键点的位移来量化某个区域的运动程度

    # 计算相邻帧之间的平均位移
    # 遍历相邻帧对
    for prev, curr in zip(valid[:-1], valid[1:]):
        # 关键点索引超出范围，跳过
        if max(point_indices) >= len(curr):
            continue
        # 提取当前区域的前一帧/当前帧关键点
        p0 = prev[point_indices]
        p1 = curr[point_indices]
        # 计算关键点的平均位移
        displacement = np.linalg.norm(p1 - p0, axis=1).mean()
        
        # 用脸宽做归一化（避免分辨率影响）
        try:
            # 脸宽：左右脸颊关键点（234/454）的欧氏距离
            face_width = np.linalg.norm(curr[234] - curr[454])
        except Exception:
            # 计算失败时默认脸宽为100
            face_width = 100.0
        # 确保脸宽不为0
        face_width = max(face_width, 1.0)
        # 归一化位移并加入列表
        motions.append(displacement / face_width)
    
    # 无有效运动数据，返回大值
    if not motions:
        return 1e6
    # 返回平均运动强度
    return float(np.mean(motions))

def motion_to_weight(motion: float, temperature: float = 30.0) -> float:
    """
    将运动强度转为 ROI 权重。

    运动越大，权重越小（指数衰减）。
    T控制权重对运动的敏感程度
    """
    # 运动强度非有限值（inf/nan），返回0权重
    if not np.isfinite(motion):
        return 0.0
    # 指数衰减函数：exp(-温度系数×运动强度)
    return float(np.exp(-temperature * motion))

def normalize_weights(raw_weights: Dict[str, float]) -> Dict[str, float]:
    """归一化ROI权重（确保权重和为1）。"""
    # 计算所有权重的总和（仅保留非负权重）
    total = sum(max(v, 0.0) for v in raw_weights.values())
    # 总和过小，返回等权重
    if total <= 1e-8:
        n = len(raw_weights)
        return {k: 1.0 / n for k in raw_weights}
    # 归一化：每个权重/总和（确保非负）
    return {k: max(v, 0.0) / total for k, v in raw_weights.items()}


# ============================================================
# 视频处理，检测人脸关键点（包括面部捕捉）
# ============================================================

def process_video(video_path: str, progress_callback=None) -> Tuple[List[FrameResult], float, float]:
    """
    读取视频，检测人脸关键点，并提取三个 ROI 的 RGB 均值序列。
    
    返回值：
        - FrameResult列表：每帧的处理结果
        - fps：视频帧率
        - duration：视频总时长（秒）
    """
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    # 视频打开失败，抛出异常
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    # 获取视频帧率（FPS）
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 帧率无效时，默认30FPS
    if fps is None or fps <= 1e-6:
        fps = 30.0
    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 计算视频总时长（秒）
    duration = total_frames / fps if total_frames > 0 else 0.0
    # 初始化帧结果列表
    results: List[FrameResult] = []
    # 帧索引初始化
    frame_idx = 0

    # 初始化MediaPipe Face Mesh模型（面部捕捉）
    with FaceMesh(
        static_image_mode=False,  # 视频流模式（非静态图片）
        max_num_faces=1,          # 最多检测1张人脸
        refine_landmarks=True,    # 启用精细化关键点
        min_detection_confidence=0.5,  # 检测置信度阈值
        min_tracking_confidence=0.5,   # 跟踪置信度阈值
    ) as face_mesh:

        # 逐帧读取视频
        while True:
            # 读取一帧（ret=是否成功，frame=帧数据）
            ret, frame = cap.read()
            if not ret or frame is None:
                break  

            # 获取帧的高度和宽度
            h, w = frame.shape[:2]
            # 计算当前帧的时间（秒）
            time_sec = frame_idx / fps
            # 转换BGR为RGB（MediaPipe要求RGB输入）
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 执行面部捕捉，得到关键点
            mp_result = face_mesh.process(frame_rgb)

            # 初始化ROI的RGB字典
            rgb_by_roi: Dict[str, np.ndarray] = {}
            # 初始化ROI有效性字典
            roi_valid: Dict[str, bool] = {}
            # 初始化关键点像素坐标
            landmarks_px: Optional[np.ndarray] = None

            # 检测到人脸
            if mp_result.multi_face_landmarks:
                # 将归一化关键点转换为像素坐标（MediaPipe 返回的关键点是归一化坐标）
                landmarks_px = normalized_landmarks_to_pixels(
                    mp_result.multi_face_landmarks[0],
                    width=w,
                    height=h,
                )

                # 遍历每个ROI分组，将关键点像素坐标转化为ROI
                for roi_name, indices in ROI_GROUPS.items():
                    # 提取当前ROI的关键点像素坐标
                    pts = landmarks_px[indices]
                    # 生成ROI的凸包mask
                    mask = polygon_mask_from_points(frame.shape, pts)
                    # 计算mask内的RGB均值
                    mean_rgb = mean_rgb_in_mask(frame, mask)

                    # RGB均值提取失败
                    if mean_rgb is None:
                        rgb_by_roi[roi_name] = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
                        roi_valid[roi_name] = False
                    else:
                        # 存储RGB均值
                        rgb_by_roi[roi_name] = mean_rgb
                        roi_valid[roi_name] = True
            else:
                # 未检测到人脸，ROI均无效
                for roi_name in ROI_GROUPS:
                    rgb_by_roi[roi_name] = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
                    roi_valid[roi_name] = False

            # 将当前帧结果加入列表
            results.append(
                FrameResult(
                    time_sec=time_sec,
                    rgb_by_roi=rgb_by_roi,
                    landmarks=landmarks_px,
                    roi_valid=roi_valid,
                )
            )

            # 帧索引+1
            frame_idx += 1

            # 更新进度（每10帧更新一次）
            if progress_callback and total_frames > 0 and frame_idx % 10 == 0:
                progress_callback(min(frame_idx / total_frames, 1.0))

    # 释放视频资源
    cap.release()

    # 进度回调：100%完成
    if progress_callback:
        progress_callback(1.0)

    # 返回帧结果、帧率、时长
    return results, float(fps), float(duration)


# ============================================================
# 区间心率估计
# ============================================================

def interpolate_nan_signal(x: np.ndarray) -> Optional[np.ndarray]:
    """
    对缺失 RGB / 信号做线性插值（填补NaN值）。
    因为心率估计要求信号在时间上是连续的（等间隔采样），缺失值（NaN）会导致频谱分析失败
    """
    # 转换为float64类型
    x = np.asarray(x, dtype=np.float64)

    # 一维信号（如单通道rPPG）
    if x.ndim == 1:
        # 标记有效数据（非NaN/inf）
        valid = np.isfinite(x)
        # 有效数据不足50%或少于3个，返回None
        if valid.sum() < max(3, len(x) * 0.5):
            return None
        # 生成索引数组
        idx = np.arange(len(x))
        # 复制信号
        y = x.copy()
        # 对无效数据做线性插值
        y[~valid] = np.interp(idx[~valid], idx[valid], x[valid])
        return y

    # 二维信号（如RGB序列，T行三列）
    if x.ndim == 2:
        cols = []
        # 逐列插值
        for i in range(x.shape[1]):
            col = interpolate_nan_signal(x[:, i])
            # 某列插值失败，整体返回None
            if col is None:
                return None
            cols.append(col)
        # 合并列为二维数组
        return np.stack(cols, axis=1)

    # 其他维度，返回None
    return None

def estimate_intervals(
    frame_results: List[FrameResult],
    fps: float,
    method: str = "pos",
    window_sec: float = 10.0,
    step_sec: float = 10.0,
    min_bpm: float = 45.0,
    max_bpm: float = 180.0,
) -> List[IntervalResult]:
    """
    每个时间窗口估计一次心率。

    window_sec = 10, step_sec = 10：
        输出 0-10s, 10-20s, 20-30s ...
    window_sec = 10, step_sec = 1：
        输出 0-10s, 1-11s, 2-12s ... 更像连续曲线。
    """
    # 无帧结果，返回空列表
    if not frame_results:
        return []

    # 1.时间窗口生成

    # 提取所有帧的时间
    times = np.array([fr.time_sec for fr in frame_results])
    # 视频总时长（最后一帧时间 + 单帧时长）
    duration = times[-1] + 1.0 / fps

    # 初始化区间结果列表
    interval_results: List[IntervalResult] = []
    # 区间起始时间初始化
    start = 0.0

    # 滑动窗口遍历视频时长
    while start + window_sec <= duration + 1e-6:
        # 计算区间结束时间
        end = start + window_sec
        # 计算区间中心时间
        center = (start + end) / 2.0
        # 找到该区间内的帧索引
        idx = np.where((times >= start) & (times < end))[0]
         
        # 2.窗口有效性判断

        # 有效帧不足（至少4秒/50%窗口时长），标记为无效区间
        if len(idx) < int(fps * min(4.0, window_sec * 0.5)):
            interval_results.append(
                IntervalResult(
                    start_sec=start,
                    end_sec=end,
                    center_sec=center,
                    hr_bpm=None,
                    hr_forehead=None,
                    hr_left_cheek=None,
                    hr_right_cheek=None,
                    weight_forehead=0.0,
                    weight_left_cheek=0.0,
                    weight_right_cheek=0.0,
                    motion_forehead=0.0,
                    motion_left_cheek=0.0,
                    motion_right_cheek=0.0,
                    valid=False,
                )
            )
            # 滑动到下一个区间
            start += step_sec
            continue

        # 3.面部运动强度计算与权重生成

        # 提取该区间内的关键点序列
        landmarks_seq = [frame_results[i].landmarks for i in idx]

        # 计算各ROI的基础运动强度
        roi_motion = {
            "forehead": compute_group_motion(landmarks_seq, FOREHEAD_POINTS),
            "left_cheek": compute_group_motion(landmarks_seq, LEFT_CHEEK_POINTS),
            "right_cheek": compute_group_motion(landmarks_seq, RIGHT_CHEEK_POINTS),
        }

        # 计算辅助区域（嘴、眼）的运动强度
        mouth_motion = compute_group_motion(landmarks_seq, MOUTH_POINTS)
        left_eye_motion = compute_group_motion(landmarks_seq, LEFT_EYE_POINTS)
        right_eye_motion = compute_group_motion(landmarks_seq, RIGHT_EYE_POINTS)

        # 面部动作辅助 ROI 加权：
        # 眼眉区域运动明显，额头更容易受影响。
        # 嘴部运动明显，脸颊区域可能受说话/表情牵拉影响。
        adjusted_motion = {
            "forehead": roi_motion["forehead"] + 0.5 * (left_eye_motion + right_eye_motion),
            "left_cheek": roi_motion["left_cheek"] + 0.3 * mouth_motion,
            "right_cheek": roi_motion["right_cheek"] + 0.3 * mouth_motion,
        }

        # 将调整后的运动强度转换为原始权重
        raw_weights = {roi: motion_to_weight(m) for roi, m in adjusted_motion.items()}
        # 归一化权重（总和为1）
        weights = normalize_weights(raw_weights)

        # 4.单ROI心率估计

        # 初始化各ROI的心率估计结果
        hr_by_roi: Dict[str, Optional[float]] = {}

        # 遍历每个ROI，单独估计心率
        for roi in ROI_GROUPS.keys():
            # 提取该ROI的RGB序列
            rgb = np.array([frame_results[i].rgb_by_roi[roi] for i in idx], dtype=np.float64)
            # 插值填补NaN值
            rgb_interp = interpolate_nan_signal(rgb)

            # 插值失败，标记为None
            if rgb_interp is None:
                hr_by_roi[roi] = None
                continue

            try:
                # 转换RGB为rPPG信号
                rppg = rgb_to_rppg_signal(rgb_interp, method=method)
                # 从rPPG信号估计心率
                hr = estimate_hr_from_signal(
                    rppg,
                    fs=fps,
                    min_bpm=min_bpm,
                    max_bpm=max_bpm,
                )
            except Exception:
                # 异常时心率为None
                hr = None

            # 存储该ROI的心率估计结果
            hr_by_roi[roi] = hr


        # 5.多ROI心率融合

        # 筛选出有效心率的ROI
        available = {roi: hr for roi, hr in hr_by_roi.items() if hr is not None}

        # 无有效ROI心率
        if not available:
            fused_hr = None
            valid = False
        else:
            # 提取有效ROI的权重
            available_weights = {roi: weights[roi] for roi in available}
            # 重新归一化有效ROI的权重（因为部分 ROI 被排除，原来的权重总和不再是1）
            available_weights = normalize_weights(available_weights)
            # 加权融合心率
            fused_hr = sum(available[roi] * available_weights[roi] for roi in available)
            # 标记为有效区间
            valid = True

        # 6.结果存储

        # 添加当前区间结果到列表
        interval_results.append(
            IntervalResult(
                start_sec=start,
                end_sec=end,
                center_sec=center,
                hr_bpm=fused_hr,
                hr_forehead=hr_by_roi.get("forehead"),
                hr_left_cheek=hr_by_roi.get("left_cheek"),
                hr_right_cheek=hr_by_roi.get("right_cheek"),
                weight_forehead=weights.get("forehead", 0.0),
                weight_left_cheek=weights.get("left_cheek", 0.0),
                weight_right_cheek=weights.get("right_cheek", 0.0),
                motion_forehead=adjusted_motion.get("forehead", 0.0),
                motion_left_cheek=adjusted_motion.get("left_cheek", 0.0),
                motion_right_cheek=adjusted_motion.get("right_cheek", 0.0),
                valid=valid,
            )
        )

        # 滑动到下一个区间
        start += step_sec

    # 返回所有区间结果
    return interval_results


# ============================================================
# 后处理与可视化
# ============================================================

def interval_results_to_dataframe(results: List[IntervalResult]) -> pd.DataFrame:
    """将区间结果转换为Pandas DataFrame（便于展示/保存）。"""
    # 初始化行列表
    rows = []
    # 遍历每个区间结果
    for r in results:
        # 转换为字典行
        rows.append(
            {
                "start_sec": r.start_sec,
                "end_sec": r.end_sec,
                "center_sec": r.center_sec,
                "hr_bpm": r.hr_bpm,
                "hr_forehead": r.hr_forehead,
                "hr_left_cheek": r.hr_left_cheek,
                "hr_right_cheek": r.hr_right_cheek,
                "weight_forehead": r.weight_forehead,
                "weight_left_cheek": r.weight_left_cheek,
                "weight_right_cheek": r.weight_right_cheek,
                "motion_forehead": r.motion_forehead,
                "motion_left_cheek": r.motion_left_cheek,
                "motion_right_cheek": r.motion_right_cheek,
                "valid": r.valid,
            }
        )
    # 转换为DataFrame并返回
    return pd.DataFrame(rows)

def smooth_hr_values(df: pd.DataFrame, median_kernel: int = 3) -> pd.DataFrame:
    """对心率曲线做简单中值滤波（平滑），不删除任何时间区间。"""
    # 复制DataFrame（避免修改原数据）
    df = df.copy()
    # 空DataFrame或无hr_bpm列，添加空的平滑列
    if df.empty or "hr_bpm" not in df:
        df["hr_bpm_smooth"] = []
        return df

    # 提取心率数组
    hr = df["hr_bpm"].astype(float).to_numpy()
    # 标记有效心率
    valid = np.isfinite(hr)

    # 有效心率不足3个，直接返回原数据
    if valid.sum() < 3:
        df["hr_bpm_smooth"] = hr
        return df

    # 生成索引数组
    idx = np.arange(len(hr))
    # 复制心率数组
    hr_interp = hr.copy()
    # 对无效心率做线性插值（填补NaN）
    hr_interp[~valid] = np.interp(idx[~valid], idx[valid], hr[valid])

    # 确保中值滤波核为奇数
    if median_kernel % 2 == 0:
        median_kernel += 1
    # 核大小至少为3
    median_kernel = max(3, median_kernel)

    # 中值滤波平滑
    if len(hr_interp) >= median_kernel:
        hr_smooth = signal.medfilt(hr_interp, kernel_size=median_kernel)
    else:
        # 数据长度不足，使用插值后的数据
        hr_smooth = hr_interp

    # 添加平滑列到DataFrame
    df["hr_bpm_smooth"] = hr_smooth
    return df

def make_hr_curve_figure(df: pd.DataFrame):
    """生成心率曲线图表（原始+平滑）。"""
    # 创建画布和轴
    fig, ax = plt.subplots(figsize=(10, 5))
    # 非空DataFrame才绘图
    if not df.empty:
        # 绘制原始心率曲线
        ax.plot(df["center_sec"], df["hr_bpm"], marker="o", label="Estimated HR")
        # 绘制平滑心率曲线（如果存在）
        if "hr_bpm_smooth" in df:
            ax.plot(df["center_sec"], df["hr_bpm_smooth"], marker="s", label="Smoothed HR")
    # 设置X轴标签
    ax.set_xlabel("Time (s)")
    # 设置Y轴标签
    ax.set_ylabel("Heart Rate (bpm)")
    # 设置标题
    ax.set_title("Video Heart Rate Curve from rPPG")
    # 显示网格
    ax.grid(True, alpha=0.3)
    # 显示图例
    ax.legend()
    # 调整布局
    fig.tight_layout()
    # 返回图表对象
    return fig

def make_roi_weight_figure(df: pd.DataFrame):
    """生成ROI权重曲线图表。"""
    # 创建画布和轴
    fig, ax = plt.subplots(figsize=(10, 5))
    # 非空DataFrame才绘图
    if not df.empty:
        # 绘制额头权重
        ax.plot(df["center_sec"], df["weight_forehead"], marker="o", label="Forehead weight")
        # 绘制左脸颊权重
        ax.plot(df["center_sec"], df["weight_left_cheek"], marker="o", label="Left cheek weight")
        # 绘制右脸颊权重
        ax.plot(df["center_sec"], df["weight_right_cheek"], marker="o", label="Right cheek weight")
    # 设置X轴标签
    ax.set_xlabel("Time (s)")
    # 设置Y轴标签
    ax.set_ylabel("ROI Weight")
    # 设置标题
    ax.set_title("Motion-aware ROI Weights")
    # 显示网格
    ax.grid(True, alpha=0.3)
    # 显示图例
    ax.legend()
    # 调整布局
    fig.tight_layout()
    # 返回图表对象
    return fig


# ============================================================
# Streamlit 视频上传接口
# ============================================================

def save_uploaded_video(uploaded_file) -> str:
    """将 Streamlit 上传的视频保存为临时文件，并返回路径。"""
    # 获取文件后缀（如.mp4）
    suffix = os.path.splitext(uploaded_file.name)[-1]
    # 无后缀时默认.mp4
    if suffix == "":
        suffix = ".mp4"
    # 创建临时文件（不自动删除）
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    # 写入上传的文件数据
    temp.write(uploaded_file.read())
    # 刷新缓冲区
    temp.flush()
    # 关闭文件（但不删除）
    temp.close()
    # 返回临时文件路径
    return temp.name

def main():
    """Streamlit应用主函数：构建界面+处理逻辑。"""
    # 设置页面配置
    st.set_page_config(
        page_title="微表情辅助 rPPG 心率估计系统",  # 页面标题
        layout="wide",  # 宽布局
    )

    # 页面主标题
    st.title("基于微表情辅助 rPPG 的视频心率区间估计系统")

    # 页面说明文字
    st.markdown(
        """
        上传一段包含人脸的视频，系统会自动提取额头、左脸颊、右脸颊 ROI，
        利用 rPPG 估计每个时间区间的心率，并根据面部局部运动强度动态调整 ROI 权重。
        """
    )

    # 侧边栏：参数设置
    with st.sidebar:
        st.header("参数设置")

        # 选择rPPG方法
        method = st.selectbox(
            "rPPG 方法",
            options=["pos", "green"],
            index=0,
            help="pos 更推荐；green 是简单绿色通道基线。",
        )

        # 分析窗口长度（秒）
        window_sec = st.slider(
            "分析窗口长度 / 秒",
            min_value=6.0,
            max_value=30.0,
            value=10.0,
            step=1.0,
            help="你的项目建议使用 10 秒左右。",
        )

        # 区间方式选择（固定/滑动）
        step_mode = st.radio(
            "区间方式",
            options=["固定 10 秒区间", "滑动窗口曲线"],
            index=0,
        )

        # 根据区间方式设置步长
        if step_mode == "固定 10 秒区间":
            step_sec = window_sec
        else:
            # 滑动窗口步长
            step_sec = st.slider(
                "滑动步长 / 秒",
                min_value=0.5,
                max_value=5.0,
                value=1.0,
                step=0.5,
            )

        # 最低心率阈值（bpm）
        min_bpm = st.number_input("最低心率 bpm", min_value=30.0, max_value=100.0, value=45.0, step=5.0)
        # 最高心率阈值（bpm）
        max_bpm = st.number_input("最高心率 bpm", min_value=100.0, max_value=220.0, value=180.0, step=5.0)

        # 是否平滑心率曲线
        smooth = st.checkbox("输出平滑心率曲线", value=True)

    # 文件上传组件：支持视频格式
    uploaded_file = st.file_uploader(
        "上传视频文件",
        type=["mp4", "avi", "mov", "mkv"],
    )

    # 未上传文件时显示提示
    if uploaded_file is None:
        st.info("请先上传一段包含清晰人脸的视频。建议视频时长不少于 20 秒，光照尽量稳定。")
        return

    # 保存上传的视频到临时文件
    video_path = save_uploaded_video(uploaded_file)

    # 显示上传的视频
    st.subheader("输入视频")
    st.video(uploaded_file)

    # 开始分析按钮
    if st.button("开始分析", type="primary"):
        try:
            # 创建进度条
            progress_bar = st.progress(0.0)
            # 状态提示文本框
            status = st.empty()

            # 更新状态：读取视频+提取ROI
            status.write("正在读取视频、检测人脸关键点并提取 ROI RGB 信号……")

            # 进度回调函数
            def update_progress(p):
                progress_bar.progress(float(p))

            # 处理视频：提取帧结果、帧率、时长
            frame_results, fps, duration = process_video(video_path, progress_callback=update_progress)

            # 视频读取完成提示
            st.success(f"视频读取完成：FPS = {fps:.2f}，时长约 {duration:.2f} 秒，帧数 = {len(frame_results)}")

            # 更新状态：估计区间心率
            status.write("正在按时间区间估计心率……")

            # 估计每个区间的心率
            interval_results = estimate_intervals(
                frame_results=frame_results,
                fps=fps,
                method=method,
                window_sec=window_sec,
                step_sec=step_sec,
                min_bpm=min_bpm,
                max_bpm=max_bpm,
            )

            # 转换为DataFrame
            df = interval_results_to_dataframe(interval_results)

            # 平滑心率曲线（如果开启）
            if smooth:
                df = smooth_hr_values(df, median_kernel=3)
            else:
                # 未开启平滑，平滑列等于原始列
                df["hr_bpm_smooth"] = df["hr_bpm"]

            # 更新状态：分析完成
            status.write("分析完成。")

            # 无区间结果时显示错误
            if df.empty:
                st.error("没有得到有效区间。请检查视频长度是否足够，或是否能检测到人脸。")
                return

            # 统计有效/总区间数
            valid_count = int(df["valid"].sum())
            total_count = len(df)

            # 显示统计指标（三列布局）
            col1, col2, col3 = st.columns(3)
            col1.metric("总区间数", total_count)
            col2.metric("有效区间数", valid_count)
            # 显示平均心率（有有效结果时）
            if valid_count > 0:
                avg_hr = df.loc[df["valid"], "hr_bpm"].mean()
                col3.metric("平均估计心率", f"{avg_hr:.1f} bpm")
            else:
                col3.metric("平均估计心率", "无有效结果")

            # 显示心率曲线
            st.subheader("心率曲线")
            fig = make_hr_curve_figure(df)
            st.pyplot(fig)

            # 显示区间心率表格
            st.subheader("每个时间区间的心率估计")
            # 复制DataFrame用于展示
            display_df = df.copy()
            # 生成时间区间字符串（如0.0-10.0s）
            display_df["time_interval"] = display_df.apply(
                lambda row: f"{row['start_sec']:.1f}-{row['end_sec']:.1f}s",
                axis=1,
            )
            # 选择要展示的列
            cols_to_show = [
                "time_interval",
                "hr_bpm",
                "hr_bpm_smooth",
                "hr_forehead",
                "hr_left_cheek",
                "hr_right_cheek",
                "weight_forehead",
                "weight_left_cheek",
                "weight_right_cheek",
                "valid",
            ]
            # 显示DataFrame（自适应宽度）
            st.dataframe(display_df[cols_to_show], use_container_width=True)

            # 显示ROI权重曲线
            st.subheader("微表情/面部动作辅助的 ROI 权重")
            st.pyplot(make_roi_weight_figure(df))

            # 下载结果
            st.subheader("下载结果")
            # 转换为CSV（UTF-8编码，支持中文）
            csv = df.to_csv(index=False, encoding="utf-8-sig")
            # 下载按钮
            st.download_button(
                label="下载 CSV 结果",
                data=csv,
                file_name="rppg_heart_rate_intervals.csv",
                mime="text/csv",
            )

            # 结果解释
            st.markdown(
                """
                **结果解释：**
                - `hr_bpm`：当前时间区间的原始融合心率估计。
                - `hr_bpm_smooth`：平滑后的心率，用于画更稳定的曲线。
                - `hr_forehead / hr_left_cheek / hr_right_cheek`：不同 ROI 单独估计的心率。
                - `weight_*`：由面部动作强度得到的 ROI 权重。运动越明显，权重越低。
                - `valid=False`：该区间可能人脸检测失败、信号太短或 rPPG 信号质量不足。
                """
            )

        except Exception as e:
            # 捕获异常并显示错误信息
            st.error(f"分析失败：{e}")

        finally:
            try:
                # 删除临时视频文件
                os.remove(video_path)
            except Exception:
                # 忽略删除失败（如文件已被删除）
                pass

# 主函数入口
if __name__ == "__main__":
    main()