#
# timeseries.py
# @author Quexuan Zhang
# @description
# @created 2024-12-04T19:37:43.491Z+08:00
# @last-modified 2026-01-15T18:47:01.623Z+08:00
#

from collections.abc import Sequence
from itertools import chain, pairwise

import numpy as np
from dtaidistance import dtw
from numba import njit
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import zscore

from .geometry import get_polygon_areas, get_polygon_vertices
from .keypoint import KeypointSeq
from .preprocessing import minmax_over_frames

Interval = tuple[int, int]

def hampel_filter(data, window_size=5, n_sigmas=3):
    """
    Hampel 滤波器实现

    Arguments:
        data: 输入的一维 numpy 数组 (x 或 y 坐标序列)
        window_size: 滑动窗口半径 (k)，实际窗口长度为 2k+1
        n_sigmas: 判别离群值的阈值系数，默认为 3

    Returns:
        滤波后的序列
    """
    n = len(data)
    new_data = data.copy()
    # 这里的 1.4826 是正态分布下的缩放因子
    k = 1.4826

    for i in range(window_size, n - window_size):
        # 提取窗口数据
        window = data[i - window_size : i + window_size + 1]
        x_median = np.median(window)

        # 计算 MAD
        mad = np.median(np.abs(window - x_median))
        sigma = k * mad

        # 判断并替换离群值
        if np.abs(data[i] - x_median) > n_sigmas * sigma:
            new_data[i] = x_median

    return new_data

def adaptive_gaussian_smooth(data, sigma_min=1, sigma_max=4, window_size=20):
    """
    自适应高斯平滑

    Arguments:
        sigma_min: 在波峰/波谷处使用的平滑强度（保护特征）
        sigma_max: 在平坦/噪声区使用的平滑强度（强力去噪）
        window_size: 用于评估局部波动的窗口大小

    Returns:
        滤波后的序列
    """
    # 1. 计算局部波动（标准差）
    # 使用滑动窗口标准差来衡量该点是否属于“变化剧烈区”
    local_std = np.array(
        [
            np.std(
                data[
                    max(0, i - window_size // 2) : min(len(data), i + window_size // 2)
                ]
            )
            for i in range(len(data))
        ]
    )

    # 2. 归一化波动权重 (0 到 1)
    # 波动越大，weight 越接近 1
    norm_std = (local_std - np.min(local_std)) / (
        np.max(local_std) - np.min(local_std) + 1e-6
    )

    # 3. 预计算两个极端的平滑效果
    low_smooth = gaussian_filter1d(data, sigma=sigma_min)  # 弱平滑，保留细节
    high_smooth = gaussian_filter1d(data, sigma=sigma_max)  # 强平滑，去除噪声

    # 4. 关键：根据局部波动进行融合
    # 在波动大（norm_std高）的地方，更多地使用 low_smooth
    # 在波动小（norm_std低）的地方，更多地使用 high_smooth
    # 注意：这里的逻辑是 (1 - norm_std)，因为波动越大 sigma 应该越小
    smoothed_data = norm_std * low_smooth + (1 - norm_std) * high_smooth

    return smoothed_data

def smooth_frame_arr(frame_arr):
    for i in range(21):
        for c in range(2):
            frame_arr[:, i, c] = hampel_filter(
                frame_arr[:, i, c], window_size=2, n_sigmas=3
            )
            # frame_arr[:, i, c] = savgol_filter(
            #     frame_arr[:, i, c], window_length=7, polyorder=2
            # )


def find_period_peaks(
    data: np.ndarray, sigma_min=1, sigma_max=4, prominence=0.4, **kwargs
) -> tuple[np.ndarray, np.ndarray]:
    """利用高斯滤波平滑器平滑数据后，寻找极值点

    Arguments:
        data -- 参考数据

    Keyword Arguments:
        base_sigma -- 自适应高斯滤波器初始标准差 (default: {1.})
        **kwargs -- 传递给`scipy.signal.find_peaks`函数的具名参数

    Returns:
        极大值点数组和极小值点数组
    """
    smoothed = adaptive_gaussian_smooth(data, sigma_min=sigma_min, sigma_max=sigma_max)
    peaks_max, _ = find_peaks(smoothed, prominence=prominence, **kwargs)  # 极大值
    peaks_min, _ = find_peaks(-smoothed, prominence=prominence, **kwargs)  # 极小值

    return (peaks_min, peaks_max)


def find_period_peaks_by_area(
    frame_arr: np.ndarray,
    vertice_ids: KeypointSeq,
    left: bool,
    prominence=0.4,
    **kwargs,
) -> tuple[list[int], list[int]]:
    """根据面积变化获取伸握过程周期极值点数组

    Arguments:
        frame_arr -- 关键点帧数据数组
        vertice_ids -- 面积顶点ID序列，按右手顺时针序
        left -- 是否左手

    Keyword Arguments:
        base_sigma -- 自适应高斯滤波器初始标准差 (default: {1.})
        prominence -- 峰值显著度阈值，传递给`scipy.signal.find_peaks` (default: {1.})
        **kwargs -- 其他传递给`scipy.signal.find_peaks`函数的具名参数

    Returns:
        极小值点数组和极大值点数组
    """
    vertices = get_polygon_vertices(frame_arr, vertice_ids, left)
    areas = get_polygon_areas(vertices)
    return find_period_peaks(areas, prominence=prominence, **kwargs)


def get_period_intervals(
    peaks_min: np.ndarray, peaks_max: np.ndarray
) -> tuple[list[Interval], list[Interval]]:
    """根据周期极值返回伸握过程变化点区间对数组

    Arguments:
        peaks_min -- 极小值点数组
        peaks_max -- 极大值点数组
    Returns:
        伸过程变化点区间数组和握过程变化点区间数组
    """
    periods_open = []
    periods_close = []

    if len(peaks_min) == 0 or len(peaks_max) == 0:
        return [], []

    if peaks_min[0] < peaks_max[0]:
        zipped = zip(peaks_min, np.append(peaks_max, -1))
        start = 0
    else:
        zipped = zip(peaks_max, np.append(peaks_min, -1))
        start = 1

    for i, pair in enumerate(pairwise(chain.from_iterable(zipped)), start):
        if pair[1] == -1:
            continue

        if i % 2 == 0:
            periods_open.append(pair)
        else:
            periods_close.append(pair)

    return periods_open, periods_close


def find_period_intervals_by_area(
    frame_arr: np.ndarray,
    vertice_ids: KeypointSeq,
    left: bool,
    base_sigma: float = 1.0,
    prominence: float | tuple[float, float] = 0.4,
    **kwargs
) -> tuple[list[Interval], list[Interval]]:
    """根据面积变化获取伸握过程周期变化点区间数组

    Arguments:
        frame_arr -- 关键点帧数据数组
        vertice_ids -- 面积顶点ID序列，按右手顺时针序
        left -- 是否左手

    Keyword Arguments:
        base_sigma -- 自适应高斯滤波器初始标准差 (default: {1.})
        prominence -- 峰值显著度阈值，传递给`scipy.signal.find_peaks` (default: {1.})
        **kwargs -- 其他传递给`scipy.signal.find_peaks`函数的具名参数

    Returns:
        伸过程周期变化点区间数组和握过程周期变化点区间数组
    """
    peaks_min, peaks_max = find_period_peaks_by_area(
        frame_arr, vertice_ids, left, prominence
    )
    return get_period_intervals(peaks_min, peaks_max)


@njit
def find_pause_intervals(
    data: np.ndarray,
    peaks_min: np.ndarray,
    peaks_max: np.ndarray,
    threshold: Interval = (-0.5, 0.5),
) -> tuple[list[Interval], list[Interval]]:
    """寻找平台区间，值在阈值区间内的点视为平台点

    Arguments:
        data -- 一维数据数组
        peaks_min -- 极小值点数组
        peaks_max -- 极大值点数组

    Keyword Arguments:
        threshold -- 平台阈值 (default: {(-0.5, 0.5)})

    Raises:
        ValueError: 阈值[0]必须小于阈值[1]

    Returns:
        伸阶段和握阶段平台区间
    """

    if threshold[0] > threshold[1]:
        raise ValueError("threshold[0]必须小于等于threshold[1]")

    pauses_open = []
    pauses_close = []

    planified = data.copy()
    planified[(threshold[0] <= data) & (data <= threshold[1])] = 0

    num = len(data)
    for indices in zip(peaks_min, peaks_max):
        for i, idx in enumerate(indices):
            if idx < 0 or idx >= num or planified[idx] != 0.0:
                continue

            start = idx
            while start > 0 and planified[start - 1] == 0.0:
                start -= 1

            # 向后寻找连续为0的区间
            end = idx
            while end < num and planified[end + 1] == 0.0:
                end += 1

            interval = (start, end)

            if i % 2 == 0:
                pauses_open.append(interval)
            else:
                pauses_close.append(interval)

    return pauses_open, pauses_close


def diff_intervals(intervals: Sequence[Interval], scaler=None) -> np.ndarray:
    """给定存有区间的序列，对每个区间求差值。如果给定缩放系数则结果乘以该系数

    Arguments:
        intervals -- 区间列表

    Keyword Arguments:
        scaler -- 缩放系数 (default: {None})

    Returns:
        差值结果数组
    """

    if len(intervals) == 0:
        return np.array([])

    intervals = np.asarray(intervals, dtype=np.float64)
    diff = np.diff(intervals.T, axis=0)[0]
    if scaler is not None:
        diff *= scaler

    return diff


def get_asyns(query: np.ndarray, reference: np.ndarray, scaler=None, **kwargs) -> np.ndarray:
    """获取查询数据相对于参考数据的时间异步性

    Arguments:
        query -- 查询的数据数组
        reference -- 参考的数据数组
        scaler -- 缩放系数

    Returns:
        异步性结果数组
    """
    query = zscore(query)
    reference = zscore(reference)

    path = dtw.warping_path_fast(reference, query, **kwargs)

    asyns = diff_intervals(path, scaler)
    return asyns


def get_periods(peaks_min: Sequence[int], peaks_max: Sequence[int]) -> list[Interval]:
    """根据极小值和极大值序列获取周期点序列

    Arguments:
        peaks_min -- 极小值序列
        peaks_max -- 极大值序列

    Returns:
        周期点序列序列
    """
    points = peaks_min if len(peaks_min) >= len(peaks_max) else peaks_max
    return list(pairwise(points))


def minmax_period_wise(frame_arr: np.ndarray, periods: Sequence[Interval]) -> np.ndarray:
    """对每个周期内数据独立归一化

    Arguments:
        frame_arr -- 关键点帧数据数组
        periods -- 周期序列

    Returns:
        周期独立归一化后的关键点帧数据数组
    """

    result = frame_arr.copy()
    end_idx = len(periods) - 1

    for i, (start, end) in enumerate(periods):
        if i == 0:
            start = 0
        elif i == end_idx:
            end = None

        result[start:end] = minmax_over_frames(frame_arr[start:end])

    return result
