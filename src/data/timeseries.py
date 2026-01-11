#
# timeseries.py
# @author Quexuan Zhang
# @description
# @created 2024-12-04T19:37:43.491Z+08:00
# @last-modified 2025-12-10T15:52:48.651Z+08:00
#

from collections.abc import Sequence
from itertools import chain, pairwise

import numpy as np
from dtaidistance import dtw
from numba import njit
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import zscore

from .geometry import get_polygon_areas, get_polygon_vertices
from .keypoint import KeypointSeq
from .preprocessing import minmax_over_frames

Interval = tuple[int, int]


def adaptive_gaussian_filter(data: np.ndarray, base_sigma: float = 1.0) -> np.ndarray:
    """根据局部变化幅度动态调整标准差的自适应高斯滤波平滑器

    Arguments:
        data -- 一维数组

    Keyword Arguments:
        base_sigma -- 初始标准差 (default: {1.})

    Returns:
        平滑后数据
    """
    #
    variations = np.abs(np.diff(data, prepend=data[0]))
    sigma_values = base_sigma * (1 + variations / np.max(variations))
    filtered = [gaussian_filter1d(data, sigma=s) for s in sigma_values]
    smoothed_data = np.asarray(filtered).mean(axis=0)
    return smoothed_data


def find_period_peaks(
    data: np.ndarray, base_sigma=1, prominence=0.4, **kwargs
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
    smoothed = adaptive_gaussian_filter(data, base_sigma=base_sigma)
    peaks_max, _ = find_peaks(smoothed, prominence=prominence, **kwargs)  # 极大值
    peaks_min, _ = find_peaks(-smoothed, prominence=prominence, **kwargs)  # 极小值

    return (peaks_min, peaks_max)


def find_period_peaks_by_area(
    frame_arr: np.ndarray,
    vertice_ids: KeypointSeq,
    left: bool,
    base_sigma=1,
    prominence=0.4,
    **kwargs
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
    return find_period_peaks(areas, base_sigma=base_sigma, prominence=prominence, **kwargs)


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
        zipped = zip(peaks_min, peaks_max)
        start = 0
    else:
        zipped = zip(peaks_max, peaks_min)
        start = 1

    for i, pair in enumerate(pairwise(chain.from_iterable(zipped)), start):
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
        frame_arr, vertice_ids, left, base_sigma, prominence
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
