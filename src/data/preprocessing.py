#
# preprocessing.py
# @author Quexuan Zhang
# @description 数据预处理相关函数
# @created 2024-11-28T12:44:09.764Z+08:00
# @last-modified 2025-08-25T20:27:12.708Z+08:00
#
from copy import copy
from typing import IO, Any

import numpy as np
import orjson
from numba import njit

from .keypoint import *
from .utils import array_frames

# 无用的原始数据键值
unused_keys = ("original_index", "is_detected", "has_face", "is_complex", "complex_type")
nan_point = Point(np.nan, np.nan)


def delete_unused_keys(json: dict[str, Any]) -> None:
    """删除无用的键值

    Parameters
    ----------
    json : dict[str, Any]
        原始数据Json格式字典
    """
    for key in unused_keys:
        if key in json:
            del json[key]


# 侦测失败点作缺失数据
failed_sample = Keypoints(*([nan_point] * len(Keypoints._fields)))


def check_left_hand(json: dict[str, Any]) -> None:
    """检查是否左手，对字典新增`left_hand`字段

    Arguments:
        json -- 原始数据Json格式字典
    """
    json["left"] = json["name"][-1] == "L"


def check_failed_frames(json: dict[str, Any]) -> None:
    """检查侦测失败情况并重新组织帧数据结构，重新组织`label`字段为`frames`字段，并删除`label`、`fail_num`和`fail_frames`字段（如有）。`frames`字段中每个元素为一个包含`failed`和`keypoints`字段的字典，对应`KeypointFrame`类

    Arguments:
        json -- 原始数据Json格式字典
    """

    frames = json["label"]
    failed_indices = set()

    if "fail_num" in json:
        if json["fail_num"] > 0:
            failed_indices = set(json["fail_frames"])

        del json["fail_num"]
        del json["fail_frames"]

    for i, frame in enumerate(frames):
        if i + 1 in failed_indices:
            frames[i] = dict(failed=True, keypoints=failed_sample)  # 对失败帧赋缺失值
            continue

        keypoints = (Point(x, y) for x, y in zip(frame[1::2], frame[2::2]))  # 抽取每对(x,y)坐标
        frames[i] = dict(failed=False, keypoints=Keypoints(*keypoints))


def get_json_data(f: IO) -> KeypointData:
    """根据Json路径获取关键点数据

    Arguments:
        path -- Json文件路径

    Raises:
        TypeError: 解析数据结构类型错误时

    Returns:
        对应的关键点数据
    """

    json = orjson.loads(f.read())

    check_left_hand(json)
    delete_unused_keys(json)
    check_failed_frames(json)

    frames = [KeypointFrame(**frame) for frame in json["label"]]
    del json["label"]

    return KeypointData(frames=frames, **json)


def set_outliers_na(data: KeypointData) -> KeypointData:
    """对非法值设置为缺失值

    Parameters
    ----------
    data : KeypointData
        原始关键点数据

    Returns
    -------
    KeypointData
        修改后的关键点数据
    """
    data = copy(data)

    new_frames = []
    for frame in data.frames:
        if frame.failed:
            continue

        new_keypoints = []
        for point in frame.keypoints:
            x, y = point

            if x == y == 0:  # (0, 0)
                x = np.nan
                y = np.nan

            # else:  # 坐标值超出[0, 1]
            #     if x < 0 or x > 1:
            #         x = np.nan
            #     if y < 0 or y > 1:
            #         y = np.nan

            # if x < 0:
            #     x = 0
            # elif x > 1:
            #     x = 1

            # if y < 0:
            #     y = 0
            # elif y > 1:
            #     y = 1

            new_keypoints.append(Point(x, y))

        new_frames.append(KeypointFrame(failed=False, keypoints=new_keypoints))

    data.frames = new_frames
    return data


def interpolate_missing_frames(data: KeypointData, interval=10.0) -> np.ndarray:
    r"""对缺失帧进行线性插值。如果数据头部或尾部出现缺失值则丢弃该帧

    Arguments:
        data -- 关键点数据

    Returns:
        :math:`N \times K \times 2`帧数据数组，N为帧数，K为关键点数
    """

    num_keypoints = len(Keypoints._fields)

    sample = array_frames(data, interval=interval)  # 单个样本的所有帧数据
    mask = ~np.isnan(sample).all(axis=(1, 2))  # 标记帧是否完全缺失
    valid_indices = mask.nonzero()[0]  # 有效帧索引

    # 如果头部或尾部帧缺失，丢弃这些帧
    start, end = valid_indices[0], valid_indices[-1]
    sample = sample[start : end + 1]
    mask = mask[start : end + 1]

    # 对每个关键点的 (x, y) 坐标进行线性插值
    for k in range(num_keypoints):
        for c in range(2):
            keypoint_data = sample[:, k, c]
            nan_mask = np.isnan(keypoint_data)
            if nan_mask.all():
                keypoint_data = 1.0

            elif nan_mask.any():
                # 插值处理
                valid = ~nan_mask
                indices = np.arange(len(keypoint_data))
                keypoint_data[nan_mask] = np.interp(
                    indices[nan_mask], indices[valid], keypoint_data[valid]
                )
            sample[:, k, c] = keypoint_data

    return sample


def minmax_over_frames(frame_arr: np.ndarray) -> np.ndarray:
    r"""利用所有帧中最大和最小的坐标值归一化每帧的关键点数据

    Arguments:
        frame_arr -- :math:`N \times K \times 2`关键点数据数组

    Returns:
        :math:`N \times K \times 2`归一化后的关键点数据数组
    """

    min_value = np.nanmin(frame_arr, axis=(0, 1))[np.newaxis, np.newaxis, :]
    max_value = np.nanmax(frame_arr, axis=(0, 1))[np.newaxis, np.newaxis, :]
    delta = max_value - min_value
    new_arr = (frame_arr.copy() - min_value) / delta

    return new_arr


@njit
def whiten_frame_arr(
    frame_arr: np.ndarray, mean: np.ndarray, sqrt_inv_cov: np.ndarray
) -> np.ndarray:
    r"""对坐标值进行马氏距离变换，即白化变换

    Arguments:
        frame_arr -- :math:`N \times K \times 2`关键点数据数组
        mean -- 大小为:math:`2`的均值数组
        sqrt_inv_cov -- 大小为:math:`2 \times 2`的协方差矩阵的逆的平方根

    Returns:
        :math:`N \times K \times 2`白化后的关键点数据数组
    """

    m_x, m_y = mean
    sic_xx = sqrt_inv_cov[0, 0]
    sic_xy = sqrt_inv_cov[0, 1]
    sic_yy = sqrt_inv_cov[1, 1]

    shape = frame_arr.shape
    f, k = shape[:2]
    whitened = np.zeros(shape)

    for i in range(f):
        for j in range(k):
            x, y = frame_arr[i, j]
            x -= m_x
            y -= m_y
            whitened[i, j, 0] = x * sic_xx + y * sic_xy
            whitened[i, j, 1] = x * sic_xy + y * sic_yy

    return whitened


@njit
def resample_frame_arr(
    data: np.ndarray,
    frame_rate: float,
    target_frame_rate=20.0,
) -> np.ndarray:
    F, K, P = data.shape
    downsampling_factor = frame_rate / target_frame_rate

    # 计算降采样后的帧数
    resampled_frame_count = int(F / downsampling_factor)

    # 使用线性插值进行降采样
    resampled_data = np.zeros((resampled_frame_count, K, P))

    for k in range(K):
        for p in range(P):
            # 提取每个点和每个坐标的数据（F,）
            point_data = data[:, k, p]

            # 创建线性插值函数
            # interp_func = sp.interpolate.interp1d(np.arange(F), point_data, interp_kind)
            # 生成目标帧数的新数据
            new_x = np.linspace(0, F - 1, resampled_frame_count)
            # resampled_data[:, k, p] = interp_func(new_x)

            resampled_data[:, k, p] = np.interp(new_x, np.arange(F), point_data)

    return resampled_data
