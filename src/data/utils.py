#
# utils.py
# @author Quexuan Zhang
# @description 未分类的工具函数
# @created 2024-11-27T11:51:36.127Z+08:00
# @last-modified 2025-01-14T14:59:02.237Z+08:00
#

from pathlib import Path
from typing import Literal, Sequence

import joblib
import numpy as np
import pandas as pd

from .keypoint import KeypointData, KeypointFrame, Keypoints, Point


def get_points_by_ids(keypoints: Keypoints, ids_to_get: list[int]) -> list[Point]:
    """根据关键点ID获取但帧对应的坐标列表

    Arguments:
        keypoints -- 单帧关键点集
        ids_to_get -- 需要获取的关键点ID列表

    Returns:
        对应的关键点坐标列表
    """

    return [keypoints[id] for id in ids_to_get]


def array_keypoints(
    frames: Sequence[KeypointFrame],
    *,
    frame_rate: float = 24.0,
    ids_to_get=tuple(range(len(Keypoints._fields))),
    interval: float = 10.0,
) -> np.ndarray:
    r"""根据给定的关键点类型列表、帧率和时长把帧数据转换成数组。根据帧率和时长换算帧数窗口，然后从全部帧的中间获取。

    Arguments:
        frames -- 帧数据

    Keyword Arguments:
        ids_to_get -- 关键点类型列表，默认为`Keypoints`类型的字段数 (default: {tuple(range(len(Keypoints._field_defaults))})
        frame_rate -- 帧率
        interval -- 时长


    Returns:
        :math:`N \times K \times 2`数组，其中:math:`N`是帧数，
        :`K`是给定的关键点类型数，:math:`2`是坐标轴数
    """

    num_frames = len(frames)
    mid = num_frames // 2
    offset = int(interval * frame_rate) // 2

    start = mid - offset
    if start < 0:
        start = 0

    end = mid + offset + 1
    if end > num_frames - 1:
        end = num_frames - 1

    arr = [get_points_by_ids(frame.keypoints, ids_to_get) for frame in frames[start:end]]
    return np.asarray(arr)


def array_frames(
    data: KeypointData, *, ids_to_get=tuple(range(len(Keypoints._fields))), interval: float = 10.0
) -> np.ndarray:
    r"""根据给定的关键点数据返回帧数据数组

    Arguments:
        frames -- 帧数据列表

    Keyword Arguments:
        ids_to_get -- 关键点类型列表，默认为`Keypoints`类型的字段数 (default: {tuple(range(len(Keypoints._field_defaults))})
        interval -- 时长

    Returns:
        :math:`N \times K \times 2`数组，其中:math:`N`是帧数，:math:`K`是给定的关键点类型数，:math:`2`是坐标轴数
    """

    return array_keypoints(
        data.frames, frame_rate=data.frame_rate, ids_to_get=ids_to_get, interval=interval
    )


def impute_data(X: pd.DataFrame, type: Literal["all", "scale"] = "all"):

    X_median = joblib.load(Path(__file__).parent / f"X_median_{type}.pkl")
    X.fillna(X_median, inplace=True)
    return X
