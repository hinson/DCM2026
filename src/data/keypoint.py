#
# keypoint.py
# @author Quexuan Zhang
# @description 原始关键点数据的相关数据结构定义和描述
# @created 2024-11-27T11:38:47.940Z+08:00
# @last-modified 2026-01-13T17:26:42.169Z+08:00
#

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import NamedTuple

import numpy as np
import tomllib

with (Path(__file__).parent / "config.toml").open("rb") as f:
    config = tomllib.load(f)


class Group(Enum):
    health_corr_ge20 = 0  # 0矫正大于20的正常人
    health_corr_lt20 = 10  # 10矫正小于20的正常人
    health_uncorr = 90  # 未矫正的正常人
    DCM_corr_lt20 = 1  # 1矫正小于20的DCM
    DCM_uncorr_for1000 = 11  # 11未矫正DCM（补够1000例）
    DCM_uncorr_upto500 = 21  # 21未矫正DCM（总DCM增加达到500）
    DCM_corr_ge20 = 31  # 31矫正大于20的DCM
    DCM_uncorr = 91  # 未矫正DCM
    CVA_corr_lt20 = 2  # 2矫正小于20的中风
    CVA_corr_ge20 = 12  # 12矫正大于20的中风
    CVA_uncorr = 92  # 未矫正的中风
    PD_corr_lt20 = 3  # 3矫正小于20的帕金森
    PD_corr_ge20 = 13  # 13矫正大于20的帕金森
    PD_uncorr = 93  # 未矫正的帕金森
    other_uncorr = 99


class Point(NamedTuple):
    """坐标点"""

    x: float = np.nan
    y: float = np.nan


class Keypoints(NamedTuple):
    """关键点有序集"""

    wrist: Point
    thumb_cmc: Point
    thumb_mcp: Point
    thumb_ip: Point
    thumb_tip: Point
    index_finger_mcp: Point
    index_finger_pip: Point
    index_finger_dip: Point
    index_finger_tip: Point
    middle_finger_mcp: Point
    middle_finger_pip: Point
    middle_finger_dip: Point
    middle_finger_tip: Point
    ring_finger_mcp: Point
    ring_finger_pip: Point
    ring_finger_dip: Point
    ring_finger_tip: Point
    pinky_mcp: Point
    pinky_pip: Point
    pinky_dip: Point
    pinky_tip: Point


@dataclass
class KeypointFrame:
    """关键点帧类型"""

    failed: bool  # 是否侦测失败
    keypoints: Keypoints


@dataclass
class KeypointData:
    """Json数据类型"""

    name: str  # 受试者ID
    left: bool  # 是否左手
    frame_num: int  # 帧数
    frame_rate: float  # 帧率
    width: int  # 像素宽度
    height: int  # 像素高度
    frames: list[KeypointFrame]


KeypointSeq = tuple[int]

# 指尖关键点ID
KEYPOINTS_TIPS: KeypointSeq = (4, 8, 12, 16, 20)
KEYPOINTS_WRIST: KeypointSeq = (0,)

# 计算角度相关的关键点ID
ANGLE_ENDPOINTS_A: KeypointSeq = (2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20, 17)
ANGLE_VERTICES: KeypointSeq = (1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 0)
ANGLE_ENDPOINTS_B: KeypointSeq = (0, 1, 2, 0, 5, 6, 0, 9, 10, 0, 13, 14, 0, 17, 18, 5)


MODEL_ENDPOINTS_C: KeypointSeq = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    17,
)
MODEL_ENDPOINTS_D: KeypointSeq = (
    0,
    1,
    2,
    3,
    0,
    5,
    6,
    7,
    5,
    9,
    10,
    11,
    9,
    13,
    14,
    15,
    13,
    17,
    18,
    19,
    0,
)

# 计算面积相关的关键点ID
AREA_VERTICES_PALM: KeypointSeq = (0, 17, 13, 9, 5, 0)
AREA_VERTICES_REF025: KeypointSeq = (0, 20, 8, 0)
AREA_VERTICES_REF034: KeypointSeq = (0, 16, 12, 0)


# 马氏距离相关参数
COV_MEAN = np.array(config["COV_MEAN"])
COV_INV_SQRT = np.array(config["COV_INV_SQRT"])
COV_MINMAX_MEAN = np.array(config["COV_MINMAX_MEAN"])
COV_MINMAX_INV_SQRT = np.array(config["COV_MINMAX_INV_SQRT"])

# COV_PMM_MEAN = np.array(config["COV_PMM_MEAN"])
# COV_PMM_INV_SQRT = np.array(config["COV_PMM_INV_SQRT"])

COV_PRMM_MEAN = np.array(config["COV_PRMM_MEAN"])
COV_PRMM_INV_SQRT = np.array(config["COV_PRMM_INV_SQRT"])
