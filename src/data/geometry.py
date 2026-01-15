#
# geometry.py
# @author Quexuan Zhang
# @description 几何计算有关的函数
# @created 2024-12-03T14:12:08.765Z+08:00
# @last-modified 2026-01-14T16:50:00.563Z+08:00
#

from collections.abc import Sequence

import numpy as np
from numba import njit

from .keypoint import (
    ANGLE_ENDPOINTS_A,
    ANGLE_ENDPOINTS_B,
    ANGLE_VERTICES,
    MODEL_ENDPOINTS_C,
    MODEL_ENDPOINTS_D,
    KeypointSeq,
)

__all__ = [
    # distance
    "eudist_on_arrs",
    "madist_on_arrs",
    # angle
    "get_angle_arr",
    "get_APB_arrs",
    # polygon
    "get_CD_arrs",
    "get_polygon_vertices",
    "get_polygon_areas",
    "get_polygon_centroids",
]


@njit
def eudist_on_arrs(X: np.ndarray, Y: np.ndarray):
    """计算两个多维数组对应行之间的欧几里德距离

    Arguments:
        X -- 从帧数据转换的关键点数组，最后一维为坐标
        Y -- 从帧数据转换的关键点数组，维度与`X`一致

    Raises:
        ValueError: `Y`与`X`维数不一致时

    Returns:
        距离结果
    """
    if len(X.shape) != len(Y.shape):
        raise ValueError("X和Y的维数必须一致")

    delta = X - Y
    # dist = np.linalg.norm(delta, axis=-1)     # 慢1倍

    F, K = delta.shape[:2]
    dist = np.zeros((F, K))

    for i in range(F):
        for j in range(K):
            x, y = delta[i, j]
            dist[i, j] = (x**2 + y**2) ** 0.5

    return dist


@njit
def madist_on_arrs(X: np.ndarray, Y: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
    r"""计算两个多维数组之间的马氏距离

    Arguments:
        X -- :math:`N \times K \times d`参考数组，:math:`N`是样本数，:math:`K`是关键点类别数， :math:`d`是特征维数
        Y -- :math:`M \times K \times d`比较数组，维度和`X`一致
        inv_cov -- 特征协方差的逆

    Raises:
        ValueError: `X`和`Y`维数不一致时

    Returns:
        马氏距离结果
    """

    # 验证维度匹配
    if X.shape[-1] != Y.shape[-1]:
        raise ValueError("X和Y的特征维度d必须一致")

    ic_xx = inv_cov[0, 0]
    ic_xy = inv_cov[0, 1]
    ic_yy = inv_cov[1, 1]

    delta = X - Y
    #  dist = np.sqrt(np.einsum("fki,ij,fkj->fk", delta, inv_cov, delta)) # 慢4倍

    F, K = delta.shape[:2]
    dist = np.zeros((F, K))
    for i in range(F):
        for j in range(K):
            x, y = delta[i, j]
            dist[i, j] = (x**2 * ic_xx + 2 * x * y * ic_xy + y**2 * ic_yy) ** 0.5

    return dist


def get_APB_arrs(frame_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""根据关键点帧数据返回以`P`为角度顶点`A``B`为两边端点的角度的坐标数组

    Arguments:
        frame_arr -- :math:`N \times K \times 2`关键点帧数据数组

    Returns:
        `(A, P, B)`角度三点坐标数组，数组大小为:math:`N \times R \times 2`，:math:`R`为作为顶点的关键点数
    """

    A = frame_arr[:, ANGLE_ENDPOINTS_A, :]
    P = frame_arr[:, ANGLE_VERTICES, :]
    B = frame_arr[:, ANGLE_ENDPOINTS_B, :]

    return A, P, B


@njit
def get_angle_arr(A: np.ndarray, P: np.ndarray, B: np.ndarray, left: bool) -> np.ndarray:
    """获取关键节点角度

    Arguments:
        A -- 角度一边端点坐标数组，大小为:math:`N \times R \times 2`
        P -- 角度顶点坐标，大小与`A`一致
        B -- 角度另一边端点坐标，大小与`A`一致
        left -- 是否左手，用于处理手性

    Returns:
        关键节点角度的弧度值
    """
    if A.shape != P.shape != B.shape:
        raise ValueError(
            "输入坐标数组大小不一致: A.shape =",
            str(A.shape),
            "P.shape =",
            str(P.shape),
            "B.shape =",
            str(B.shape),
        )

    # # 两边向量
    # AP = A - P
    # BP = B - P

    # # 向量夹角公式
    # dot_product = np.einsum("nmd,nmd->nm", AP, BP)
    # norm1 = np.linalg.norm(AP, axis=-1)
    # norm2 = np.linalg.norm(BP, axis=-1)
    # cos_theta = dot_product / norm1 / norm2
    # cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 防止精度问题导致数值越界
    # angle_rad = np.arccos(cos_theta)

    # cross_product = np.cross(AP, BP)  # 叉积检验角度是否超过180度
    # cond = cross_product > 0 if left else cross_product < 0  # 根据手性变换判断条件

    # angle_rad = np.where(cond, 2 * np.pi - angle_rad, angle_rad)      # 慢2倍

    # 获取形状信息
    N = A.shape[0]
    R = A.shape[1]

    angle_rad = np.zeros((N, R))

    for i in range(N):
        for j in range(R):
            # 计算向量AP和BP
            ax, ay = A[i, j]
            px, py = P[i, j]
            bx, by = B[i, j]

            APx = ax - px
            APy = ay - py

            BPx = bx - px
            BPy = by - py

            # 计算点积和模长
            dot_product = APx * BPx + APy * BPy
            norm1 = (APx**2 + APy**2) ** 0.5 + 1e-8  # 防止0除
            norm2 = (BPx**2 + BPy**2) ** 0.5 + 1e-8  # 防止0除

            cos_theta = dot_product / (norm1 * norm2)

            # 处理因精度问题溢出
            if cos_theta > 1:
                cos_theta = 1
            elif cos_theta < -1:
                cos_theta = -1

            angle_rad[i, j] = np.arccos(cos_theta)

    # 处理手性
    cross_product = np.zeros((N, R))
    for i in range(N):
        for j in range(R):
            cross_product[i, j] = A[i, j, 0] * B[i, j, 1] - A[i, j, 1] * B[i, j, 0]

    for i in range(N):
        for j in range(R):
            cp = cross_product[i, j]
            if (left and cp > 0) or (not left and cp < 0):
                angle_rad[i, j] = 2 * np.pi - angle_rad[i, j]

    return angle_rad


def get_CD_arrs(frame_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""根据关键点帧数据返回以`C``D`为手掌模型线段端点的数组

    Arguments:
        frame_arr -- :math:`N \times K \times 2`关键点帧数据数组

    Returns:
        `(C, D)`手掌骨架线段端点数组
    """

    C = frame_arr[:, MODEL_ENDPOINTS_C, :]
    D = frame_arr[:, MODEL_ENDPOINTS_D, :]

    return C, D


def get_polygon_vertices(
    frame_arr: np.ndarray, area_keypoints: KeypointSeq, left: bool
) -> np.ndarray:
    """根据给定的多边形关键点ID返回顶点数组，顶点头尾必须闭合且以右手顺时针顺序排列

    Arguments:
        frame_arr -- :math:`N \times K \times 2`关键点帧数据数组
        area_keypoints -- 作为多边形顶点的关键点ID列表
        left -- 是否左手

    Raises:
        ValueError -- 当多边形不闭合时

    Returns:
        多边形顶点坐标数组

    """
    if area_keypoints[0] != area_keypoints[-1]:
        raise ValueError(f"多边形必须闭合：({area_keypoints[0]=}) != ({area_keypoints[-1]=})")

    keypoints_v = tuple(reversed(area_keypoints)) if left else area_keypoints

    V = frame_arr[:, keypoints_v, :]
    return V


@njit
def get_polygon_areas(vertices: np.ndarray) -> np.ndarray:
    r"""根据顶点计算每帧的多边形面积

    Arguments:
        vertices -- :math:`N \times V \times 2`帧多边形顶点数组

    Returns:
        :math:`N`个元素的帧面积数组
    """

    #     x1 = vertices[:, :-1, 0]
    #     y1 = vertices[:, :-1, 1]
    #     x2 = vertices[:, 1:, 0]
    #     y2 = vertices[:, 1:, 1]
    #     return np.abs(np.sum(x1 * y2 - y1 * x2, axis=-1)) / 2.0

    F, N = vertices.shape[:2]
    result = np.zeros(F)
    for i in range(F):
        area = 0.0
        for j in range(N - 1):
            x1, y1 = vertices[i, j]
            x2, y2 = vertices[i, j + 1]
            area += x1 * y2 - y1 * x2
        result[i] = np.abs(area) / 2.0

    return result


@njit
def get_polygon_centroids(vertices: np.ndarray, areas: np.ndarray) -> np.ndarray:
    r"""计算各帧多边形的质心位置

    Arguments:
        vertices -- :math:`N \times V \times 2`帧多边形顶点数组
        areas -- :math:`N`帧多边形面积数组

    Returns:
        :math:`N \times 2`帧多边形质心坐标数组
    """

    #     x1 = vertices[:, :-1, 0]
    #     y1 = vertices[:, :-1, 1]
    #     x2 = vertices[:, 1:, 0]
    #     y2 = vertices[:, 1:, 1]

    #     mean_area = (x1 * y2 - y1 * x2) / 6.0 / areas[:, np.newaxis]

    #     cx = np.sum((x1 + x2) * mean_area, axis=-1)
    #     cy = np.sum((y1 + y2) * mean_area, axis=-1)

    #     return np.stack([cx, cy], axis=1)

    F, N = vertices.shape[:2]
    result = np.zeros((F, 2))
    for i in range(F):
        cx = cy = 0.0
        for j in range(N - 1):
            x1, y1 = vertices[i, j]
            x2, y2 = vertices[i, j + 1]
            cx += (x1 + x2) * (x1 * y2 - y1 * x2)
            cy += (y1 + y2) * (x1 * y2 - y1 * x2)

        cx /= 6.0 * areas[i]
        cy /= 6.0 * areas[i]
        result[i] = cx, cy

    return result
