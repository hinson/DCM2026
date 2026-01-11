#
# visualize.py
# @author Quexuan Zhang
# @description
# @created 2024-12-13T19:11:46.547Z+08:00
# @last-modified 2025-12-11T10:53:53.719Z+08:00
#

from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Polygon
from matplotlib.text import Text

from data.geometry import (
    eudist_on_arrs,
    get_angle_arr,
    get_APB_arrs,
    get_CD_arrs,
    get_polygon_areas,
    get_polygon_centroids,
    get_polygon_vertices,
)
from data.timeseries import find_period_peaks, get_period_intervals
from feature import AREA_VERTICES_REF025, KeypointSeq
from transform import whiten_only


def _save_ani(
    ani,
    save_path: Optional[Path | str] = None,
    save_fps: Optional[int] = None,
    save_bitrate: int = 1800,
    save_dpi: int = 180,
):
    if save_path is not None:
        if isinstance(save_path, str):
            save_path = Path(save_path)

        match save_path.suffix:
            case ".apng" | ".gif":

                writer = PillowWriter(fps=save_fps, bitrate=save_bitrate)
            case ".mp4":
                writer = FFMpegWriter(fps=save_fps, bitrate=save_bitrate)
            case _:
                raise ValueError(f"Unkwon format: {save_path.suffix}")

        ani.save(save_path, writer=writer, dpi=save_dpi)


def animate_keypoint_angles(
    frame_arr: np.ndarray,
    left: bool,
    title="angles",
    figsize: tuple[float, float] = (5, 5),
    interval: Optional[int] = 1,
    save_path: Optional[Path | str] = None,
    save_fps: Optional[int] = None,
    save_bitrate: int = 1800,
    save_dpi: int = 180,
    fig: plt.Figure = None,
) -> FuncAnimation:
    r"""对关键点帧数据进行动画可视化

    Arguments:
        frames -- :math:`N \times K \times 2`帧数据数组，其中:math:`N`为帧数，:math:`K`为关键点数，:math:`2`为坐标轴数

    Keyword Arguments:
        title -- 动画标题 (default: {"keypoints"})
        figsize -- 动画尺寸 (default: {(5, 5)})
        interval -- 每帧延迟毫秒数 (default: {30})
        save_path -- 动画保存路径，为`None`时不保存 (default: {None})
        save_fps -- 保存动画帧率 (default: {15})
        save_bitrate -- 保存动画码率 (default: {1800})

    Returns:
        动画对象
    """
    if fig is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = fig.gca()

    frame_arr = frame_arr.copy()
    frame_arr[:, :, 1] = -frame_arr[:, :, 1]
    A, P, B = get_APB_arrs(frame_arr)
    AP = A - P
    BP = B - P

    min_x, min_y = frame_arr.min(axis=(0, 1))
    max_x, max_y = frame_arr.max(axis=(0, 1))

    angle_deg = np.degrees(get_angle_arr(A, P, B, left))

    ax.set_title(title)
    ax.set_xlim((min_x, max_x))
    ax.set_ylim((min_y, max_y))

    lines: list[Line2D] = []
    arcs: list[Arc] = []
    texts: list[Text] = []

    vertices = P[0]
    endpoints_a = A[0]
    endpoints_b = B[0]
    vectors_a = AP[0]
    vectors_b = BP[0]

    for i in range(AP.shape[1]):

        line = ax.plot(
            [endpoints_a[i, 0], vertices[i, 0], endpoints_b[i, 0]],
            [endpoints_a[i, 1], vertices[i, 1], endpoints_b[i, 1]],
            "-",
            c="dimgray",
            lw=1,
        )[0]
        lines.append(line)

        arc_radius = 0.05 * max_y
        angle_start = np.degrees(np.arctan2(vectors_a[i, 1], vectors_a[i, 0]))
        angle_end = np.degrees(np.arctan2(vectors_b[i, 1], vectors_b[i, 0]))

        if left:  # 左手手性
            angle_start, angle_end = angle_end, angle_start

        arc = Arc(
            vertices[i],
            2 * arc_radius,
            2 * arc_radius,
            theta1=angle_start,
            theta2=angle_end,
            color="darkgreen",
            lw=0.5,
        )
        ax.add_patch(arc)
        arcs.append(arc)

        offset = arc_radius * np.array([0.5, 0.5])
        ha = "left"
        if left:
            offset[0] = -offset[0]
            ha = "right"

        angle_label_pos = vertices[i] + offset
        text = ax.text(
            angle_label_pos[0],
            angle_label_pos[1],
            f"{angle_deg[0,i]:.2f}°",
            color="darkgreen",
            ha=ha,
        )
        texts.append(text)

    def update(t):
        vertices = P[t]
        endpoints_a = A[t]
        endpoints_b = B[t]
        vectors_a = AP[t]
        vectors_b = BP[t]

        for i, (line, arc, text) in enumerate(zip(lines, arcs, texts)):

            # 更新线段
            line.set_data(
                [endpoints_a[i, 0], vertices[i, 0], endpoints_b[i, 0]],
                [endpoints_a[i, 1], vertices[i, 1], endpoints_b[i, 1]],
            )

            # 更新角弧
            arc_radius = 0.05 * max_y
            angle_start = np.degrees(np.arctan2(vectors_a[i, 1], vectors_a[i, 0]))
            angle_end = np.degrees(np.arctan2(vectors_b[i, 1], vectors_b[i, 0]))

            if left:  # 左手手性
                angle_start, angle_end = angle_end, angle_start

            arc.center = vertices[i]
            arc.width = 2 * arc_radius
            arc.height = 2 * arc_radius
            arc.theta1 = angle_start
            arc.theta2 = angle_end

            # 更新度数
            offset = arc_radius * np.array([0.5, 0.5])
            if left:
                offset[0] = -offset[0]
            angle_label_pos = vertices[i] + offset

            text.set_position(angle_label_pos)
            text.set_text(f"{angle_deg[t,i]:.2f}°")

        return lines + arcs + texts

    num_frames = frame_arr.shape[0]
    ani = FuncAnimation(fig=fig, func=update, frames=num_frames, interval=interval, blit=True)
    plt.close()

    if save_fps is None:
        save_fps = num_frames / 10
    _save_ani(ani, save_path, save_fps, save_bitrate, save_dpi)

    return ani


def animate_palm_areas(
    frame_arr: np.ndarray,
    area_keypoints: Sequence[int],
    left: bool,
    title="areas",
    figsize: tuple[float, float] = (5, 5),
    interval: Optional[int] = 1,
    save_path: Optional[Path | str] = None,
    save_fps: Optional[int] = None,
    save_bitrate: int = 1800,
    save_dpi: int = 180,
    fig: plt.Figure = None,
) -> FuncAnimation:
    r"""对关键点帧数据进行动画可视化

    Arguments:
        frames -- :math:`N \times K \times 2`帧数据数组，其中:math:`N`为帧数，:math:`K`为关键点数，:math:`2`为坐标轴数

    Keyword Arguments:
        title -- 动画标题 (default: {"keypoints"})
        figsize -- 动画尺寸 (default: {(5, 5)})
        interval -- 每帧延迟毫秒数 (default: {30})
        save_path -- 动画保存路径，为`None`时不保存 (default: {None})
        save_fps -- 保存动画帧率 (default: {15})
        save_bitrate -- 保存动画码率 (default: {1800})

    Returns:
        动画对象
    """
    if fig is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = fig.gca()

    frame_arr = frame_arr.copy()
    frame_arr[:, :, 1] = -frame_arr[:, :, 1]

    C, D = get_CD_arrs(frame_arr)
    V = get_polygon_vertices(frame_arr, area_keypoints, left)
    areas = get_polygon_areas(V)
    centroids = get_polygon_centroids(V, areas)

    min_x, min_y = frame_arr.min(axis=(0, 1))
    max_x, max_y = frame_arr.max(axis=(0, 1))

    ax.set_title(title)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    endpoints_c = C[0]
    endpoints_d = D[0]
    lines: list[Line2D] = []
    for i in range(C.shape[1]):
        line = ax.plot(
            [endpoints_c[i, 0], endpoints_d[i, 0]],
            [endpoints_c[i, 1], endpoints_d[i, 1]],
            "-",
            c="dimgray",
            lw=1,
        )[0]
        lines.append(line)

    polygon = Polygon(V[0], closed=True, facecolor="red", edgecolor="darkred", lw=2, alpha=0.35)
    ax.add_patch(polygon)

    text: Text = ax.text(*centroids[0], f"{areas[0]:.4f}", c="k", ha="center", va="center")

    def update(t):
        endpoints_c = C[t]
        endpoints_d = D[t]

        # 更新线段
        for i, line in enumerate(lines):
            line.set_data(
                [endpoints_c[i, 0], endpoints_d[i, 0]], [endpoints_c[i, 1], endpoints_d[i, 1]]
            )

        # 更新多边形
        polygon.xy = V[t]

        # 更新面积
        text.set_position(centroids[t])
        text.set_text(f"{areas[t]:.4f}")

        return lines + [polygon, text]

    num_frames = frame_arr.shape[0]
    ani = FuncAnimation(fig=fig, func=update, frames=num_frames, interval=interval, blit=True)
    plt.close()

    if save_fps is None:
        save_fps = num_frames / 10
    _save_ani(ani, save_path, save_fps, save_bitrate, save_dpi)

    return ani


def vspan(ax, xmin, xmax, span_limit_factor=1e5, **kwargs):
    """From https://github.com/mpld3/mpld3/issues/298

    Arguments:
        ax -- _description_
        xmin -- _description_
        xmax -- _description_

    Keyword Arguments:
        span_limit_factor -- _description_ (default: {1e5})
    """
    yrange = plt.ylim()
    ax.fill_between(
        [xmin, xmax],
        [
            (np.abs(yrange[0]) + 1) * -span_limit_factor,
            (np.abs(yrange[0]) + 1) * -span_limit_factor,
        ],
        [(np.abs(yrange[1]) + 1) * span_limit_factor, (np.abs(yrange[1]) + 1) * span_limit_factor],
        **kwargs,
    )
    plt.ylim(*yrange)


def plot_dist_changes(
    frame_arr: np.ndarray,
    endpoints: KeypointSeq,
    origins: KeypointSeq,
    labels: list[str],
    area_keypoints: KeypointSeq = AREA_VERTICES_REF025,
    left=False,
    title="Distance Changes",
    figsize=(7, 3),
    fig: Optional[Figure] = None,
) -> Figure:
    # frame_arr = whiten_only(frame_arr)
    vertics = get_polygon_vertices(frame_arr, area_keypoints, left=left)
    areas = get_polygon_areas(vertics)

    if fig is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = fig.gca()

    dist_arr = eudist_on_arrs(frame_arr[:, endpoints], frame_arr[:, origins])
    ax.set_prop_cycle(color=plt.cm.Dark2(np.linspace(0, 1, 8)))
    ax.plot(dist_arr, label=labels, lw=1)

    #
    peaks_min, peaks_max = find_period_peaks(areas, prominence=0.25)
    # plt.plot(peaks_min, areas[peaks_min], "g^", label="平滑后极小值点")
    # plt.plot(peaks_max, areas[peaks_max], "rv", label="平滑后极大值点")

    periods_open, periods_close = get_period_intervals(peaks_min, peaks_max)

    for start, end in periods_open:
        # ax.axvspan(start, end, color="red", alpha=0.1, label="open")
        vspan(ax, start, end, color="red", alpha=0.35, label="open")
    for start, end in periods_close:
        vspan(ax, start, end, color="deepskyblue", alpha=0.35, label="close")
        # ax.axvspan(start, end, color="deepskyblue", alpha=0.1, label="close")

    # 设置图例
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = list(dict(zip(labels, handles)).items())  # 去除重复的图例
    ax.legend(
        [h for _, h in unique_labels],
        [l for l, _ in unique_labels],
        ncol=len(unique_labels),
        loc="lower center",
        frameon=True,
        framealpha=0.5,
        columnspacing=1,
        labelspacing=0,
    )

    ax.set_title(title)
    plt.close()

    return fig


def plot_area_changes(
    frame_arr: np.ndarray,
    area_keypoints: KeypointSeq,
    left=False,
    title="Area Changes",
    figsize=(7, 3),
) -> matplotlib.figure.Figure:
    vertics = get_polygon_vertices(frame_arr, area_keypoints, left=left)
    areas = get_polygon_areas(vertics)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_prop_cycle(color=plt.cm.Dark2(np.linspace(0, 1, 8)))
    ax.plot(areas, c="k", lw=1)

    #
    peaks_min, peaks_max = find_period_peaks(areas, prominence=1)
    # plt.plot(peaks_min, areas[peaks_min], "g^", label="平滑后极小值点")
    # plt.plot(peaks_max, areas[peaks_max], "rv", label="平滑后极大值点")

    periods_open, periods_close = get_period_intervals(peaks_min, peaks_max)

    for start, end in periods_open:
        # ax.axvspan(start, end, color="red", alpha=0.3, label="open")
        vspan(ax, start, end, color="red", alpha=0.3, label="open")
    for start, end in periods_close:
        # ax.axvspan(start, end, color="deepskyblue", alpha=0.3, label="close")
        vspan(ax, start, end, color="deepskyblue", alpha=0.3, label="close")

    # 设置图例
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = list(dict(zip(labels, handles)).items())  # 去除重复的图例
    ax.legend(
        [h for _, h in unique_labels], [l for l, _ in unique_labels], frameon=True, framealpha=0.5
    )

    ax.set_title(title)
    plt.close()

    return fig
