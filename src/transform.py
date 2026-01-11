#
# transform.py
# @author Quexuan Zhang
# @description
# @created 2024-12-04T20:15:24.748Z+08:00
# @last-modified 2025-03-20T17:13:07.106Z+08:00
#


import functools
from collections.abc import Callable

import pandas as pd

from data.keypoint import *
from data.preprocessing import (
    interpolate_missing_frames,
    minmax_over_frames,
    set_outliers_na,
    whiten_frame_arr,
)
from data.timeseries import find_period_peaks_by_area, get_periods, minmax_period_wise
from data.utils import array_frames


def func_df(func: Callable, source_col: str, target_col=None):

    if target_col is None:
        target_col = source_col

    @functools.wraps(func)
    def new_func(df: pd.DataFrame):
        df[target_col] = df[source_col].map(lambda x: func(x))
        return df

    return new_func


# def whiten(frame_arr: np.ndarray) -> np.ndarray:
#     return whiten_frame_arr(frame_arr, COV_MEAN, COV_INV_SQRT)


def whiten_minmax(frame_arr: np.ndarray) -> np.ndarray:
    return whiten_frame_arr(frame_arr, COV_MINMAX_MEAN, COV_MINMAX_INV_SQRT)


# def whiten_per_minmax(frame_arr: np.ndarray) -> np.ndarray:
#     return whiten_frame_arr(frame_arr, COV_PMM_MEAN, COV_PMM_INV_SQRT)


def whiten_per_reminmax(frame_arr: np.ndarray) -> np.ndarray:
    return whiten_frame_arr(frame_arr, COV_PRMM_MEAN, COV_PRMM_INV_SQRT)


def whiten_only(data: KeypointData) -> np.ndarray:
    frame_arr = interpolate_missing_frames(set_outliers_na(data))
    frame_arr = whiten_frame_arr(frame_arr, COV_MEAN, COV_INV_SQRT)
    return frame_arr


def minmax_whiten(data: KeypointData) -> np.ndarray:
    frame_arr = interpolate_missing_frames(set_outliers_na(data))
    frame_arr = minmax_over_frames(frame_arr)
    frame_arr = whiten_frame_arr(frame_arr, COV_MINMAX_MEAN, COV_MINMAX_INV_SQRT)
    return frame_arr


# def per_minmax(data: KeypointData) -> np.ndarray:

#     frame_arr = whiten_only(data)

#     peaks_min, peaks_max = find_period_peaks_by_area(
#         frame_arr, AREA_VERTICES_REF025, left=data.left
#     )
#     periods = get_periods(peaks_min, peaks_max)

#     rescaled = minmax_period_wise(frame_arr, periods)
#     whitened = whiten_frame_arr(rescaled, COV_PMM_MEAN, COV_PMM_INV_SQRT)
#     return whitened


def per_reminmax(data: KeypointData) -> np.ndarray:

    frame_arr = minmax_whiten(data)

    peaks_min, peaks_max = find_period_peaks_by_area(
        frame_arr, AREA_VERTICES_REF025, left=data.left
    )
    periods = get_periods(peaks_min, peaks_max)

    rescaled = minmax_period_wise(frame_arr, periods)
    whitened = whiten_frame_arr(rescaled, COV_PRMM_MEAN, COV_PRMM_INV_SQRT)
    return whitened


# FunctionTransformer用函数
transform_d2d_outlier = func_df(set_outliers_na, "keypoint_data")

transform_d2a_array = func_df(array_frames, "keypoint_data", "frame_arr")
transform_d2a_iterpolation = func_df(interpolate_missing_frames, "keypoint_data", "frame_arr")

transform_a2a_minmax = func_df(minmax_over_frames, "frame_arr")
# transform_a2a_whiten = func_df(whiten, "frame_arr")
transform_a2a_whiten_minmax = func_df(whiten_minmax, "frame_arr")
# transform_a2a_whiten_per_minmax = func_df(whiten_per_minmax, "frame_arr")
transform_a2a_whiten_per_reminmax = func_df(whiten_per_reminmax, "frame_arr")

# 一步到位
# transform_d2a_whiten_only = func_df(whiten_only, "keypoint_data", "frame_arr")
transform_d2a_minmax_whiten = func_df(minmax_whiten, "keypoint_data", "frame_arr")
# transform_d2a_per_minmax = func_df(per_minmax, "keypoint_data", "frame_arr")
transform_d2a_per_reminmax = func_df(per_reminmax, "keypoint_data", "frame_arr")
