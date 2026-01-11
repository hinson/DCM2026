#
# feature.py
# @author Quexuan Zhang
# @description
# @created 2024-12-05T15:49:19.357Z+08:00
# @last-modified 2025-12-11T11:05:38.053Z+08:00
#

import multiprocessing as mp
from collections import defaultdict
from collections.abc import Callable
from functools import partial
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
import psutil
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import minmax_scale
from tqdm.auto import tqdm
from tsfresh import extract_features as ts_extract_features
from tsfresh.feature_extraction.settings import (
    ComprehensiveFCParameters,
    MinimalFCParameters,
)

from data.geometry import (
    eudist_on_arrs,
    get_angle_arr,
    get_APB_arrs,
    get_polygon_vertices,
)
from data.keypoint import (
    ANGLE_VERTICES,
    AREA_VERTICES_PALM,
    AREA_VERTICES_REF025,
    AREA_VERTICES_REF034,
    KEYPOINTS_TIPS,
    KEYPOINTS_WRIST,
    KeypointData,
    KeypointSeq,
)
from data.timeseries import (
    Interval,
    diff_intervals,
    find_pause_intervals,
    find_period_peaks,
    get_asyns,
    get_period_intervals,
    get_periods,
    get_polygon_areas,
)
from transform import whiten_only


def default_tsfc_settings() -> ComprehensiveFCParameters:
    """默认时序列统计特征：maximum, mean, median, minimum, quantile(q=0.25), quantile(q=0.75), root_mean_square, skewness, standard_deviation, variance, variation_coefficien

    Returns:
        默认时序列统计特征参数配置
    """
    settings = MinimalFCParameters()
    del settings["length"]
    del settings["absolute_maximum"]
    del settings["sum_values"]
    settings.update(
        dict(
            variation_coefficient=None,
            skewness=None,
            quantile=[dict(q=0.25), dict(q=0.75)],
            # sample_entropy=None,
        )
    )
    return settings


class FeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(
        self, tsfc_settings: ComprehensiveFCParameters = default_tsfc_settings(), n_jobs=-1
    ) -> None:
        super().__init__()
        self.feature_columns_ = None
        self.tsfc_settings = tsfc_settings
        self.n_jobs = n_jobs

    def fit(self, X: pd.DataFrame, y=None):
        X_transformed = self._transform(X)
        self.feature_columns_ = X_transformed.columns
        return self

    def transform(self, X: pd.DataFrame):
        X_transformed: pd.DataFrame = self._transform(X)
        if self.feature_columns_ is not None:
            X_transformed = X_transformed.reindex(columns=self.feature_columns_, fill_value=0)
        return X_transformed

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return extract_features(X, self.tsfc_settings, self.n_jobs).sort_index()


FeatureDict = dict[str, np.ndarray]


def __extract_kinematic_features(datum: KeypointData, feat: np.ndarray, name: str) -> FeatureDict:
    """
    抽取运动学特征
    """
    feature_dict = {}

    frame_rate = datum.frame_rate

    geom = minmax_scale(feat)
    vel = np.diff(geom, axis=0) * frame_rate
    acc = np.diff(vel, axis=0) * frame_rate
    # jerk = np.diff(acc, axis=0) * frame_rate

    feature_dict[f"{name}_geom"] = geom
    feature_dict[f"{name}_vel"] = vel
    feature_dict[f"{name}_acc"] = acc
    # feature_dict[f"{name}_jerk"] = jerk

    return feature_dict


ComboMethod = Callable[[np.ndarray, np.ndarray], np.ndarray]
combo_method_dict: dict[str, ComboMethod] = {
    # "|{f1}-{f2}|": lambda F1, F2: np.abs(F1 - F2),
    # "||{f1}|-|{f2}||": lambda F1, F2: np.abs(np.abs(F1) - np.abs(F2)),
    "{f1}-{f2}": lambda F1, F2: F1 - F2,
    "|{f1}|-|{f2}|": lambda F1, F2: np.abs(F1) - np.abs(F2),
    "|{f1}|+|{f2}|": lambda F1, F2: np.abs(F1) + np.abs(F2),
    "2(|{f1}|-|{f2}|)/(|{f1}|+|{f2}|)": lambda F1, F2: 2.0
    * (np.abs(F1) - np.abs(F2))
    / (np.abs(F1) + np.abs(F2) + 1),  # 防止0除
}


def _extract_combo_dist_features(
    frame_feature_dict: FeatureDict, keypoint_ids: KeypointSeq = KEYPOINTS_TIPS
) -> FeatureDict:
    """
    抽取组合特征
    """
    feature_dict = {}
    for f1, f2 in combinations(keypoint_ids, 2):
        for kinetic in ("vel", "acc"):
            fname1 = f"dist_{f1}_{kinetic}"
            fname2 = f"dist_{f2}_{kinetic}"
            F1 = frame_feature_dict[fname1]
            F2 = frame_feature_dict[fname2]
            for name_tpl, method in combo_method_dict.items():
                name = f"combo_{name_tpl.format(f1=f'{f1}', f2=f'{f2}')}_{kinetic}"
                feature_dict[name] = method(F1, F2)

    return feature_dict


def _extract_dist_features(
    datum: KeypointData,
    frame_arr: np.ndarray,
    keypoint_ids: KeypointSeq = KEYPOINTS_TIPS,
    origin_ids: KeypointSeq = KEYPOINTS_WRIST,
) -> FeatureDict:
    """
    抽取距离运动学特征
    """
    dist_arr = eudist_on_arrs(frame_arr[:, keypoint_ids], frame_arr[:, origin_ids])

    feature_dict = {}
    for i, kid in enumerate(keypoint_ids):
        name = f"dist_{kid}"

        feature_dict.update(__extract_kinematic_features(datum, dist_arr[:, i], name))

    return feature_dict


def _extract_angle_features(
    datum: KeypointData, frame_arr: np.ndarray, vertex_ids: KeypointSeq = ANGLE_VERTICES
) -> FeatureDict:
    """
    抽取角度运动学特征
    """
    angle_arr = get_angle_arr(*get_APB_arrs(frame_arr), datum.left)

    feature_dict = {}
    for i, vid in enumerate(vertex_ids):
        name = f"angle_{vid}"
        feature_dict.update(__extract_kinematic_features(datum, angle_arr[:, i], name))

    return feature_dict


def _extract_area_features(
    datum: KeypointData,
    frame_arr: np.ndarray,
    vertex_ids_dict: dict[str, KeypointSeq] = {
        "palm": AREA_VERTICES_PALM,
        "025": AREA_VERTICES_REF025,
        "034": AREA_VERTICES_REF034,
    },
) -> FeatureDict:
    """
    抽取面积运动学特征
    """
    area_arr = np.column_stack(
        [
            get_polygon_areas(get_polygon_vertices(frame_arr, keypoints, datum.left))
            for keypoints in vertex_ids_dict.values()
        ]
    )

    feature_dict = {}
    for i, aid in enumerate(vertex_ids_dict.keys()):
        name = f"area_{aid}"
        feature_dict.update(__extract_kinematic_features(datum, area_arr[:, i], name))

    return feature_dict


def _extract_period_features(
    datum: KeypointData,
    frame_arr: np.ndarray,
    frame_feature_dict: FeatureDict,
    ref_vertex_ids: KeypointSeq = AREA_VERTICES_REF025,
    num_headtail_period_limits: int = 5,
) -> FeatureDict:
    """
    抽取周期相关特征
    """
    # 获取手掌面积大小极值点
    # frame_arr = whiten_only(datum)
    vertics = get_polygon_vertices(frame_arr, ref_vertex_ids, left=datum.left)
    areas = get_polygon_areas(vertics)
    peaks_min, peaks_max = find_period_peaks(areas, prominence=0.25)

    feature_dict = {}

    # 获取周期
    periods = get_periods(peaks_min, peaks_max)
    feature_dict["periods"] = periods

    # 动作前停顿/延迟
    feature_dict.update(__extract_pause_features(datum, peaks_min, peaks_max, frame_feature_dict))

    # 伸握阶段
    periods_open, periods_close = get_period_intervals(peaks_min, peaks_max)
    # if periods != 0 and not periods_open and not periods_close:
    #     raise RuntimeError(f"{peaks_min=}\n{peaks_max=}")

    feature_dict.update(__extract_phase_features(periods_open, periods_close, frame_feature_dict))

    # 首尾周期段伸握特征
    feature_dict.update(
        __extract_headtail_periods_features(
            periods_open, periods_close, frame_feature_dict, num_headtail_period_limits
        )
    )

    return feature_dict


def __extract_pause_features(
    datum: KeypointData,
    peaks_min: np.ndarray,
    peaks_max: np.ndarray,
    frame_feature_dict: FeatureDict,
):
    """
    抽取延迟特征
    """
    feature_dict = {}

    for name, feat in frame_feature_dict.items():
        pieces = name.rsplit("_", 1)

        if pieces[1] != "geom":
            continue

        # 对几何量进行延迟分析
        feat_diff = np.diff(feat, axis=0)
        threshold = ((0.2 * feat_diff.min()), (0.2 * feat_diff.max()))
        pause_intervals_open, pause_intervals_close = find_pause_intervals(
            feat_diff, peaks_min, peaks_max, threshold=threshold
        )

        # scaler = 100.0 / len(feat)  # 占比
        scaler = 1.0 / datum.frame_rate
        pauses_open = diff_intervals(
            pause_intervals_open,
            scaler=scaler,
        )
        pauses_close = diff_intervals(
            pause_intervals_close,
            scaler=scaler,
        )

        fname = pieces[0]
        feature_dict[f"pause_{fname}_whole"] = np.concatenate([pauses_open, pauses_close])
        feature_dict[f"pause_{fname}_open"] = pauses_open
        feature_dict[f"pause_{fname}_close"] = pauses_close

    return feature_dict


def __extract_phase_features(
    periods_open: list[Interval],
    periods_close: list[Interval],
    frame_feature_dict: FeatureDict,
) -> FeatureDict:
    """
    区分伸握阶段抽取运动学特征
    """
    feature_dict = {}
    for name, feat in frame_feature_dict.items():
        pieces = name.split("_")
        if pieces[0] not in ("dist", "combo", "angle", "area") or pieces[-1] in ("open", "close"):
            continue

        pfeat_dict = defaultdict(list)

        for pname, periods in (("open", periods_open), ("close", periods_close)):
            for start, end in periods:
                pfeat = feat[start:end]
                # pfeat_dict[pname].append(pfeat)

                pfeat_dict[f"mean_{pname}"].append(np.nanmean(pfeat))
                # pfeat_dict[f"std_{pname}"].append(np.nanstd(pfeat))
                pfeat_dict[f"median_{pname}"].append(np.nanmedian(pfeat))
                pfeat_dict[f"absmax_{pname}"].append(np.nanmax(np.abs(pfeat)))
                # pfeat_dict[f"sum_{pname}"].append(np.nansum(pfeat))

        for pfname, pfeats in pfeat_dict.items():
            feature_dict[f"{name}_{pfname}"] = np.asarray(pfeats)
            # feature_dict[f"{name}_{pfname}"] = np.concatenate(pfeats)

    return feature_dict


def __extract_headtail_periods_features(
    periods_open: list[Interval],
    periods_close: list[Interval],
    frame_feature_dict: FeatureDict,
    num_headtail_period_limits: int,
) -> FeatureDict:
    """
    区分伸握首尾分段分别抽取运动学特征
    """
    feature_dict = {}

    for name, feat in frame_feature_dict.items():
        pieces = name.split("_")
        if pieces[0] not in ("dist", "area") or pieces[-1] in (
            "open",
            "close",
        ):
            continue

        pfeat_dict = defaultdict(list)

        for pname, periods in (("open", periods_open), ("close", periods_close)):
            for sname, speriods in (
                ("head", periods[:num_headtail_period_limits]),
                ("tail", periods[-num_headtail_period_limits:]),
            ):
                for start, end in speriods:
                    pfeat = feat[start:end]
                    # pfeat_dict[pname].append(pfeat)

                    pfeat_dict[f"mean_{pname}{sname}"].append(np.nanmean(pfeat))
                    # pfeat_dict[f"std_{pname}{sname}"].append(np.nanstd(pfeat))
                    pfeat_dict[f"median_{pname}{sname}"].append(np.nanmedian(pfeat))
                    pfeat_dict[f"absmax_{pname}{sname}"].append(
                        np.nanmax(np.abs(pfeat))
                    )

                    # pfeat_dict[f"sum_{pname}"].append(np.nansum(pfeat))

        for pfname, pfeats in pfeat_dict.items():
            feature_dict[f"{name}_{pfname}"] = np.asarray(pfeats)
            # feature_dict[f"{name}_{pfname}"] = np.concatenate(pfeats)

    return feature_dict


def _extract_dtw_features(
    datum: KeypointData,
    frame_feature_dict: FeatureDict,
    keypoint_ids: KeypointSeq = KEYPOINTS_TIPS,
    ref_key: str = "area_025",
) -> FeatureDict:
    """
    抽取动态时间规整
    """
    feature_dict = {}

    scaler = 1.0 / datum.frame_rate

    # 指尖 - 025面积
    reference = frame_feature_dict[f"{ref_key}_vel"]

    for tid in keypoint_ids:
        fname = f"dist_{tid}_vel"
        feat = frame_feature_dict[fname]

        feature_dict[f"asyn_{tid}_ref"] = get_asyns(feat, reference, scaler=scaler, window=25)

    # 指尖 - 指尖
    for t1, t2 in combinations(keypoint_ids, 2):
        fname1 = f"dist_{t1}_vel"
        fname2 = f"dist_{t2}_vel"
        feat1 = frame_feature_dict[fname1]
        feat2 = frame_feature_dict[fname2]

        # scaler = 1000.0 / len(feat1)

        feature_dict[f"asyn_{t1}_{t2}"] = get_asyns(feat1, feat2, scaler=scaler, window=25)

    return feature_dict


def _merge_feature_dfs(id: int, feature_dict: FeatureDict) -> dict[str, pd.DataFrame]:
    """
    整合特征数据框
    """
    df_dict = {}

    to_merge_dict: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for name, feature in feature_dict.items():
        pieces = name.split("_")

        if pieces[0] in ("asyn", "pause"):  # 时序列特征不定长
            num = len(feature)
            df = pd.DataFrame({"id": [id] * num, "time": np.arange(num), name: feature})

            df_dict[name] = df  # 不定长特征独立存放

        else:
            cname = pieces[-1]  # 根据cname分类
            # if (
            #     pieces[0] == "combo" and "|-|" in pieces[1] and "|+|" not in pieces[1]
            # ) or cname == "acc":
            to_merge_dict[cname].update({name: feature})  # 存放同类特征待合并

    # 合并同类等长特征
    for cname, to_merge in to_merge_dict.items():
        num = len(next(iter(to_merge.values())))
        df = pd.DataFrame(dict(id=[id] * num, time=np.arange(num)) | to_merge)

        df_dict[cname] = df  # 存放合并后特征

    return df_dict


def _preprocess_sample(
    id: int, datum: KeypointData, frame_arr: np.ndarray
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, Any],
]:
    """
    处理单个样本
    """
    # mask = ~np.isnan(frame_arr).any(axis=(1, 2))
    # frame_arr = frame_arr[mask]
    time = len(frame_arr) / datum.frame_rate

    tip_ids = KEYPOINTS_TIPS  # [1:]
    angle_vertices = ANGLE_VERTICES  # [3:]

    feature_dict = {}

    try:
        # 距离相关
        feature_dict.update(_extract_dist_features(datum, frame_arr, tip_ids))
        feature_dict.update(_extract_combo_dist_features(feature_dict, tip_ids))

        # 角度和面积
        feature_dict.update(_extract_angle_features(datum, frame_arr, angle_vertices))
        feature_dict.update(_extract_area_features(datum, frame_arr))

        # 时序列
        feature_dict.update(_extract_period_features(datum, frame_arr, feature_dict))
        feature_dict.update(_extract_dtw_features(datum, feature_dict, tip_ids))
    except Exception as e:
        raise RuntimeError(f"处理数据{datum.name}出错：", e)

    periods = feature_dict.pop("periods")
    num_periods = len(periods) + 1  # 头尾大约缺失1个周期，同时防止零除
    period = time / num_periods
    frequency = 1.0 / period
    other_dict = {
        # "periodnum__-": num_periods,
        "period__mean": period,
        "frequency__mean": frequency,
    }

    merged_feature_df_dict = _merge_feature_dfs(id, feature_dict)

    return merged_feature_df_dict, other_dict


def _preprocess_sample_wrap(args):
    return _preprocess_sample(*args)


def _extract_ts_features(
    dfs: pd.DataFrame, tsfc_settings: ComprehensiveFCParameters
) -> pd.DataFrame:
    """
    利用tsfresh抽取时序聚合特征
    """
    df_feat = pd.concat(dfs, axis=0)
    return ts_extract_features(
        df_feat,
        column_id="id",
        column_sort="time",
        default_fc_parameters=tsfc_settings,
        disable_progressbar=True,
        n_jobs=1,
    )


def extract_features(df: pd.DataFrame, tsfc_settings: ComprehensiveFCParameters, n_jobs: int = -1):
    if n_jobs < 1:
        n_jobs = psutil.cpu_count(logical=False) - 1

    data = df["keypoint_data"]
    frame_arrs = df["frame_arr"]

    with mp.Pool(n_jobs) as pool:
        dfs_dict = defaultdict(list)
        other_values_dict = defaultdict(list)
        total = len(data)
        results = pool.imap(_preprocess_sample_wrap, zip(range(total), data, frame_arrs))
        for merged_feature_df_dict, other_dict in tqdm(
            results,
            desc="Preprocessing: ",
            total=total,
        ):
            for name, value in other_dict.items():
                other_values_dict[name].append(value)

            for name, df in merged_feature_df_dict.items():
                dfs_dict[name].append(df)

        extract_ts_features = partial(_extract_ts_features, tsfc_settings=tsfc_settings)
        results = pool.imap(extract_ts_features, dfs_dict.values())
        df_result = pd.concat(
            list(tqdm(results, desc="Extracting TS features: ", total=len(dfs_dict))), axis=1
        )

        for name, values in other_values_dict.items():
            df_result[f"other_{name}"] = np.asarray(values, dtype=np.float64)

    return df_result.sort_index()


def values_to_dataframe(values, feature_names):
    rows = []

    for value, fname in zip(values, feature_names):
        key = attr = period = period_agg = ts_agg = "-"

        # if fname == "num_periods":
        #     row = dict(
        #         type="num",
        #         key="periods",
        #         attr=attr,
        #         period="whole",
        #         period_agg=period_agg,
        #         ts_agg=ts_agg,
        #         value=value,
        #     )

        # else:

        feat, ts_agg = fname.split("__", maxsplit=1)

        pieces = feat.split("_")
        type = pieces[0]
        match type:
            case "angle" | "dist" | "area" | "combo":
                key = pieces[1]
                attr = pieces[2]
                if len(pieces) > 3:
                    period = pieces[4]
                    period_agg = pieces[3]
                else:
                    period = "whole"

            case "pause":
                type = pieces[1]
                key = pieces[2]
                attr = "pause"
                period = pieces[3]

            case "asyn":
                key = pieces[1]
                attr = "vel"
                period = "whole"

            case "other":
                key = pieces[1]

        row = dict(
            type=type,
            key=key,
            attr=attr,
            period=period,
            period_agg=period_agg,
            ts_agg=ts_agg,
            value=value,
        )

        rows.append(row)

    return pd.DataFrame(rows)


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.features]
