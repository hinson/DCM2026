#
# app.py
# @author Quexuan Zhang
# @description
# @created 2024-12-20T10:15:32.949Z+08:00
# @last-modified 2026-01-15T17:00:21.758Z+08:00
#


import shutil
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from multiprocessing import Process
from pathlib import Path
from threading import RLock, current_thread
from typing import Callable, Final, Literal, Optional

import joblib
import lightgbm.plotting as lgbplt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import shap
import streamlit as st
import streamlit.components.v1 as components
from lightgbm import LGBMClassifier
from mpld3 import fig_to_html, plugins
from sklearn.pipeline import FunctionTransformer
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from streamlit.runtime.uploaded_file_manager import UploadedFile

from data.keypoint import *
from data.preprocessing import get_json_data
from data.timeseries import smooth_frame_arr
from data.utils import impute_data
from feature import FeatureGenerator, values_to_dataframe
from transform import transform_d2a_per_reminmax
from visualize import animate_keypoint_angles, animate_palm_areas, plot_dist_changes

matplotlib.use("agg")
plt.ioff()

_lock: Final[RLock] = RLock()

# 设置matplotlib的风格
plt.style.use("seaborn-v0_8-colorblind")
plt.rcParams.update(
    {
        "axes.titlesize": 10,  # - 坐标轴标题大小
        "ytick.labelsize": 9,  # - y轴标签大小
        "xtick.labelsize": 9,  # - x轴标签大小
        "legend.fontsize": 8,
        "font.size": 9,
        "font.family": [
            "Noto Serif CJK JP",
            "Times New Roman",
        ],  # 支持中文
        "mathtext.fontset": "stix",  # - 数学文本的字体集
        "axes.unicode_minus": False,  # - 坐标轴上的负号不使用Unicode编码
    }
)

st.set_page_config("DCM predictor", layout="wide", page_icon=":material/stethoscope:")
st.markdown(
    """<style>
div[data-testid="stDialog"] div[role="dialog"] {
    width: 80vw;
}
</style>
""",
    unsafe_allow_html=True,
)

data_dir: Final[Path] = Path("./data")
# static_dir: Final[Path] = Path("./static")
cache_dir: Final[Path] = Path("~/.cache/dcmapp").expanduser()
if not cache_dir.exists():
    cache_dir.mkdir(parents=True)


def visualize_data(row: pd.Series):
    name: str = row["filename"]
    data: KeypointData = row["keypoint_data"]
    frame_arr: np.ndarray = row["frame_arr"]
    st.header(name)

    columns = st.columns([2, 1, 1, 1, 1])

    # smooth_frame_arr(frame_arr)

    with columns[0], _lock:
        fig = plot_dist_changes(
            frame_arr,
            KEYPOINTS_TIPS,
            KEYPOINTS_WRIST,
            ("Thumb", "Index", "Middle", "Ring", "Pinky"),
            left=data.left,
            title=f"Distance Changes (tips-wrist) - {name}",
            figsize=(6.5, 3.2),
        )
        # st.pyplot(fig, use_container_width=False)

        fig.tight_layout()
        ax = fig.gca()
        lines = ax.get_lines()
        labels = [line.get_label()[0] for line in lines]

        plugins.connect(
            fig,
            plugins.InteractiveLegendPlugin(
                lines,
                labels,
                ax=ax,
                alpha_unsel=0.35,
                alpha_over=1.5,
                legend_offset=(-30, 0),
            ),
        )
        fig_html = fig_to_html(
            fig,
            template_type="simple",
        )
        components.html(fig_html, height=350, scrolling=True)

    # 动画 (GIFs)
    num_images = 4
    videos = [columns[1 + i].empty() for i in range(num_images)]

    def ani_angle():
        save_path = cache_dir / f"angels_{name}.mp4"
        if not save_path.exists():
            with columns[1], st.spinner():
                p = Process(
                    target=animate_keypoint_angles,
                    args=(frame_arr,),
                    kwargs=dict(
                        left=data.left,
                        title=f"Angles Visualization - {name}",
                        figsize=(3, 3),
                        save_fps=data.frame_rate,
                        save_path=save_path,
                    ),
                )
                p.start()
                p.join()

        f = save_path.open("rb")
        videos[0].video(f.read(), loop=True, autoplay=True)

    def ani_area(i, label, points):
        save_path = cache_dir / f"area_{label}_{name}.mp4"
        if not save_path.exists():
            with columns[i + 1], st.spinner():
                p = Process(
                    target=animate_palm_areas,
                    args=(
                        frame_arr,
                        points,
                    ),
                    kwargs=dict(
                        left=data.left,
                        title=f"Area ({label.title()}) - {name}",
                        figsize=(3, 3),
                        save_path=save_path,
                        save_fps=data.frame_rate,
                    ),
                )
                p.start()
                p.join()

        f = save_path.open("rb")
        videos[i].video(f.read(), loop=True, autoplay=True)

    ctx = get_script_run_ctx()
    with ThreadPoolExecutor(
        max_workers=num_images,
        initializer=lambda: add_script_run_ctx(current_thread(), ctx),
    ) as executor:
        futures = []
        futures.append(executor.submit(ani_angle))
        for i, label, points in zip(
            range(1, num_images),
            ("palm", "025", "034"),
            (AREA_VERTICES_PALM, AREA_VERTICES_REF025, AREA_VERTICES_REF034),
        ):
            futures.append(executor.submit(ani_area, i, label, points))

    # ani_angle()
    # for i, label, points in zip(
    #     range(1, num_images),
    #     ("palm", "025", "034"),
    #     (AREA_VERTICES_PALM, AREA_VERTICES_REF025, AREA_VERTICES_REF034),
    # ):
    #     ani_area(i, label, points)

    st.divider()


def _aggregate_shap_values(
    shap_values: list[shap.Explanation],
    df_value: pd.DataFrame,
    by=None,
    filter: Callable[[pd.DataFrame], bool] = lambda x: x == x,
):
    shap_values = deepcopy(shap_values)

    keys = []
    values = []
    for key, value in df_value[filter(df_value)].groupby(by=by)["value"].sum().items():
        keys.append("_".join(key))
        values.append(value)

    shap_values.values = np.asarray(values).T
    shap_values.feature_names = keys
    return shap_values


def _show_shap_waterfall(
    shap_value: shap.Explanation,
    period: Literal["whole", "open", "close"],
    name: str,
    max_display: int,
):
    with st.spinner(), _lock:
        plt.subplots(figsize=(0.7 * max_display, 10))
        shap.plots.waterfall(shap_value, max_display=max_display, show=False)
        fig = plt.gcf()
        ax = plt.gca()
        ax.set_title(f"SHAP waterfall ({period}) - {name}")

        # 修改因聚合而错误的y轴标签
        ylabels = ax.get_yticklabels()
        half_num_ylabels = len(ylabels) // 2
        ax.set_yticklabels(
            [""] * half_num_ylabels
            + [label.get_text().rsplit("=", 1)[0].strip() for label in ylabels[half_num_ylabels:]]
        )

        st.pyplot(fig)


def _show_treemap(
    df_value: pd.DataFrame,
    name: str,
    idx: Optional[int] = None,
    path: Sequence[str] = [],
    hovertemplate: str = "",
    max_depth=7,
    **kwargs,
):
    df_value = df_value.copy()

    path = ["all"] + path

    if idx is not None:
        df_value["value"] = df_value["value"].map(lambda x: x[idx])

    df_value["all"] = "all"
    df_value["abs_value"] = df_value["value"].abs() + 1e-12
    fig = px.treemap(
        df_value,
        path=path,
        values="abs_value",
        color="value",
        width=1280,
        height=1024,
        title=name,
        maxdepth=max_depth,
        **kwargs,
    )

    # 显示SHAP值
    dfs = []
    for i in range(len(path), 0, -1):
        by = path[:i]
        df_grouped = df_value.groupby(by)["value"].sum()
        df = pd.DataFrame(
            dict(
                ids=df_grouped.index.map(lambda t: t if isinstance(t, str) else "/".join(t[:i])),
                value=df_grouped,
            )
        )
        dfs.append(df)
    df_tree = pd.concat(dfs)
    df_tree.index = df_tree["ids"]
    tmdata = fig.data[0]
    tmdata["hovertemplate"] = hovertemplate
    for i, id in enumerate(tmdata["ids"]):
        if id in df_tree.index:
            tmdata["customdata"][i] = df_tree.loc[id, "value"]

    st.plotly_chart(fig, use_container_width=True)


@st.dialog("SHAP for sample")
def show_shap(
    df_value: pd.DataFrame,
    shap_values_dict: dict[str, list[shap.Explanation]],
    names: Sequence[str],
    waterfall_max_display: int,
    treemap_path: Sequence[str],
    treemap_max_depth: int,
):

    idx = st.session_state.result["selection"]["rows"][0]
    name = names[idx]

    tab_waterfall, tab_treemap = st.tabs(["Waterfall", "Treemap"])

    columns = tab_waterfall.columns(3)
    for column, (period, shap_values) in zip(columns, shap_values_dict.items()):

        with column:
            _show_shap_waterfall(shap_values[idx], period, name, waterfall_max_display)

    with tab_treemap:
        _show_treemap(
            df_value,
            name,
            idx,
            path=treemap_path,
            max_depth=treemap_max_depth,
            hovertemplate="id = %{id}<br>shap = %{customdata:.5f}<br>weighted_avg = %{color:.5f}",
            color_continuous_scale="jet",
            color_continuous_midpoint=0.0,
        )


def show_feature_importance(
    model_checkpoint: UploadedFile,
    raw_max_display: int,
    treemap_path: Sequence[str],
    treemap_max_depth: int,
):
    model = joblib.load(model_checkpoint)
    lgbc: LGBMClassifier = model.estimator_

    tab_raw, tab_treemap = st.tabs(["Raw", "Treemap"])

    with tab_raw, st.spinner(), _lock, plt.style.context("ggplot"):
        fig, ax = plt.subplots(
            figsize=(
                10,
                0.25 * raw_max_display,
            )
        )
        lgbplt.plot_importance(lgbc, ax=ax, max_num_features=raw_max_display)
        st.pyplot(fig, use_container_width=False)

    with tab_treemap, st.spinner():
        df_value = values_to_dataframe(np.log1p(lgbc.feature_importances_), lgbc.feature_name_)
        _show_treemap(
            df_value,
            model_checkpoint.name,
            path=treemap_path,
            max_depth=treemap_max_depth,
            hovertemplate="id = %{id}<br>sum_log_value = %{customdata:.5f}<br>weighted_avg = %{color:.5f}",
            color_continuous_scale="orrd",
        )


def show_tree_plots(model_checkpoint: UploadedFile, orientation: str):
    model = joblib.load(model_checkpoint)
    lgbc: LGBMClassifier = model.estimator_
    num_trees = lgbc.n_estimators_

    tree_id = st.pills("Choose a tree id to show its structure:", range(num_trees))

    if tree_id is not None:
        with st.spinner(), _lock:
            fig, ax = plt.subplots(figsize=(40, 10))
            lgbplt.plot_tree(
                lgbc,
                ax=ax,
                tree_index=tree_id,
                orientation=orientation,
            )
            fig.tight_layout()
            cache_path = cache_dir / f"{model_checkpoint.name}_tree_{tree_id}.svg"
            fig.savefig(cache_path)
            st.image(str(cache_path))


def predict_samples(
    uploaded_jsons: list[UploadedFile],
    input: pd.DataFrame,
    model_checkpoint: UploadedFile,
    heatmap_max_display: int,
    waterfall_max_display: int,
    treemap_path: Sequence[str],
    treemap_max_depth: int,
):
    col_pred, col_shap = st.columns([2, 3])

    with st.spinner():
        # 预测
        with col_pred:
            df_X = FeatureGenerator().fit_transform(input)
            df_X = impute_data(df_X)

            model = joblib.load(model_checkpoint)
            explainer = shap.Explainer(model.estimator_)

            features_selected = model.estimator_.feature_name_
            df_X = df_X[features_selected]

            y_prob = model.estimator_.predict_proba(df_X)[:, 1]
            y_pred = model.predict(df_X)

            shap_values = explainer(df_X)

            # 结果可视化
            ser_names = pd.Series([json.name for json in uploaded_jsons], name="Filename")
            ser_prob = pd.Series(
                y_prob, name=f"Prediction Score (cutoff = {model.best_threshold_:.4f})"
            )
            ser_pred = pd.Series(y_pred.astype(np.bool_), name="DCM Prediction")
            ser_shap = pd.Series(shap_values.values.sum(axis=1), name="SHAP value")

            df_value = values_to_dataframe(shap_values.values.T, shap_values.feature_names)
            shap_values_dict = {}

            for period in ("whole", "open", "close"):
                shap_values_dict[period] = _aggregate_shap_values(
                    shap_values,
                    df_value,
                    by=["type", "key"],
                    filter=lambda x: x["period"] == period,
                )

            st.header("Prediction Results")
            st.markdown("↓↓↓ Select a row to show SHAP visualizations for the sample.")
            st.dataframe(
                pd.concat([ser_names, ser_prob, ser_pred, ser_shap], axis=1),
                use_container_width=True,
                height=(len(uploaded_jsons) + 1) * 35 + 3,
                on_select=lambda: (
                    show_shap(
                        df_value,
                        shap_values_dict,
                        ser_names,
                        waterfall_max_display=waterfall_max_display,
                        treemap_path=treemap_path,
                        treemap_max_depth=treemap_max_depth,
                    )
                    if st.session_state.result["selection"]["rows"]
                    else "ignore"
                ),
                selection_mode="single-row",
                key="result",
            )

        # 整体SHAP可视化
        with col_shap, _lock:
            if len(df_X) > 1:
                for period, shap_values in shap_values_dict.items():
                    fig, ax = plt.subplots(
                        figsize=(0.5 * heatmap_max_display, 0.5 * heatmap_max_display)
                    )
                    ax.set_title(f"SHAP heatmap ({period})")
                    shap.heatmap_plot(
                        shap_values,
                        max_display=heatmap_max_display,
                        plot_width=0.5 * heatmap_max_display,
                        ax=ax,
                        show=False,
                    )

                    st.pyplot(fig, use_container_width=False)


# Sidebar
st.sidebar.title("DCM Predictor")

default_path: Final[list[str]] = ["type", "key", "attr", "period", "period_agg", "ts_agg"]
path_display_dict: Final[dict[str, str]] = dict(
    type="Type",
    key="Key",
    attr="Attribute",
    period="Period",
    period_agg="Period Aggregation",
    ts_agg="Timeseries Aggregation",
)


def display_path(option):
    return path_display_dict[option]


func = st.sidebar.selectbox(
    "Choose a tool",
    (
        "Data Visualization",
        "Feature Importance",
        "Tree Visualization",
        "Prediction",
    ),
)

if func in ("Prediction", "Feature Importance", "Tree Visualization"):
    model_checkpoint = st.sidebar.file_uploader(
        "Choose a model file", type="pkl", accept_multiple_files=False
    )

if func in ("Data Visualization", "Prediction"):
    uploaded_jsons = st.sidebar.file_uploader(
        "Choose sample files",
        type="json",
        accept_multiple_files=True,
    )

with st.sidebar.expander("Advanced Settings", icon=":material/settings:"):
    if func == "Feature Importance":
        raw_max_display = st.number_input("Raw: Max Display", 1, value=500, step=1, format="%d")

    if func == "Tree Visualization":
        orientation = st.pills(
            "Tree: Orientation",
            ["horizontal", "vertical"],
            default="horizontal",
            format_func=lambda s: s.title(),
        )

    if func == "Prediction":
        heatmap_max_display = st.number_input(
            "Heatmap: Max Display", 1, value=30, step=1, format="%d"
        )

        waterfall_max_display = st.number_input(
            "Waterfall: Max Display", 1, value=50, step=1, format="%d"
        )

    if func in ("Prediction", "Feature Importance"):

        treemap_path = st.segmented_control(
            "Treemap: Path",
            options=default_path,
            selection_mode="multi",
            default=default_path,
            format_func=display_path,
        )
        st.markdown(
            "**Current Path**:<br>" + " :arrow_right: ".join(map(display_path, treemap_path)),
            unsafe_allow_html=True,
        )

        treemap_max_depth = st.number_input(
            "Treemap: Max Depth",
            1,
            len(treemap_path) + 1,
            value=len(treemap_path) + 1,
            step=1,
            format="%d",
        )


if st.sidebar.button("Clear Cache"):
    with st.spinner():
        shutil.rmtree(cache_dir)
        cache_dir.mkdir()
        st.sidebar.success("Clearing cache succeed!")


# 功能执行
if func in ("Data Visualization", "Prediction") and uploaded_jsons:
    data_list = [
        {"filename": json.name, "keypoint_data": get_json_data(json)} for json in uploaded_jsons
    ]
    df_raw = pd.DataFrame(data_list)
    df_raw: pd.DataFrame = FunctionTransformer(transform_d2a_per_reminmax).fit_transform(df_raw)
    num_samples = len(df_raw)

    match func:
        case "Data Visualization":
            df_raw.apply(visualize_data, axis=1)

        case "Prediction":
            if model_checkpoint:
                predict_samples(
                    uploaded_jsons,
                    df_raw,
                    model_checkpoint,
                    heatmap_max_display,
                    waterfall_max_display,
                    treemap_path,
                    treemap_max_depth,
                )

if func == "Feature Importance" and model_checkpoint:
    show_feature_importance(model_checkpoint, raw_max_display, treemap_path, treemap_max_depth)


if func == "Tree Visualization" and model_checkpoint:
    show_tree_plots(model_checkpoint, orientation)
