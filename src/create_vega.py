from pathlib import Path
from os.path import relpath
import altair as alt
import pandas as pd
import numpy as np
import io

_z = "frequency"
_x = "channel"
_y = "type"


def modify_default(in_data):
    return in_data


def modify_matrix(in_data):
    # Handle legacy matrix specification
    if "ClustName" in in_data.columns:
        in_data = pd.melt(in_data, id_vars=["ClustName"], var_name=_x, value_name=_z)
        in_data = in_data.rename(columns={"ClustName": _y})
        in_data = in_data[[_x, _y, _z]]
    return in_data


def create_matrix(in_data, params={}):

    in_data = modify_matrix(in_data)

    x_order = []
    y_order = []
    # Ensure matrix is sorted by occurence in csv file
    for x, y in zip(in_data[_x], in_data[_y]):
        if x not in x_order:
            x_order.append(x)
        if len(x_order) == 1:
            y_order.append(y)

    x = alt.X(_x, type="nominal", sort=x_order)
    y = alt.Y(_y, type="nominal", sort=y_order)

    out_chart = alt.Chart(in_data).mark_rect().encode(x, y, color=f"{_z}:Q")
    return out_chart


def create_scatterplot(in_data, params={}):
    xLabel = params["xLabel"]
    yLabel = params["yLabel"]
    clusters = params["clusters"]
    colors = [f"#{c}" for c in params["colors"]]
    dict_clusters = pd.DataFrame(
        [{"clust_ID": index + 1, "Cluster": c} for (index, c) in enumerate(clusters)]
    )
    out_chart = (
        alt.Chart(in_data)
        .mark_circle(size=60)
        .encode(
            alt.X(xLabel, type="quantitative", scale=alt.Scale(zero=False)),
            alt.Y(yLabel, type="quantitative", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "Cluster:N", scale=alt.Scale(domain=clusters, range=colors)
            ),
        )
        .transform_lookup(
            lookup="clust_ID",
            from_=alt.LookupData(
                data=dict_clusters, key="clust_ID", fields=["Cluster"]
            ),
        )
        .transform_filter(alt.datum.Cluster != None)
    )
    return out_chart


def create_barchart(in_data, params={}):
    out_chart = alt.Chart(in_data).mark_bar().encode(x="type", y="frequency")
    return out_chart


def create_vega_csv(in_path, out_path, modify_fn):
    in_data = pd.read_csv(in_path)
    in_data = modify_fn(in_data)

    if out_path is None:
        bytes_io = io.BytesIO()
        in_data.to_csv(bytes_io, index=False)
        bytes_io.seek(0)
        return bytes_io

    with open(out_path, "w+") as wf:
        in_data.to_csv(wf, index=False)

    return None


def create_vega_dict(in_path, out_path, create_fn, params={}):
    alt.renderers.set_embed_options(theme="dark")
    in_data = pd.read_csv(in_path)
    vega_chart = create_fn(in_data, params)
    vega_dict = vega_chart.to_dict()
    del vega_dict["datasets"][vega_dict["data"]["name"]]
    vega_dict["data"] = {"url": str(out_path)}
    if "config" not in vega_dict:
        vega_dict["config"] = {}
    if "color" not in vega_dict["encoding"]:
        vega_dict["encoding"]["color"] = {}
    # Style the output chart
    vega_dict["encoding"]["color"]["legend"] = {
        "direction": "horizontal",
        "orient": "bottom"
    }
    vega_dict["encoding"]["x"]["grid"] = False
    vega_dict["encoding"]["y"]["grid"] = False
    vega_dict["config"]["background"] = None
    vega_dict["width"] = "container"
    return vega_dict
