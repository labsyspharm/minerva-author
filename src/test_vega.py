import os
from pathlib import Path
from create_vega import create_scatterplot
from create_vega import create_barchart
from create_vega import create_matrix
from create_vega import create_vega_dict

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def create_vega_test(name, fn, params):
    prefix = "../testcharts"
    in_path = Path(prefix) / f"{name}.csv"
    out_path = Path(f"{name}_vega.csv")
    return create_vega_dict(in_path, out_path, fn, params)


def test_create_vega_scatterplot():
    result = create_vega_test(
        "scatterplot",
        create_scatterplot,
        {
            "clusters": ["Tumor", "Other", "Immune", "Stromal"],
            "xLabel": "KERATIN",
            "yLabel": "CD45",
        },
    )
    assert result["data"]["url"] == "scatterplot_vega.csv"


def test_create_vega_barchart():
    result = create_vega_test("barchart", create_barchart, {})
    assert result["data"]["url"] == "barchart_vega.csv"


def test_create_vega_matrix():
    result = create_vega_test("matrix", create_matrix, {})
    assert result["data"]["url"] == "matrix_vega.csv"
