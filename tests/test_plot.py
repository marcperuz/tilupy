import pytest
import pandas as pd
import os

import matplotlib.pyplot as plt

import tilupy.plot as tiplot


@pytest.fixture
def calib_results_coulomb():
    file = os.path.join(
        os.path.dirname(__file__), "data", "frankslide", "shaltop", "calib_coulomb.csv"
    )
    calib_coulomb = pd.read_csv(file, sep=";")
    return calib_coulomb


@pytest.fixture
def axs():
    fig, axes = plt.subplots(2, 1)
    return axes


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (dict(axs=None), (1, 2)),
        (dict(ncols=1), (2, 1)),
    ],
)
def test_plot_heatmaps1(calib_results_coulomb, kwargs, expected):
    fig, axes = tiplot.plot_heatmaps(
        calib_results_coulomb, ["CSI", "diff_runout"], "h_threshs", "delta1", **kwargs
    )
    assert axes.shape == expected


def test_plot_heatmaps2(calib_results_coulomb):
    fig, axes = plt.subplots(2, 3)
    fig, axes = tiplot.plot_heatmaps(
        calib_results_coulomb, ["CSI", "diff_runout"], "h_threshs", "delta1", axs=axes
    )
    assert axes.shape == (2, 3)
