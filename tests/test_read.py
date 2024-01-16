# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 14:06:54 2024

@author: peruzzetto
"""

import pytest
import numpy as np
import scipy

import tilupy.read as tiread


@pytest.fixture
def simple_temporal_results():
    d = np.ones((10, 15, 5))
    d[..., 0] = 0
    d[..., -1] = -1
    test_data = tiread.TemporalResults(
        "h",
        d,
        np.arange(5),
        x=np.arange(10),
        y=np.arange(15),
    )
    return test_data


@pytest.fixture
def gaussian_temporal_results():
    xmin, xmax = -1, 1
    ymin, ymax = -1, 1
    dx = dy = 0.05
    x = np.arange(xmin, xmax + dx / 2, dx)
    y = np.arange(ymin, ymax + dy / 2, dy)

    Dx = xmax - xmin
    Dy = ymax - ymin
    nx = len(x)
    ny = len(y)
    nt = 10

    xv, yv = np.meshgrid(x, y, indexing="xy")
    yv = np.flip(yv, axis=0)
    pos = np.dstack((xv, yv))
    h = np.zeros((ny, nx, nt))
    meanx = np.linspace(xmin + 0.2 * Dx, xmax - 0.4 * Dx, nt)
    meany = np.linspace(ymin + 0.3 * Dy, ymax - 0.3 * Dy, nt)
    stdx = np.linspace(0.02 * Dx, 0.08 * Dx, nt)
    stdy = [0.05 * Dy for i in range(nt)]
    cov_r = np.linspace(0, 0.5, nt)
    for i in range(nt):
        mean = [meanx[i], meany[i]]
        cov = cov_r[i] * np.sqrt(stdx[i] * stdy[i])
        cov_m = [[stdx[i], cov], [cov, stdy[i]]]
        rm = scipy.stats.multivariate_normal(mean, cov_m)
        h[:, :, i] = rm.pdf(pos)

    test_data = tiread.TemporalResults(
        "h",
        h,
        [0, 1, 2],
        x=x,
        y=y,
    )

    return test_data


@pytest.mark.parametrize(
    "arg, expected",
    [
        ("int", ("h_int", 2, 2.5)),
        ("mean", ("h_mean", 2, 0.4)),
    ],
)
def test_get_temporal_stat(arg, expected, simple_temporal_results):
    res = simple_temporal_results.get_temporal_stat(arg)
    res_out = (res.name, res.d.ndim, res.d[0, 0])
    assert res_out == expected


@pytest.mark.parametrize(
    "args, expected",
    [
        (("int", "y"), ("h_int_y", 2, 0, 10)),
        (("int", "x"), ("h_int_x", 2, 0, 15)),
        (("mean", "x"), ("h_mean_x", 2, 0, 1)),
    ],
)
def test_get_spatial_stat(args, expected, simple_temporal_results):
    res = simple_temporal_results.get_spatial_stat(*args)
    res_out = (res.name, res.d.ndim, res.d[0, 0], res.d[0, 1])
    assert res_out == expected
