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
    dx = dy = 0.1
    x = np.arange(xmin, xmax + dx / 2, dx)
    y = np.arange(ymin, ymax + dy / 2, dy)

    nx = len(x)
    ny = len(y)
    nt = 3

    step_x = (xmax - xmin) / 5
    step_y = (ymax - ymin) / 8
    xv, yv = np.meshgrid(x, y, indexing="xy")
    yv = np.flip(yv, axis=0)
    pos = np.dstack((xv, yv))
    h = np.zeros((ny, nx, nt))
    meanx = [xmin + step_x, xmin + 1.5 * step_x, xmin + 2.5 * step_x]
    meany = [
        (ymax + ymin) / 2 - step_y,
        (ymax + ymin) / 2,
        (ymax + ymin) / 2 + step_y,
    ]
    stdx = [step_x / 4, step_x / 3, step_x / 2]
    stdy = [step_y / 3, step_y / 3, step_y / 3]
    cov_r = [0, 0.3, 0]
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
