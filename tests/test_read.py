# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 14:06:54 2024

@author: peruzzetto
"""

import pytest
import numpy as np

import tilupy.read as tiread


@pytest.fixture
def data_temporal_results():
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


@pytest.mark.parametrize(
    "arg, expected",
    [
        ("int", ("h_int", 2, 2.5)),
        ("mean", ("h_mean", 2, 0.4)),
    ],
)
def test_get_temporal_stat(arg, expected, data_temporal_results):
    res = data_temporal_results.get_temporal_stat(arg)
    res_out = (res.name, res.d.ndim, res.d[0, 0])
    assert res_out == expected


@pytest.mark.parametrize(
    "args, expected",
    [
        (("int", "y"), ("h_int_y", 2, 0, 15)),
        (("int", "x"), ("h_int_x", 2, 0, 10)),
        (("mean", "x"), ("h_mean_x", 2, 0, 1)),
    ],
)
def test_get_spatial_stat(args, expected, data_temporal_results):
    res = data_temporal_results.get_spatial_stat(*args)
    res_out = (res.name, res.d.ndim, res.d[0, 0], res.d[0, 1])
    assert res_out == expected
