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


def test_get_temporal_stat(data_temporal_results):
    res = data_temporal_results.get_temporal_stat("int")
    assert res.name == "h_int"
    assert res.d.ndim == 2
    assert res.d[0, 0] == 2.5


def test_get_spatial_stat_npfn(data_temporal_results):
    res = data_temporal_results.get_spatial_stat("int", "y")
    assert res.name == "h_int_y"
    assert res.d.ndim == 2
    assert res.d[0, 0] == 0
    assert res.d[0, 1] == 15.0
