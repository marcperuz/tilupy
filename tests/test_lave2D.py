# -*- coding: utf-8 -*-
"""
Created on Fri May 31 15:37:05 2024

@author: peruzzetto
"""
import pytest
import numpy as np

import tilupy.models.lave2D.initsimus as lave2Dinit


@pytest.mark.parametrize(
    "arg, expected",
    [
        ([1, 1], (np.array([[3], [1]]), np.array([[4, 2]]))),
        ([2, 1], (np.array([[3, 7], [1, 5]]), np.array([[4, 2, 6]]))),
        ([1, 2], (np.array([[6], [3], [1]]), np.array([[7, 5], [4, 2]]))),
        (
            [3, 2],
            (
                np.array([[12, 15, 17], [3, 7, 10], [1, 5, 8]]),
                np.array([[13, 11, 14, 16], [4, 2, 6, 9]]),
            ),
        ),
        (
            [4, 3],
            (
                np.array(
                    [
                        [24, 27, 29, 31],
                        [15, 18, 20, 22],
                        [3, 7, 10, 13],
                        [1, 5, 8, 11],
                    ]
                ),
                np.array(
                    [
                        [25, 23, 26, 28, 30],
                        [16, 14, 17, 19, 21],
                        [4, 2, 6, 9, 12],
                    ]
                ),
            ),
        ),
    ],
)
def test_make_edges_matrices(arg, expected):
    res = lave2Dinit.make_edges_matrices(*arg)
    assert np.array_equal(res[0], expected[0])
    assert np.array_equal(res[1], expected[1])


@pytest.mark.parametrize(
    "arg, expected",
    [
        ([[1.5, 2.5], [0, 0], "S"], [5, 8]),
        ([[0, 0], [0.1, 1.9], "W"], [4, 13]),
    ],
)
def test_get_edges(arg, expected):
    tmp = lave2Dinit.ModellingDomain(nx=4, ny=3, dx=1, dy=1, xmin=0, ymin=0)
    tmp.set_edges()
    edges = tmp.get_edges(*arg)
    assert edges[arg[2][0]] == expected
