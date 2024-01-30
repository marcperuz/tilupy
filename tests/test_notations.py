# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 14:38:19 2024

@author: peruzzetto
"""

import pytest

import tilupy.notations


@pytest.mark.parametrize(
    "arg, expected",
    [
        ("h_max", "$h_{max}$ (m)"),
        ("drdr_max", "$drdr_{max}$"),
    ],
)
def test_notations_with_unit(arg, expected):
    res = tilupy.notations.get_label(arg, label_type="symbol")
    assert res == expected


@pytest.mark.parametrize(
    "arg, expected",
    [
        (("h", "max", None), "$h_{max}$ (m)"),
        (("u", "int", "x"), "$u_{int(x)}$ (m$^2$ s$^{-1}$)"),
    ],
)
def test_add_operator_unit(arg, expected):
    notation = tilupy.notations.add_operator(arg[0], arg[1], axis=arg[2])
    res = tilupy.notations.get_label(notation, label_type="symbol")
    assert res == expected
