# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 14:38:19 2024

@author: peruzzetto
"""

import pytest

import tilupy.notations as notations


@pytest.fixture
def custom_notation():
    tmp = notations.Notation(
        "tau",
        long_name=dict(english="stress", french="contrainte"),
        symbol="\tau",
        unit=notations.Unit(Pa=1),
        gender=dict(english=None, french="f"),
    )
    return tmp


def test_get_label(custom_notation):
    label = notations.get_label(custom_notation)
    assert label == "$\tau$ (Pa)"


@pytest.mark.parametrize(
    "arg, expected",
    [
        ("h_max", "$h_{max}$ (m)"),
        ("drdr_max", "$drdr_{max}$"),
    ],
)
def test_get_label_from_string(arg, expected):
    res = notations.get_label(arg, label_type="symbol")
    assert res == expected


@pytest.mark.parametrize(
    "arg, expected",
    [
        (("max", None), "$\tau_{max}$ (Pa)"),
        (("int", "x"), "$\\int_{x} \tau$ (Pa m)"),
        (("int", "xy"), "$\\int_{xy} \tau$ (Pa m$^2$)"),
        (("int", "t"), "$\\int_{t} \tau$ (Pa s)"),
        (("int", None), "$\\int_{t} \tau$ (Pa s)"),
    ],
)
def test_add_operator(arg, expected, custom_notation):
    notation = notations.add_operator(custom_notation, arg[0], axis=arg[1])
    res = notations.get_label(notation, label_type="symbol")
    assert res == expected


@pytest.mark.parametrize(
    "arg, expected",
    [
        (("h", "max", None), "$h_{max}$ (m)"),
        (("u", "int", "x"), "$\\int_{x} u$ (m$^2$ s$^{-1}$)"),
    ],
)
def test_add_operator_fromstring(arg, expected):
    notation = notations.add_operator(arg[0], arg[1], axis=arg[2])
    res = notations.get_label(notation, label_type="symbol")
    assert res == expected
