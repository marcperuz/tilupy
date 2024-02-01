#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 12:20:34 2021

@author: peruzzetto
"""
import pandas as pd

LABEL_OPTIONS = dict(language="english", label_type="symbol")


class Notation:
    def __init__(
        self, name, long_name=None, gender=None, symbol=None, unit=None
    ):
        self.name = name
        self.long_name = long_name
        self.gender = gender
        self.symbol = symbol
        self.unit = unit

    @property
    def unit(self):
        return self._unit

    @unit.setter
    def unit(self, value):
        if value is None:
            self._unit = None
        else:
            self._unit = value

    @property
    def gender(self):
        return self._gender

    @gender.setter
    def gender(self, value):
        if value is None:
            self._gender = Gender()
        elif isinstance(value, dict):
            self._gender = Gender(**value)
        else:
            self._gender = value

    @property
    def long_name(self):
        return self._long_name

    @long_name.setter
    def long_name(self, value):
        if value is None:
            self._long_name = LongName()
        elif isinstance(value, dict):
            self._long_name = LongName(**value)
        else:
            self._long_name = value

    def get_long_name(self, language=None, gender=None):
        if isinstance(self.long_name, str):
            return self.long_name

        if language is None:
            language = LABEL_OPTIONS["language"]

        res = getattr(self.long_name, language)
        if gender is not None:
            res = res[gender]

        return res


class Unit(pd.Series):
    UNITS = ["Pa", "N", "kg", "m", "s"]

    def __init__(self, series=None, **kwargs):
        if series is not None:
            super().__init__(series)
        else:
            super().__init__()
            for key in kwargs:
                if key not in Unit.UNITS:
                    raise ValueError("unrecognized unit")
                self[key] = kwargs[key]

    def __mul__(self, other):
        tmp = self.add(other, fill_value=0)
        return Unit(tmp[tmp != 0])

    def get_label(self):
        if self.empty:
            return ""

        positives = self[self >= 1].reindex(Unit.UNITS).dropna()
        negatives = self[self < 0].reindex(Unit.UNITS).dropna()
        text_label = [
            index + "$^{:.0f}$".format(positives[index])
            for index in positives.index
        ]
        text_label += [
            index + "$^{{{:.0f}}}$".format(negatives[index])
            for index in negatives.index
        ]
        text_label = " ".join(text_label)
        text_label = text_label.replace("$^1$", "")

        return text_label


class LongName(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Gender(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


NOTATIONS = dict()

NOTATIONS["x"] = Notation(
    "x",
    symbol="X",
    unit=Unit(m=1),
    long_name=LongName(english="X", french="X"),
    gender=Gender(english=None, french="f"),
)
NOTATIONS["y"] = Notation(
    "y",
    symbol="Y",
    unit=Unit(m=1),
    long_name=LongName(english="Y", french="Y"),
    gender=Gender(english=None, french="f"),
)
NOTATIONS["t"] = Notation(
    "t",
    symbol="t",
    unit=Unit(s=1),
    long_name=LongName(english="Time", french="Temps"),
    gender=Gender(english=None, french="m"),
)
NOTATIONS["xy"] = Notation(
    "xy",
    symbol="XY",
    unit=Unit(m=2),
    long_name=LongName(english="Time", french="Temps"),
    gender=Gender(english=None, french="m"),
)
NOTATIONS["h"] = Notation(
    "h",
    symbol="h",
    unit=Unit(m=1),
    long_name=LongName(english="thickness", french="épaisseur"),
    gender=Gender(english=None, french="f"),
)
NOTATIONS["hvert"] = Notation(
    "hvert",
    symbol="h^v",
    unit=Unit(m=1),
    long_name=LongName(
        english="vertical thickness", french="épaisseur verticale"
    ),
    gender=Gender(english=None, french="f"),
)
NOTATIONS["u"] = Notation(
    "u",
    symbol="u",
    unit=Unit(m=1, s=-1),
    long_name=LongName(english="velocity", french="vitesse"),
    gender=Gender(english=None, french="f"),
)
NOTATIONS["hu"] = Notation(
    "hu",
    symbol="hu",
    unit=Unit(m=2, s=-1),
    long_name=None,
    gender=None,
)

NOTATIONS["max"] = Notation(
    "max",
    symbol="max",
    unit=None,
    long_name=LongName(
        english="maximum", french=dict(m="maximum", f="maximale")
    ),
    gender=None,
)
NOTATIONS["int"] = Notation(
    "int",
    symbol="int",
    unit=None,
    long_name=LongName(
        english="maximum", french=dict(m="maximum", f="maximale")
    ),
    gender=None,
)


def get_notation(name, language=None):
    try:
        notation = NOTATIONS[name]
    except KeyError:
        strings = name.split("_")
        if len(strings) == 1:
            notation = Notation(
                name, symbol=name, unit=None, long_name=name, gender=None
            )
        else:
            state = get_notation(strings[0])
            operator = get_notation(strings[1])
            if len(strings) == 3:
                axis = strings[2]
            else:
                axis = None
            notation = add_operator(
                state, operator, axis=axis, language=language
            )

    return notation


def get_operator_unit(name, axis):
    if name == "int":
        if axis == "t" or axis is None:
            unit = Unit(s=1)
        if axis in ["x", "y"]:
            unit = Unit(m=1)
        if axis == "xy":
            unit = Unit(m=2)
    else:
        unit = Unit(pd.Series())
    return unit


def make_long_name(notation, operator, language=None):
    if language is None:
        language = LABEL_OPTIONS["language"]

    str_notation = notation.get_long_name(language=language)
    try:
        gender = gender = getattr(notation.gender, language)
    except AttributeError:
        gender = None
    str_operator = operator.get_long_name(language=language, gender=gender)

    if language == "english":
        res = str_operator + " " + str_notation
    elif language == "french":
        res = str_notation + " " + str_operator

    return res


def add_operator(notation, operator, axis=None, language=None):
    if isinstance(operator, str):
        operator = get_notation(operator)

    if isinstance(notation, str):
        notation = get_notation(notation)

    operator_symbol = operator.symbol
    if axis is not None:
        operator_symbol += "({})".format(axis)

    unit_operator = get_operator_unit(operator.name, axis)

    if operator.name == "int":
        if axis is None:
            ll = "t"  # If axis is not specified by default integration is over time
        else:
            ll = axis
        symbol = "\\int_{{{}}} {}".format(ll, notation.symbol)
    else:
        operator_symbol = operator.symbol
        if axis is not None:
            operator_symbol += "({})".format(axis)
        symbol = notation.symbol + "_{{{}}}".format(operator_symbol)

    if notation.unit is None:
        unit = None
    else:
        unit = notation.unit * unit_operator

    res = Notation(
        name=notation.name + "_" + operator.name,
        symbol=symbol,
        unit=unit,
        long_name=make_long_name(notation, operator, language=language),
    )
    return res


def get_label(notation, with_unit=True, label_type=None, language=None):
    if isinstance(notation, str):
        notation = get_notation(notation)

    if label_type is None:
        label_type = LABEL_OPTIONS["label_type"]
    if language is None:
        language = LABEL_OPTIONS["language"]

    if label_type == "litteral":
        label = notation.get_long_name(language=language, gender=None)
    elif label_type == "symbol":
        label = "$" + notation.symbol + "$"

    if with_unit and notation.unit is not None:
        unit_string = notation.unit.get_label()
        # Add unit only if string is not empty
        if unit_string:
            label = label + " ({})".format(unit_string)

    return label


def set_label_options(**kwargs):
    global LABEL_OPTIONS
    LABEL_OPTIONS.update(**kwargs)


def readme_to_params(file, readme_param_match=None):
    """
    Convert README file as dictionnary of parameters, following matching names.

    Parameters
    ----------
    file : TYPE
        DESCRIPTION.
    readme_param_match : TYPE
        DESCRIPTION.

    Returns
    -------
    params : TYPE
        DESCRIPTION.

    """
    params = dict()
    with open(file, "r") as f:
        if readme_param_match is not None:
            for line in f:
                (key, val) = line.split()
                if key in readme_param_match:
                    params[readme_param_match[key]] = val
        else:
            for line in f:
                (key, val) = line.split()
                params[key] = val

    return params


def make_rheol_string_fmt(rheoldict, law="coulomb"):
    """Make string from rheological parameters."""
    text = ""
    for name in ["delta1", "delta2", "delta3", "delta4"]:
        if name in rheoldict:
            new_txt = name + "_{:05.2f}_"
            text += new_txt
    if "ksi" in rheoldict:
        new_txt = name + "ksi_{:06.1f}_"
        text += new_txt
    text = text[:-1]

    return text


def make_rheol_string(rheoldict, law):
    """Make strings from rheological parameters."""

    keys = [key for key in rheoldict]
    for key in keys:
        if not isinstance(rheoldict[key], list):
            rheoldict[key] = [rheoldict[key]]

    nparams = len(rheoldict[keys[0]])
    txt_fmt = make_rheol_string_fmt(rheoldict)
    texts = []

    for i in range(nparams):
        if law == "coulomb":
            txt_fmt = "delta1_{:05.2f}"
            text = txt_fmt.format(rheoldict["delta1"][i]).replace(".", "p")
        if law == "voellmy":
            txt_fmt = "delta1_{:05.2f}_ksi_{:06.1f}"
            text = txt_fmt.format(
                rheoldict["delta1"][i], rheoldict["ksi"][i]
            ).replace(".", "p")
        if law == "pouliquen_2002":
            txt_fmt = "delta1_{:05.2f}_L_{:05.2f}"
            text = txt_fmt.format(
                rheoldict["delta1"][i], rheoldict["wlong"][i]
            ).replace(".", "p")
        texts.append(text)

    if nparams == 1:
        texts = texts[0]

    return texts
