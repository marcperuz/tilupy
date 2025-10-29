#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import tilupy
import pandas as pd


LABEL_OPTIONS = dict(language="english", label_type="symbol")
"""Dictionary of configuration options for label generation.

This dictionary specifies the settings used for generating or formatting labels.
It includes the language of the labels and the type of label representation.

Keys:
    - language : str
        Language used for the labels (e.g., "english").
    - label_type : str
        Type of label representation (e.g., "symbol" for symbolic labels).
"""

class Notation:
    """Notation system for physical quantities, symbols, or variables.

    This class allows the definition of a notation with a name, long name, gender, symbol, and unit.
    It provides properties to access and modify these attributes, and a method to retrieve the long name
    in a specific language.
    
    Parameters
    ----------
    name : str
        The short name or identifier of the notation.
    long_name : tilupy.notations.LongName or str, optional
        The long name of the notation. If a dictionary is provided, it is converted to a 
        :any:`tilupy.notations.LongName` object. By default None.
    gender : tilupy.notations.Gender, optional
        The gender associated with the notation. If a dictionary is provided, it is converted to a 
        :obj:`tilupy.notations.Gender` object. By default None.
    symbol : str, optional
        The symbol representing the notation. By default None.
    unit : tilupy.notations.Unit, optional
        The unit associated with the notation. By default None.

    Attributes
    ----------
    _name : str
        The short name or identifier of the notation.
    _long_name : tilupy.notations.LongName or str
        The long name of the notation, which can be language-specific or gender-specific.
    _gender : tilupy.notations.Gender
        The gender associated with the notation, if applicable.
    _symbol : str
        The symbol representing the notation (e.g., mathematical symbol).
    _unit : tilupy.notations.Unit
        The unit associated with the notation (e.g., physical unit).
    """
    def __init__(self, 
                 name: str, 
                 long_name: tilupy.notations.LongName | str = None, 
                 gender: tilupy.notations.Gender | str = None, 
                 symbol: str = None, 
                 unit: tilupy.notations.Unit = None):
        self._name = name
        self._long_name = long_name
        self._gender = gender
        self._symbol = symbol
        self._unit = unit
    
    
    @property
    def name(self):
        """Get name.

        Returns
        -------
        str
            Attribute :attr:`_name`
        """
        return self._name
    
    @name.setter
    def name(self, value: str):
        """Set name.
        
        Parameters
        ----------
        value : str
            Name value
        """
        if value is None:
            self._name = None
        else:
            self._name = value
    
    @property
    def unit(self):
        """Get unit.

        Returns
        -------
        tilupy.notations.Unit
            Attribute :attr:`_unit`
        """
        return self._unit

    @unit.setter
    def unit(self, value: str):
        """Set unit.
        
        Parameters
        ----------
        value : tilupy.notations.Unit
            Unit value
        """
        if value is None:
            self._unit = None
        else:
            self._unit = value

    @property
    def gender(self):
        """Get gender.

        Returns
        -------
        tilupy.notations.Gender
            Attribute :attr:`_gender`
        """
        return self._gender

    @gender.setter
    def gender(self, value):
        """Set unit.
        
        Parameters
        ----------
        value : tilupy.notations.Gender
            Gender value
        """
        if value is None:
            self._gender = Gender()
        elif isinstance(value, dict):
            self._gender = Gender(**value)
        else:
            self._gender = value
            
    @property
    def symbol(self):
        """Get symbol.

        Returns
        -------
        str
            Attribute :attr:`_symbol`
        """
        return self._symbol
    
    @symbol.setter
    def symbol(self, value: str):
        """Set symbol.
        
        Parameters
        ----------
        value : str
            Symbol value
        """
        if value is None:
            self._symbol = None
        else:
            self._symbol = value

    @property
    def long_name(self):
        """Get long name.

        Returns
        -------
        tilupy.notations.LongName
            Attribute :attr:`_long_name`
        """
        return self._long_name

    @long_name.setter
    def long_name(self, value):
        """Set long name.
        
        Parameters
        ----------
        value : tilupy.notations.LongName
            Long name value
        """
        if value is None:
            self._long_name = LongName()
        elif isinstance(value, dict):
            self._long_name = LongName(**value)
        else:
            self._long_name = value

    def get_long_name(self, 
                      language: str = None, 
                      gender: str = None
                      ) -> str:
        """Retrieve the long name of the notation in the specified language and gender form.

        The method retrieves the long name in the specified language (defaulting to the language 
        in :data:`tilupy.notations.LABEL_OPTIONS`). If a gender is provided, the method returns 
        the gender-specific form of the long name.

        Parameters
        ----------
        language : str, optional
            The language in which to retrieve the long name. If not provided, the language from 
            :data:`tilupy.notations.LABEL_OPTIONS` is used.
        gender : str, optional
            The gender form of the long name to retrieve. If not provided, the default form is returned.

        Returns
        -------
        str
            The long name in the specified language and gender form. If the long name is a string, it is returned as-is.
        """
        if isinstance(self._long_name, str):
            return self._long_name

        if language is None:
            language = LABEL_OPTIONS["language"]

        if isinstance(self._long_name, dict):
            return self._long_name[language]
        
        res = getattr(self._long_name, language)
        if gender is not None:
            res = res[gender]

        return res


class Unit(pd.Series):
    """Subclass of pandas.Series to represent physical units and their exponents.

    This class allows the creation of unit objects as combinations of base units (e.g., "Pa", "N", "kg")
    with their respective exponents. It supports basic operations like multiplication and provides
    a method to generate a LaTeX-formatted label for the unit combination.
    
    Parameters
    ----------
    series : pandas.Series, optional
        An existing Series to initialize the Unit object.
    **kwargs : dict
        Key-value pairs where keys are unit names (from `Unit.UNITS`)
        and values are their exponents (as integers or floats).
        If provided, only units in `Unit.UNITS` are allowed.

    Raises
    ------
    ValueError
        If a key in `kwargs` is not in `Unit.UNITS`.
    """
    
    UNITS = ["Pa", "N", "kg", "m", "s", "J"]
    """List of available units."""

    def __init__(self, 
                 series: pd.Series = None, 
                 **kwargs):
        if series is not None:
            super().__init__(series)
        else:
            super().__init__(dtype="object")
            for key in kwargs:
                if key not in Unit.UNITS:
                    raise ValueError("unrecognized unit")
                self[key] = kwargs[key]


    def __mul__(self, other):
        """Multiply two Unit objects.

        The multiplication combines the exponents of matching units.
        Units with zero exponents are dropped from the result.

        Parameters
        ----------
        other : tilupy.notations.Unit
            Another :class:`tilupy.notations.Unit` object to multiply with.

        Returns
        -------
        tilupy.notations.Unit
            A new Unit object representing the product of the two units.
        """
        tmp = self.add(other, fill_value=0)
        return Unit(tmp[tmp != 0])


    def get_label(self):
        """Generate a LaTeX-formatted string representation of the unit.

        The label combines positive and negative exponents, omitting exponents of 1.
        Positive exponents are written as superscripts, while negative exponents
        are enclosed in curly braces (for LaTeX compatibility).

        Returns
        -------
        str
            A LaTeX-formatted string representing the unit (e.g., "kg m s$^{-2}$").
            Returns an empty string if the Unit object is empty.
        """
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
    """Generic container class to dynamically store LongName attributes as key-value pairs.

    This class allows the creation of objects with arbitrary attributes,
    which are passed as keyword arguments during initialization.

    Parameters
    ----------
    **kwargs : dict
        Arbitrary keyword arguments. Each key-value pair is added as an attribute to 
        the object.
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Gender(object):
    """Generic container class to dynamically store Gender attributes as key-value pairs.

    This class allows the creation of objects with arbitrary attributes,
    which are passed as keyword arguments during initialization.

    Parameters
    ----------
    **kwargs : dict
        Arbitrary keyword arguments. Each key-value pair is added as an attribute to 
        the object.
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


NOTATIONS = dict()
"""Dictionary containing predefined notations.

Pre-made notations:

    - x [m]
    - y [m]
    - d [m]
    - z [m]
    - t [s]
    - xy [m2]
    - h [m]
    - hvert [m]
    - u [m1.s-1]
    - hu [m2.s-1]
    - ek [J]

Also some operators:
    
    - max
    - int
"""

NOTATIONS["x"] = Notation("x",
                          symbol="X",
                          unit=Unit(m=1),
                          long_name=LongName(english="X", 
                                             french="X"),
                          gender=Gender(english=None, 
                                        french="f"),
                          )
NOTATIONS["y"] = Notation("y",
                          symbol="Y",
                          unit=Unit(m=1),
                          long_name=LongName(english="Y", 
                                             french="Y"),
                          gender=Gender(english=None, 
                                        french="f"),
                          )
NOTATIONS["d"] = Notation("d",
                          symbol="D",
                          unit=Unit(m=1),
                          long_name=LongName(english="Distance", 
                                             french="Distance"),
                          gender=Gender(english=None, 
                                        french="f"),
                          )
NOTATIONS["z"] = Notation("z",
                          symbol="z",
                          unit=Unit(m=1),
                          long_name=LongName(english="Altitude", 
                                             french="Altitude"),
                          gender=Gender(english=None, 
                                        french="f"),
                          )
NOTATIONS["t"] = Notation("t",
                          symbol="t",
                          unit=Unit(s=1),
                          long_name=LongName(english="Time", 
                                             french="Temps"),
                          gender=Gender(english=None, 
                                        french="m"),
                          )
NOTATIONS["xy"] = Notation("xy",
                           symbol="XY",
                           unit=Unit(m=2),
                           long_name=LongName(english="Surface", 
                                              french="Surface"),
                           gender=Gender(english=None, 
                                         french="f"),
                           )
NOTATIONS["h"] = Notation("h",
                          symbol="h",
                          unit=Unit(m=1),
                          long_name=LongName(english="Thickness", 
                                             french="Epaisseur"),
                          gender=Gender(english=None, 
                                        french="f"),
                          )
NOTATIONS["hvert"] = Notation("hvert",
                              symbol="h^v",
                              unit=Unit(m=1),
                              long_name=LongName(english="Vertical thickness", 
                                                 french="Epaisseur verticale"),
                              gender=Gender(english=None, 
                                            french="f"),
                              )
NOTATIONS["u"] = Notation("u",
                          symbol="u",
                          unit=Unit(m=1, s=-1),
                          long_name=LongName(english="Velocity", 
                                             french="Vitesse"),
                          gender=Gender(english=None, 
                                        french="f"),
                          )
NOTATIONS["hu"] = Notation("hu",
                           symbol="hu",
                           unit=Unit(m=2, s=-1),
                           long_name=LongName(english="Momentum", 
                                              french="Moment"),
                           gender=Gender(english=None, 
                                        french="m"),
                           )
NOTATIONS["hu2"] = Notation("hu2",
                            symbol="hu2",
                            unit=Unit(m=3, s=-2),
                            long_name=None,
                            gender=None,
                            )
NOTATIONS["ek"] = Notation("ek",
                           symbol="ek",
                           unit=Unit(J=1),
                           long_name=LongName(english="Kinetic energy", 
                                              french="Energie cinétique"),
                           gender=Gender(english=None, 
                                         french="f"),
                           )

NOTATIONS["max"] = Notation("max",
                            symbol="max",
                            unit=None,
                            long_name=LongName(english="Maximum", 
                                               french=dict(m="Maximum", f="Maximale")),
                            gender=None,
                            )
NOTATIONS["int"] = Notation("int",
                            symbol="int",
                            unit=None,
                            long_name=LongName(english="Integrate", 
                                               french=dict(m="Intégré", f="Intégrée")),
                            gender=None,
                            )


def get_notation(name: str, 
                 language: str = None
                 ) -> tilupy.notations.Notation:
    """Retrieve or construct a Notation object for a given name.

    This function attempts to fetch a predefined notation from the global :data:`tilupy.notations.NOTATIONS` 
    dictionary. If the notation is not found, it constructs a new :class:`tilupy.notations.Notation` object 
    based on the provided name. For composite names (e.g., "state_operator"), it recursively resolves the 
    state and operator, then combines them using :func:`tilupy.notations.add_operator`.

    Parameters
    ----------
    name : str
        The name of the notation to retrieve or construct.
        Can be a simple name (e.g., "velocity") or a composite name (e.g., "velocity_int_t").
    language : str, optional
        The language to use for the long name of the notation. If not provided, the default language from `LABEL_OPTIONS` is used.

    Returns
    -------
    tilupy.notations.Notation
        The retrieved or constructed :class:`tilupy.notations.Notation` object.
    """
    try:
        notation = NOTATIONS[name]
    except KeyError:
        strings = name.split("_")
        if len(strings) == 1:
            notation = Notation(name, 
                                symbol=name, 
                                unit=None, 
                                long_name=name, 
                                gender=None)
        else:
            state = get_notation(strings[0])
            operator = get_notation(strings[1])
            if len(strings) == 3:
                axis = strings[2]
            else:
                axis = None
            notation = add_operator(state, 
                                    operator, 
                                    axis=axis, 
                                    language=language)

    return notation


def get_operator_unit(name: str, 
                      axis: str
                      ) -> tilupy.notations.Unit:
    """Determine the unit associated with an operator based on its name and axis.

    This function returns a :class:`tilupy.notations.Unit` object corresponding to the operator's 
    name and the axis it operates on. For example, an integral operator ("int") has different units 
    depending on whether it integrates over time ("t") or space ("x", "y", "xy").

    Parameters
    ----------
    name : str
        The name of the operator (e.g., "int").
    axis : str or None
        The axis over which the operator acts (e.g., "t", "x", "y", "xy").
        If None, the default axis is used (e.g., "t" for "int").

    Returns
    -------
    tilupy.notations.Unit
        The unit associated with the operator and axis.
    """
    if name == "int":
        if axis == "t" or axis is None:
            unit = Unit(s=1)
        if axis in ["x", "y"]:
            unit = Unit(m=1)
        if axis == "xy":
            unit = Unit(m=2)
    else:
        unit = Unit(pd.Series(dtype="object"))
    return unit


def make_long_name(notation: tilupy.notations.Notation, 
                   operator: tilupy.notations.Notation, 
                   language: str = None
                   ) -> str:
    """Construct a long name for a notation combined with an operator.

    This function generates a readable long name by combining the long names of a notation and an operator.
    It can be written in French or English.

    Parameters
    ----------
    notation : tilupy.notations.Notation
        The base notation object.
    operator : tilupy.notations.Notation
        The operator notation object.
    language : str, optional
        The language to use for the long name. If not provided, the default language from 
        :data:`tilupy.notations.LABEL_OPTIONS` is used. By default None.

    Returns
    -------
    str
        The combined long name in the specified language.
    """
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


def add_operator(notation: tilupy.notations.Notation | str, 
                 operator: tilupy.notations.Notation | str, 
                 axis: str = None, 
                 language: str = None
                 ) -> tilupy.notations.Notation:
    """Combine a notation with an operator to create a new notation.

    This function constructs a new :class:`tilupy.notations.Notation` object by combining 
    a base notation with an operator. It handles the symbol, unit, and long name of the 
    resulting notation, taking into account the operator's axis (if any).

    Parameters
    ----------
    notation : tilupy.notations.Notation or str
        The base notation object or its name.
    operator : tilupy.notations.Notation or str
        The operator notation object or its name.
    axis : str, optional
        The axis over which the operator acts (e.g., "t", "x", "y"). By default None.
    language : str, optional
        The language to use for the long name. If not provided, the default language from 
        :data:`tilupy.notations.LABEL_OPTIONS` is used. By default None.

    Returns
    -------
    tilupy.notations.Notation
        The new Notation object resulting from the combination of the base notation and the operator.
    """
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

    res = Notation(name=notation.name + "_" + operator.name,
                   symbol=symbol,
                   unit=unit,
                   long_name=make_long_name(notation, operator, language=language),
                   )
    return res


def get_label(notation: tilupy.notations.Notation,
              with_unit: bool = True, 
              label_type: str = None, 
              language: str = None
              ) -> str:
    """Generate a formatted label for a notation.

    This function creates a label for a notation, either in literal form (long name) 
    or symbolic form (symbol). It can optionally include the unit in the label.

    Parameters
    ----------
    notation : tilupy.notations.Notation or str
        The notation object or its name.
    with_unit : bool, optional
        If True, the unit is included in the label. By default True.
    label_type : str, optional
        The type of label to generate: "litteral" (long name) or "symbol" (symbol).
        If not provided, the default label type from :data:`tilupy.notations.LABEL_OPTIONS` 
        is used. By default None.
    language : str, optional
        The language to use for the label. If not provided, the default language from 
        :data:`tilupy.notations.LABEL_OPTIONS` is used.
        By default None.

    Returns
    -------
    str
        The formatted label for the notation, optionally including the unit.
    """
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
    """Update the global label options.

    This function updates the global :data:`tilupy.notations.LABEL_OPTIONS` dictionary 
    with the provided keyword arguments. It allows dynamic configuration of label 
    generation settings, such as language and label type.

    Parameters
    ----------
    **kwargs : dict
        Key-value pairs to update in :data:`tilupy.notations.LABEL_OPTIONS`.
        Valid keys include "language" and "label_type".
    """
    global LABEL_OPTIONS
    LABEL_OPTIONS.update(**kwargs)


def readme_to_params(file: str, 
                     readme_param_match: dict = None
                     ) -> dict:
    """Convert a README file into a dictionary of parameters.

    This function reads a README file and extracts key-value pairs, optionally using a mapping dictionary
    to rename the keys. Each line in the file is expected to contain a key and a value separated by whitespace.

    Parameters
    ----------
    file : str
        Path to the README file to read.
    readme_param_match : dict, optional
        A dictionary mapping keys in the README file to new keys in the output dictionary.
        If not provided, the keys are used as-is. By default None.

    Returns
    -------
    dict
        A dictionary of parameters extracted from the README file.
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


def make_rheol_string_fmt(rheoldict: dict) -> str:
    """Generate a formatted string template for rheological parameters.

    This function constructs a string template for rheological parameters based on the provided dictionary and rheological law.
    The template includes placeholders for parameter values, which can later be formatted with specific values.

    Parameters
    ----------
    rheoldict : dict
        A dictionary of rheological parameters (e.g., "delta1", "ksi").
    
    Returns
    -------
    str
        A formatted string template for the rheological parameters.
    """
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


def make_rheol_string(rheoldict: dict, 
                      law: str
                      ) -> str | list[str]:
    """Generate formatted strings for rheological parameters.

    This function creates formatted strings for rheological parameters based on the provided dictionary and rheological law.
    It handles multiple parameter sets (e.g., for different time steps or conditions) and formats each set according to the specified law.

    Parameters
    ----------
    rheoldict : dict
        A dictionary of rheological parameters. Values can be lists of parameters for multiple sets.
    law : str
        The rheological law to use. Can be "coulomb", "voellmy" or "pouliquen_2002".

    Returns
    -------
    str or list[str]
        The formatted string(s) for the rheological parameters.
        If there is only one parameter set, a single string is returned. Otherwise, a list of strings is returned.
    """
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
