#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 12:20:34 2021

@author: peruzzetto
"""

LABELS = dict(h='Thickness (m)',
              hmax='Maximum thickness (m)',
              hfinal='Final thickness (m)',
              u='Velocity (m s$^{-1}$)')


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
    with open(file, 'r') as f:
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


def make_rheol_string_fmt(rheoldict):
    """Make string from rheological parameters."""
    text = ''
    for name in ['delta1', 'delta2', 'delta3', 'delta4']:
        if name in rheoldict:
            new_txt = name + '_{:05.2f}_'
            text += new_txt
    if 'ksi' in rheoldict:
        new_txt = name + 'ksi_{:06.1f}_'
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
        if law == 'coulomb':
            text = txt_fmt.format(rheoldict['delta1'][i]).replace('.', 'p')
        if law == 'voellmy':
            text = txt_fmt.format(rheoldict['delta1'][i],
                                  rheoldict['ksi'][i]).replace('.', 'p')
        texts.append(text)

    if nparams == 1:
        texts = texts[0]

    return texts
