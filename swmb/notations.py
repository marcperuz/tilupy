#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 12:20:34 2021

@author: peruzzetto
"""


def make_rheol_string(rheoldict):
    """Make string from rheological parameters."""
    text = ''
    for name in ['delta1', 'delta2', 'delta3', 'delta4']:
        if name in rheoldict:
            new_txt = name + '_{:05.2f}_'.format(rheoldict[name])
            text += new_txt.replace('.', 'p')
    if 'ksi' in rheoldict:
        new_txt = name + '_{:06.1f}_'.format(rheoldict[name])
        text += new_txt.replace('.', 'p')
    text = text[:-1]
    return text
