# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 16:19:18 2023

@author: peruzzetto
"""

import pytest
import os

@pytest.fixture
def folder_data(request):
    folder_tests = os.path.dirname(request.module.__file__)
    return os.path.join(folder_tests, 'data')
    
    