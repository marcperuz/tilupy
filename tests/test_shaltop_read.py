import pytest
import os

import numpy

import tilupy.models.shaltop.read as sharead
import tilupy.read as tiread

@pytest.fixture
def simu_data():
    res = dict(folder=os.path.join(os.path.dirname(__file__), "data", "gray99", "shaltop"),
               param_file="params.txt",
               )
    return res


def test_read_param(simu_data):
    param_file = os.path.join(simu_data["folder"], simu_data["param_file"])
    di = sharead.read_params(param_file)
    assert list(di.keys()) == ["nx", "ny", "per", "pery", "tmax", "dt_im", "initz", "file_z_init", "ipr", "file_m_init", "folder_output", "icomp", "tau_density", "K_tau"]
    

def test_read_bin(simu_data):
    bin_file = os.path.join(simu_data["folder"], "data2", "ekmax.bin")
    data = sharead.read_file_bin(bin_file, 236, 68)
    assert isinstance(data, numpy.ndarray)
    
    
def test_read_init(simu_data):
    init_file = os.path.join(simu_data["folder"], "z.d")
    data = sharead.read_file_init(init_file, 236, 68)
    assert isinstance(data, numpy.ndarray)
    

@pytest.mark.parametrize(
    "args, expected",
    [
        ("h", (3, tiread.TemporalResults2D)),
        ("ux", (3, tiread.TemporalResults2D)),
        ("uy", (3, tiread.TemporalResults2D)),
        ("hvert", (3, tiread.TemporalResults2D)),
        ("u", (3, tiread.TemporalResults2D)),
        ("hu", (3, tiread.TemporalResults2D)),
        ("hu2", (3, tiread.TemporalResults2D)),
        ("ek", (1, tiread.TemporalResults0D)),
        ("ep", (1, tiread.TemporalResults0D)),
        ("etot", (1, tiread.TemporalResults0D)),
        ("faccx", (3, tiread.TemporalResults2D)),
        ("fcurvx", (3, tiread.TemporalResults2D)),
        ("ffricx", (3, tiread.TemporalResults2D)),
        ("fgravx", (3, tiread.TemporalResults2D)),
        ("finertx", (3, tiread.TemporalResults2D)),
        ("fpressionx", (3, tiread.TemporalResults2D)),
        ("shearx", (3, tiread.TemporalResults2D)),
        ("normalx", (3, tiread.TemporalResults2D)),
        ("pbottom", (3, tiread.TemporalResults2D)),
        ("pcorrdt", (3, tiread.TemporalResults2D)),
        ("pcorrdiv", (3, tiread.TemporalResults2D)),
    ],
)


def test_extract_output(simu_data, args, expected):
    res = sharead.Results(simu_data["folder"], 
                          file_params=simu_data["param_file"])
    output = res._extract_output(args)
    
    assert output.d.ndim == expected[0]
    assert isinstance(output, expected[1])
    
    
def test_extract_invalid_output(simu_data):
    res = sharead.Results(simu_data["folder"], 
                          file_params=simu_data["param_file"])
    output = res._extract_output("xx")
    
    assert isinstance(output, tiread.AbstractResults)