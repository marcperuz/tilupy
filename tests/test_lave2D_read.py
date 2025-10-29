import pytest
import os

import tilupy.models.lave2D.read as laread
import tilupy.read as tiread

@pytest.fixture
def simu_data():
    res = dict(folder=os.path.join(os.path.dirname(__file__), "data", "gray99", "lave2D"),
               simu_name="projects",
               raster_name="toposimu",
               )
    return res


@pytest.mark.parametrize(
    "args, expected",
    [
        ("h", (3, tiread.TemporalResults2D)),
        ("hvert", (3, tiread.TemporalResults2D)),
        ("u", (3, tiread.TemporalResults2D)),
        ("hu", (3, tiread.TemporalResults2D)),
        ("hu2", (3, tiread.TemporalResults2D)),
    ],
)


def test_extract_output(simu_data, args, expected):
    res = laread.Results(folder=simu_data["folder"], 
                         name=simu_data["simu_name"], 
                         raster=simu_data["raster_name"])
    output = res._extract_output(args)
    
    assert output.d.ndim == expected[0]
    assert isinstance(output, expected[1])
    

def test_extract_invalid_output(simu_data):
    res = laread.Results(folder=simu_data["folder"], 
                         name=simu_data["simu_name"], 
                         raster=simu_data["raster_name"])
    output = res._extract_output("xx")
    
    assert isinstance(output, tiread.AbstractResults)