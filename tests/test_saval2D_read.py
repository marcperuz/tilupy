import pytest
import os

import tilupy.models.saval2D.read as saread
import tilupy.read as tiread

@pytest.fixture
def simu_data():
    res = dict(folder=os.path.join(os.path.dirname(__file__), "data", "gray99", "saval2D"),
               raster_name="mntsimulation",
               )
    return res


@pytest.mark.parametrize(
    "args, expected",
    [
        ("h", (3, tiread.TemporalResults2D)),
        ("u", (3, tiread.TemporalResults2D)),
        ("hu", (3, tiread.TemporalResults2D)),
        ("hu2", (3, tiread.TemporalResults2D)),
        ("ux", (3, tiread.TemporalResults2D)),
        ("uy", (3, tiread.TemporalResults2D)),
        ("hvert", (3, tiread.TemporalResults2D)),
    ],
)


def test_extract_output(simu_data, args, expected):
    res = saread.Results(folder=simu_data["folder"],
                         raster_topo=simu_data["raster_name"])
    output = res._extract_output(args)
    
    assert output.d.ndim == expected[0]
    assert isinstance(output, expected[1])
    

def test_extract_invalid_output(simu_data):
    res = saread.Results(folder=simu_data["folder"],
                         raster_topo=simu_data["raster_name"])
    output = res._extract_output("xx")
    
    assert isinstance(output, tiread.AbstractResults)