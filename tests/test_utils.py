import pytest
import numpy as np
import tilupy.utils


@pytest.fixture
def axis():
    x = np.linspace(0, 1, 5)
    y = np.flip(np.linspace(0, 2, 10))
    return dict(x=x, y=y)


@pytest.mark.parametrize(
    "ax, ay",
    [(0.5, 0.5), (0.5, -1), (-1, 0.5)],
)
def test_normal_vector(axis, ax, ay):
    z = ax * axis["x"][np.newaxis, :] + ay * axis["y"][:, np.newaxis]
    nn = np.sqrt(1 + ax**2 + ay**2)
    nx = -ax / nn
    ny = -ay / nn
    nz = 1 / nn
    Nx, Ny, Nz = tilupy.utils.normal_vector(axis["x"], axis["y"], z)
    assert Nx.shape == Ny.shape == Nz.shape == (len(axis["y"]), len(axis["x"]))
    assert np.allclose(Nx, nx)
    assert np.allclose(Ny, ny)
    assert np.allclose(Nz, nz)
