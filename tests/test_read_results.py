# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 23:29:00 2024

@author: peruzzetto
"""

import pytest
import os

import tilupy.read as tiread
import tilupy.cmd


@pytest.fixture
def simu_data():
    res = dict(
        simu_name="frankslide",
        param_file="delta_25p00.txt",
        h_thresh=0.1,
        n_im=3,
    )
    return res


@pytest.mark.parametrize(
    "args, expected",
    [
        (("shaltop", "h"), (3, tiread.TemporalResults2D)),
        (("shaltop", "u"), (3, tiread.TemporalResults2D)),
        (("shaltop", "h_max"), (2, tiread.StaticResults)),
        (("shaltop", "h_int"), (2, tiread.StaticResults)),
        (("shaltop", "h_int_x"), (2, tiread.TemporalResults1D)),
        (("shaltop", "h_int_xy"), (1, tiread.TemporalResults0D)),
    ],
)
def test_get_output(folder_data, simu_data, args, expected):
    folder_simus = os.path.join(folder_data, simu_data["simu_name"], args[0])
    res = tiread.get_results(
        args[0], file_params=simu_data["param_file"], folder=folder_simus
    )
    # pytest.set_trace()
    output = res.get_output(args[1])
    assert output.d.ndim == expected[0]
    assert isinstance(output, expected[1])
    

@pytest.mark.parametrize(
    "args, expected",
    [
        (("shaltop", "h"), (2, tiread.TemporalResults1D)),
        (("shaltop", "u"), (2, tiread.TemporalResults1D)),
        (("shaltop", "h_max"), (1, tiread.StaticResults1D)),
    ],
)
def test_get_profile(folder_data, simu_data, args, expected):
    folder_simus = os.path.join(folder_data, simu_data["simu_name"], args[0])
    res = tiread.get_results(
        args[0], file_params=simu_data["param_file"], folder=folder_simus
    )
    # pytest.set_trace()
    output, _ = res.get_profile(args[1])
    assert output.d.ndim == expected[0]
    assert isinstance(output, expected[1])


@pytest.mark.parametrize(
    "args, expected",
    [
        (("shaltop", "h_init"), (2, tiread.StaticResults2D)),
        (("shaltop", "h_max"), (2, tiread.StaticResults2D)),
        (("shaltop", "h_int"), (2, tiread.StaticResults2D)),
        (("shaltop", "h_int_x"), (2, tiread.TemporalResults1D)),
        (("shaltop", "hvert_int_xy"), (1, tiread.TemporalResults0D)),
        (("shaltop", "shearx_int"), (2, tiread.StaticResults2D)),
    ],
)
def test_plot(folder_data, folder_plots, simu_data, args, expected):
    folder_simus = os.path.join(folder_data, simu_data["simu_name"], args[0])
    folder_output = os.path.join(folder_plots, simu_data["simu_name"], args[0])
    file_out = os.path.join(folder_output, args[1] + ".png")
    if os.path.isfile(file_out):
        os.remove(file_out)
    os.makedirs(folder_output, exist_ok=True)
    res = tiread.get_results(
        args[0], file_params=simu_data["param_file"], folder=folder_simus
    )
    # Test generation of ouput and plot
    output = res.get_output(args[1])
    axe = output.plot()
    axe.figure.savefig(file_out)
    assert output.d.ndim == expected[0]
    assert isinstance(output, expected[1])


@pytest.mark.parametrize(
    "args, expected",
    [
        (("shaltop", "h_init"), None),
        (("shaltop", "h_max"), None),
        (("shaltop", "h_int"), None),
        (("shaltop", "h_int_x"), None),
        (("shaltop", "hvert_int_xy"), None),
        (("shaltop", "shearx_int"), None),
    ],
)
def test_plot_Results(folder_data, folder_plots, simu_data, args, expected):
    folder_simus = os.path.join(folder_data, simu_data["simu_name"], args[0])
    folder_output = os.path.join(folder_plots, simu_data["simu_name"], args[0])
    file_out = os.path.join(folder_output, args[1] + "_from_res.png")
    if os.path.isfile(file_out):
        os.remove(file_out)
    os.makedirs(folder_output, exist_ok=True)
    res = tiread.get_results(
        args[0], file_params=simu_data["param_file"], folder=folder_simus
    )
    # Test generation of ouput and plot
    res.plot(args[1], folder_out=folder_output, file_suffix="from_res")
    assert os.path.isfile(file_out)


@pytest.mark.parametrize(
    "args, expected",
    [
        (("shaltop", ["h", "u"]), None),
    ],
)
def test_plot_cmd(folder_data, folder_plots, simu_data, args, expected):
    folder_simus = os.path.join(folder_data, simu_data["simu_name"], args[0])
    folder_res = os.path.join(
        folder_plots,
        simu_data["simu_name"],
        args[0],
        "cmd",
    )
    os.makedirs(folder_res, exist_ok=True)
    for state in args[1]:
        tilupy.cmd.plot_results(
            "shaltop",
            state,
            simu_data["param_file"],
            folder_simus,
            save=True,
            folder_out=folder_res,
            display_plot=False,
            h_thresh=simu_data["h_thresh"],
            figsize=(10 / 2.54, 10 / 2.54),
        )

@pytest.fixture
def simu_data_lave2D():
    res = dict(folder=os.path.join(os.path.dirname(__file__), "data", "gray99", "lave2D"),
               simu_name="projects",
               raster_name="toposimu",
               )
    return res

@pytest.mark.parametrize(
    "args, expected",
    [
        ("h_init", (2, tiread.StaticResults2D)),
        ("h_max", (2, tiread.StaticResults2D)),
        ("h_int", (2, tiread.StaticResults2D)),
        ("h_int_x", (2, tiread.TemporalResults1D)),
        ("hvert_int_xy", (1, tiread.TemporalResults0D)),
    ],
)
def test_get_output_lave2D(simu_data_lave2D, args, expected):
    res = tiread.get_results("lave2D",
                             folder=simu_data_lave2D["folder"], 
                             name=simu_data_lave2D["simu_name"], 
                             raster=simu_data_lave2D["raster_name"])
    
    output = res.get_output(args)
    
    assert output.d.ndim == expected[0]
    assert isinstance(output, expected[1])
    
@pytest.fixture
def simu_data_saval2D():
    res = dict(folder=os.path.join(os.path.dirname(__file__), "data", "gray99", "saval2D"),
               raster_name="mntsimulation",
               )
    return res

@pytest.mark.parametrize(
    "args, expected",
    [
        ("h_init", (2, tiread.StaticResults2D)),
        ("h_max", (2, tiread.StaticResults2D)),
        ("h_int", (2, tiread.StaticResults2D)),
        ("h_int_x", (2, tiread.TemporalResults1D)),
        ("hvert_int_xy", (1, tiread.TemporalResults0D)),
    ],
)
def test_get_output_saval2D(simu_data_saval2D, args, expected):
    res = tiread.get_results("saval2D",
                             folder=simu_data_saval2D["folder"],
                             raster_topo=simu_data_saval2D["raster_name"])
    
    output = res.get_output(args)
    
    assert output.d.ndim == expected[0]
    assert isinstance(output, expected[1])