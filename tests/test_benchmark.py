import pytest
import os as os

import numpy as np
import pandas as pd
import tilupy.benchmark as tibench
import tilupy.analytic_sol as tiAS
import tilupy.read as tiread

@pytest.fixture
def preload_benchmark():
    case_error = tibench.Benchmark() 
    case_error.load_numerical_result("lave2D", 
                                     folder=os.path.join(os.path.dirname(__file__), 
                                                         "data", 
                                                         "gray99", 
                                                         "lave2D"), 
                                     name="projects", 
                                     raster="toposimu.asc")
    case_working = tibench.Benchmark()
    case_working.load_numerical_result("shaltop", 
                                       file_params="params.txt", 
                                       folder=os.path.join(os.path.dirname(__file__), 
                                                           "data", 
                                                           "gray99",
                                                           "shaltop"))
    case_working.load_numerical_result("lave2D", 
                                       folder=os.path.join(os.path.dirname(__file__), 
                                                           "data", 
                                                           "gray99", 
                                                           "lave2D"), 
                                       name="projects", 
                                       raster="toposimu.asc")
    res = dict(error=case_error,
               working=case_working,)
    
    return res

def test_error_load_numerical_result(preload_benchmark):
    case_error = preload_benchmark["error"]
    
    with pytest.raises(ValueError):  # Invalid model name
        case_error.load_numerical_result("sheltopz")

    with pytest.raises(ValueError): # Different simulations loaded        
        case_error.load_numerical_result("shaltop", 
                                         file_params="delta_25p00.txt", 
                                         folder=os.path.join(os.path.dirname(__file__), 
                                                             "data", 
                                                             "frankslide", 
                                                             "shaltop"))
        

def test_error_show_output(preload_benchmark):
    case_error = preload_benchmark["error"]
    
    with pytest.raises(ValueError):  # Model not loaded
        case_error.show_output("h", "shaltop")


def test_error_show_output_profile(preload_benchmark):
    case_error = preload_benchmark["error"]
    
    with pytest.raises(ValueError):  # Model not loaded
        case_error.show_output_profile("h", "shaltop")


def test_error_show_comparison_temporal1D(preload_benchmark):
    case_error = preload_benchmark["error"]
    
    with pytest.raises(ValueError):  # Model not loaded
        case_error.show_comparison_temporal1D("h", ["lave2D", "shaltop"])

@pytest.mark.parametrize(
    "time",
    [
        (None),
        (0.0),
        ([1.0, 2.0]),
    ],
)
def test_show_comparison_temporal1D(preload_benchmark, time):
    working_case = preload_benchmark["working"]
    
    ax = working_case.show_comparison_temporal1D(output="h", models=["shaltop", "lave2D"], time_steps = time, show_plot=False)
    assert isinstance(ax, np.ndarray)

@pytest.mark.parametrize(
    "args, expected",
    [
        ("h", tiread.TemporalResults2D),
        ("hu", tiread.TemporalResults2D),
        ("u_max", tiread.StaticResults2D),
        ("h_max_x", tiread.TemporalResults1D),
        ("u_min_xy", tiread.TemporalResults0D),
    ],
)
def test_get_avrg_result(preload_benchmark, args, expected):
    working_case = preload_benchmark["working"]
    
    res = working_case.get_avrg_result(args)
    assert isinstance(res, expected)
    

def test_compute_area(preload_benchmark):
    working_case = preload_benchmark["working"]
    
    area_surf, area_num = working_case.compute_area()
    assert isinstance(area_surf, dict)
    assert isinstance(area_num, dict)


def test_compute_impacted_area(preload_benchmark):
    working_case = preload_benchmark["working"]
    
    area_surf, area_num = working_case.compute_impacted_area()
    assert isinstance(area_surf, dict)
    assert isinstance(area_num, dict)
    

def test_compute_impacted_area_rms_from_avrg(preload_benchmark):
    working_case = preload_benchmark["working"]
    
    area_rms, avrg_area = working_case.compute_impacted_area_rms_from_avrg()
    assert isinstance(area_rms, dict)
    assert isinstance(avrg_area, np.ndarray)


@pytest.mark.parametrize(
    "output",
    [
        ("h"),
        ("hu"),
        ("u_max"),
        ("h_max_x"),
        ("u_min_xy"),
    ],
)
def test_compute_rms_from_avrg(preload_benchmark, output):
    working_case = preload_benchmark["working"]
    
    output_dict = working_case.compute_rms_from_avrg(output)
    assert isinstance(output_dict, dict)


def test_compute_dist_centermass(preload_benchmark):
    working_case = preload_benchmark["working"]
    
    init_coord_centermass, model_max_dist = working_case.compute_impacted_area()
    assert isinstance(init_coord_centermass, dict)
    assert isinstance(model_max_dist, dict)


def test_compute_average_velocity(preload_benchmark):
    working_case = preload_benchmark["working"]
    
    model_avrg_vel, model_time, distance, model_pos = working_case.compute_average_velocity()
    assert isinstance(model_avrg_vel, dict)
    assert isinstance(model_time, dict)
    assert isinstance(distance, float)
    assert isinstance(model_pos, dict)


def test_compute_rms_from_coussot(preload_benchmark):
    working_case = preload_benchmark["working"]
    
    output_rms, model_front_pos, model_coussot = working_case.compute_rms_from_coussot(coussot_params=({"rho": 2000, 
                                                                                                        "tau": 1000, 
                                                                                                        "theta": 0, 
                                                                                                        "H_size": 100}))
    assert isinstance(output_rms, dict)
    assert isinstance(model_front_pos, dict)
    assert isinstance(model_coussot, dict)
    
    
def test_generate_simulation_comparison_csv(preload_benchmark):
    working_case = preload_benchmark["working"]
    
    csv = working_case.generate_simulation_comparison_csv()
    assert isinstance(csv, pd.DataFrame)


def test_generate_analytical_comparison_csv(preload_benchmark):
    working_case = preload_benchmark["working"]
    
    csv = working_case.generate_analytical_comparison_csv(analytic_solution=dict({"solution": tiAS.Ritter_dry,
                                                                                  "h_0": 5, 
                                                                                  "x_0": 100}))
    assert isinstance(csv, pd.DataFrame)