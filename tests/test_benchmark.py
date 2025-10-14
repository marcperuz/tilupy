import pytest
import os as os
import matplotlib

import tilupy.benchmark as tibench

case_error = tibench.Benchmark()
case_error.load_numerical_result("shaltop", file_params="params.txt", folder=os.path.join(os.path.dirname(__file__), "data", "gray99", "shaltop"))
case_error.extract_profiles("h", "shaltop", 0.0)
case_error.extract_profiles("u", "shaltop", 0.0)
case_error.extract_field("h", "shaltop", 0.0)
case_error.extract_field("u", "shaltop", 0.0)

def test_error_load_numerical_result():
    with pytest.raises(ValueError): 
        case_error.load_numerical_result("sheltopz") # Invalid model name
    
def test_error_extract_invalid_model(): # Invalid model name
    with pytest.raises(ValueError): 
        case_error.extract_profiles("h", "sheltopz")
    with pytest.raises(ValueError): 
        case_error.extract_field("h", "sheltopz") 
        
def test_error_extract_invalid_output(): # Invalid output name
    with pytest.raises(ValueError):     
        case_error.extract_profiles("xx", "shaltop") 
    with pytest.raises(ValueError): 
        case_error.extract_field("xx", "shaltop")

def test_error_invalid_time_step(): # Invalid time step
    with pytest.raises(ValueError): 
        case_error.extract_profiles("h", "shaltop", 99.0)
    with pytest.raises(ValueError): 
        case_error.extract_field("h", "shaltop", 99.0)
    
def test_error_show_profiles():
    with pytest.raises(ValueError): 
        case_error.show_profile("h", "as") # No AS computed
    with pytest.raises(ValueError): 
        case_error.show_profile("h", "sheltopz") # Invalid model
    with pytest.raises(ValueError): 
        case_error.show_profile("xx", "shaltop") # Invalid output
    with pytest.raises(ValueError): 
        case_error.show_profile("h", "shaltop", "Z") # Invalid axis
    
def test_error_show_field():
    with pytest.raises(ValueError): 
        case_error.show_field("h", "sheltopz", 0.0) # Invalid model
    with pytest.raises(ValueError): 
        case_error.show_field("xx", "shaltop", 0.0) # Invalid output
    with pytest.raises(ValueError): 
        case_error.show_field("h", "shaltop", 99.0) # Invalid time step

# def test_error_show_compare():
#     with pytest.raises(ValueError): 
#         case_error.show_height_profile_comparison(["shaltop"], 99.0) # Invalid time step
#     with pytest.raises(ValueError): 
#         case_error.show_height_profile_comparison(["shaltop"], time_step=0.0, axis = "Z") # Invalid axis
#     with pytest.raises(ValueError): 
#         case_error.show_height_profile_comparison(["shaltop"], time_step=0.0, plot_as=True) # No AS computed
    
#     with pytest.raises(ValueError): 
#         case_error.show_velocity_profile_comparison(["shaltop"], 99.0) # Invalid time step
#     with pytest.raises(ValueError): 
#         case_error.show_velocity_profile_comparison(["shaltop"], time_step=0.0, axis = "Z") # Invalid axis
#     with pytest.raises(ValueError): 
#         case_error.show_velocity_profile_comparison(["shaltop"], time_step=0.0, plot_as=True) # No AS computed
    
working_case = tibench.Benchmark()
working_case.load_numerical_result("shaltop", file_params="params.txt", folder=os.path.join(os.path.dirname(__file__), "data", "gray99","shaltop"))
working_case.load_numerical_result("lave2D", folder=os.path.join(os.path.dirname(__file__), "data", "gray99", "lave2D"), name="projects", raster="toposimu.asc")
working_case.extract_profiles("h", "shaltop", 0.0)
working_case.extract_profiles("u", "shaltop", 0.0)
working_case.extract_profiles("h", "lave2D", 1.0)
working_case.extract_profiles("u", "lave2D", 1.0)

def test_show_fields():
    ax = working_case.show_field("h", "shaltop", 0.0, show_plot=False)
    assert isinstance(ax, matplotlib.axes._axes.Axes)
    
    ax = working_case.show_field("u", "shaltop", 0.0, show_plot=False)
    assert isinstance(ax, matplotlib.axes._axes.Axes)
    
def test_show_profiles():
    ax = working_case.show_profile("h", "shaltop", time_steps=0.0, show_plot=False)
    assert isinstance(ax, matplotlib.axes._axes.Axes)
    
    ax = working_case.show_profile("u", "shaltop", time_steps=0.0, show_plot=False)
    assert isinstance(ax, matplotlib.axes._axes.Axes)

# def test_show_profiles_compare():
#     ax = working_case.show_height_profile_comparison(["shaltop", "lave2D"], time_step=1.0, show_plot=False)
#     assert isinstance(ax, matplotlib.axes._axes.Axes)

#     ax = working_case.show_velocity_profile_comparison(["shaltop", "lave2D"], time_step=1.0, show_plot=False)
#     assert isinstance(ax, matplotlib.axes._axes.Axes)