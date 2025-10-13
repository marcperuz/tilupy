# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 12:09:41 2023

@author: peruzzetto
"""
from typing import Callable

import math as math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import tilupy.read
# import tilupy.analytic_sol as AS
# import pytopomap.plot as pyplt


class Benchmark:
    """Benchmark of simulation results.
    
    This class groups together all the methods for processing and analyzing 
    the results of various simulation models, allowing, among other things, 
    the comparison of results between models.
    
    Global Attributes:
    ------------------
        _allowed_models : list[str]
            List of implemented models in tilupy. 
            For now, this list allows reading the results of:
            "shaltop", "lave2D", "saval2D".
        _current_model : str
            Last model loaded in the class.
        _loaded_result : list[tilupy.read.Results]
            Last result of :attr:`_current_model` loaded in the class.
        _allowed_extracting_outputs : list[str]
            List of allowed outputs : "h", "u", "ux", "uy".
        _x : numpy.ndarray
            X-coordinates of the simulation.
        _y : numpy.ndarray
            Y-coordinates of the simulation.
        _z : numpy.ndarray
            Initial topographic elevation of the simulation.
    
    Extracted Attributes:
    ---------------------
        These attributes correspond to dictionaries storing the results extracted from model simulations.
        Their name is constructed as follows:
        
        :data:`_{property}_num_{dimension}_{axis} = (model_name, list_values)`
        
        :data:`property`:
            - :data:`h` for the fluid thickness (normal to the surface)
            - :data:`u` for the fluid velocity (norm)
            - :data:`ux` for the X component of the fluid velocity
            - :data:`uy` for the Y component of the fluid velocity
        
        :data:`dimension`:
            - :data:`1d` is for profile extraction. It is associated with an axis according 
              to the profile that was extracted and with a parameters dictionary containing the
              extraction details of the profiles (indexes of the extracted profiles) and the 
              indexes of the maximal lateral extension for each profiles. The parameters dictionary
              is constructed like so :
            
              :data:`_{property}_num_1d_params[time_step] = (model_name, direction_index, list_limit_index)`

            - :data:`2d` is for surface extraction.
        
    Computed Attributes:
    --------------------
        These attributes correspond to list storing the results of computed analytical solutions, constructed as follows:
        
        :data:`_{property}_as_1d = [(time_step, list_values)]`

        :data:`property`:
            - :data:`h` for the fluid thickness (normal to the surface)
            - :data:`u` for the fluid velocity (norm)
    """
    def __init__(self):
        self._allowed_models = ["shaltop", "lave2D", "saval2D"]
        
        self._current_model = None
        self._loaded_results = {}
        self._x, self._y, self._z = None, None, None
        
        self._allowed_extracting_outputs = ["h", "u", "ux", "uy"]
        
        for var in self._allowed_extracting_outputs:
            for suffix in ["num_1d_X", "num_1d_Y", "num_1d_params", "num_2d"]:
                setattr(self, f"_{var}_{suffix}", {})
        
        as_vars = ["h", "u"]
        for var in as_vars:
            setattr(self, f"_{var}_as_1d", [])
    
    
    def load_numerical_result(self, 
                              model: str, 
                              **kwargs,
                              ) -> None:
        """Load numerical simulation results using :func:`tilupy.read.get_results` for a given model.

        Parameters
        ----------
        model : str
            Name of the model to load. Must be one of the :attr:`_allowed_models`.
        **kwargs
            Keyword arguments passed to the result reader for the specific model.

        Raises
        ------
        ValueError
            If size of stored :attr:`_x`, :attr:`_y` are different with new loaded ones.
        UserWarning
            If stored :attr:`_x`, :attr:`_y` or :attr:`_z` are different with new loaded ones.
        ValueError
            If the provided model is not in the allowed list.
        
        Notes
        -----
        Store x, y and z property of the result in :attr:`_x`, :attr:`_y`, :attr:`_z`. The result (instance of :class:`tilupy.read.Results`) is  
        store in :attr:`_loaded_results`.
        """
        if model in self._allowed_models:
            self._current_model = model
            if model not in self._loaded_results:
                self._loaded_results[model] = tilupy.read.get_results(model, **kwargs)
            if self._x is None and self._y is None and self._z is None:
                self._x, self._y, self._z = self._loaded_results[model].x, self._loaded_results[model].y, self._loaded_results[model].z
            
            if len(self._x) != len(self._loaded_results[model].x):
                raise ValueError("NX size not the same with previous result loaded.")
            if len(self._y) != len(self._loaded_results[model].y):
                raise ValueError("NY size not the same with previous result loaded.")
            
            try :
                if not np.allclose(self._x, self._loaded_results[model].x, rtol=0.1, atol=0.1):
                    mean_error = np.mean(self._x - self._loaded_results[model].x)
                    self._x = self._loaded_results[model].x
                    raise UserWarning(f"Stored X coordinates different from loaded ones; average error: {mean_error}")
            except UserWarning as w:
                print(f"[WARNING] {w}")
            
            try :
                if not np.allclose(self._y, self._loaded_results[model].y, rtol=0.1, atol=0.1):
                    mean_error = np.mean(self._y - self._loaded_results[model].y)
                    self._y = self._loaded_results[model].y
                    raise UserWarning(f"Stored Y coordinates different from loaded ones; average error: {mean_error}")
            except UserWarning as w:
                print(f"[WARNING] {w}")
            
            try :
                if not np.allclose(self._z, self._loaded_results[model].z, rtol=0.1, atol=0.1):
                    mean_error = np.mean(self._z - self._loaded_results[model].z)
                    self._z = self._loaded_results[model].z
                    raise UserWarning(f"Stored elevation values different from loaded ones; average error: {mean_error}")
            except UserWarning as w:
                print(f"[WARNING] {w}")
                
        else:
            raise ValueError(f" -> No correct model selected, choose between:\n       {self._allowed_models}")
    
       
    def compute_analytic_solution(self,
                                  output: str,
                                  model: Callable, 
                                  T: float | list[float], 
                                  **kwargs
                                  ) -> None:
        """
        Compute the analytic solution for the given time steps using the provided model.

        Parameters
        ----------
        output : str
            Wanted output for the analytical solution. Can be : "h" or "u".
        model : Callable
            Callable object representing the analytic solution model (model from :class:`tilupy.analytic_sol.Depth_results`)
        T : float | list[float]
            Time or list of times at which to compute the analytic solution.
        **kwargs
            Keyword arguments passed to the analytic solution for the specific model.
        """
        solution = model(**kwargs)
        
        if isinstance(T, float):
            T = [T]
        
        if output not in ['h', 'u']:
            raise ValueError(" -> Available ouput : 'h', 'u'.")
        
        if output == 'h':
            solution.compute_h(self._x, T)
            
            if solution.h is not None:
                for i in range(len(T)):
                    # self._h_as_1d.append((T[i], solution.h[i]))
                    self._h_as_1d.append((T[i], tilupy.read.StaticResults1D(name=output,
                                                                            d=solution.h[i],
                                                                            coords=self._x,
                                                                            coords_name='x')))
        
        if output == 'u':
            solution.compute_u(self._x, T)
            
            if solution.u is not None:
                for i in range(len(T)):
                    # self._u_as_1d.append((T[i], solution.u[i]))
                    self._u_as_1d.append((T[i], tilupy.read.StaticResults1D(name=output,
                                                                            d=solution.u[i],
                                                                            coords=self._x,
                                                                            coords_name='x')))
    
    
    def process_data_field(self, 
                           model: str, 
                           field_name: str, 
                           storage_dict_X: dict, 
                           storage_dict_Y: dict, 
                           storage_params: dict, 
                           t: float = None,
                           direction_index: str | list[float] = None, 
                           flow_threshold: float = 1e-3,
                           show_message: bool = False,
                           ) -> str | list[float]:
        """Process data field to extract profiles.

        Parameters
        ----------
        model : str
            Model to extract the profile.
        field_name : str
            Data field. Can be: "h", "u", "ux", "uy".
        storage_dict_X : dict
            Dictionary to save the X axis profile. 
        storage_dict_Y : dict
            Dictionary to save the Y axis profile.
        storage_params : dict
            Dictionary to save the parameters of the profiles.
        t : float, optional
            Time index to extract. If None, uses the last available time step. By default None.
        direction_index : list[float] or str, optional
            Index along each axis to extract the profile from: (X, Y). Depending on the information given, the extract method change:
                - list[float]: position (in meter) of the wanted profiles.
                - str: "max" for finding the best profiles, based on the farthest flow front and the position of maximum fluid height.
                - None: select the medians.
            By default None.
        flow_threshold : float, optional
            Minimum velocity to consider as part of the flow, by default 1e-3.
        show_message : bool, optional
            If True, print the indexes and the front positions saved, by default False

        Raises
        ------
        UserWarning
            If the output variable is not accessible from the model.
        ValueError
            If invalid time step.
        
        Returns
        -------
        str | list[float]
            direction_index
        """
        def find_max_index_along_axis(matrix: np.ndarray, axis: int, threshold: float):
            """Find the best profile index based on the farthest flow front and the position of maximum fluid data.

            Parameters
            ----------
            matrix : np.ndarray
                Data field
            axis : int
                Axis to look at (0: X, 1: Y).
            threshold : float
                Minimum value considered.

            Returns
            -------
            int
                Index of the selected profile.
            """
            max_front_idx = 0
            list_with_front = []

            for i in range(matrix.shape[1 - axis]):
                profile = matrix[:, i] if axis == 0 else matrix[i, :]
                idx_max = np.argmax(profile)
                sub_profile = profile[idx_max:]
                idx = np.where(sub_profile < threshold)[0]

                if len(idx) == 0:
                    temp_front = 0
                else:
                    temp_front = idx[0] + idx_max

                if temp_front > max_front_idx:
                    max_front_idx = temp_front
                    list_with_front = [i]
                elif temp_front == max_front_idx:
                    list_with_front.append(i)

            max_front_value = 0
            list_with_value = []
            for idx in list_with_front:
                val = matrix[max_front_idx - 1, idx] if axis == 0 else matrix[idx, max_front_idx - 1]
                if val > max_front_value:
                    max_front_value = val
                    list_with_value = [idx]
                elif val == max_front_value:
                    list_with_value.append(idx)

            if len(list_with_value) > 1:
                max_index = np.unravel_index(
                    np.argmax(matrix[:, list_with_value]) if axis == 0 else np.argmax(matrix[list_with_value, :]),
                    matrix.shape
                )
                return list_with_value[max_index[1 if axis == 0 else 0]]
            else:
                return list_with_value[0]

        def get_front_index(profile):
            idx = np.where(profile <= flow_threshold)[0]
            return (idx[0] - 1) if len(idx) else len(profile) - 1

        def get_back_index(profile):
            idx = np.where(profile <= flow_threshold)[0]
            return (idx[-1] + 1) if len(idx) else 0

        # Extract data and time
        field_all = self._loaded_results[model].get_output(field_name)
        tim = self._loaded_results[model].tim
        t_idx = None
        
        
        # If no data extracted, stop the program
        if isinstance(field_all, tilupy.read.AbstractResults) and not isinstance(field_all, tilupy.read.TemporalResults):
            raise UserWarning(f"{field_name} is not accessible in {model}.")
                
        
        # Find the index of the wanted time (or the closest)
        if t is not None:
            for idx in range(len(tim)):
                if tim[idx] == t:
                    t_idx = idx
                    break
                else:
                    if abs(tim[idx] - t) < 0.01:
                        t_idx = idx
                        break
            
            if t_idx is None:
                raise ValueError(f"Invalid time step. Recorded time steps are : {tim}")
        else:
            t_idx = field_all.d.shape[2]-1

        
        # Keep only the data field at the wanted time
        field_t = field_all.d[:, :, t_idx]
        
        
        # Apply mask on data
        field_t[field_t <= flow_threshold] = 0

        y_size, x_size = field_t.shape


        # Different cases depending on the type of direction_index 
        if direction_index is None:
            direction_index = [None, None]
            direction_index[0] = x_size//2
            direction_index[1] = y_size//2
            
        elif isinstance(direction_index[0], float):
            x_val = direction_index[0]
            y_val = direction_index[1]
            
            x_index, y_index = None, None
            for x in range(x_size):
                if abs(x_val - self._x[x]) <= 0.1:
                    x_index = x
                    break
            for y in range(y_size):
                if abs(y_val - self._y[y]) <= 0.1:
                    y_index = y
                    break
            direction_index[0] = x_size//2 if x_index is None else x_index
            direction_index[1] = y_size//2 if y_index is None else y_index
            
        elif direction_index == "max":
            direction_index = [None, None]
            direction_index[0] = find_max_index_along_axis(field_t, axis=0, threshold=flow_threshold)
            direction_index[1] = find_max_index_along_axis(field_t, axis=1, threshold=flow_threshold)

        elif direction_index[0] is None or direction_index[1] is None:
            direction_index[0] = x_size//2
            direction_index[1] = y_size//2
        
        
        # Save profiles in the dictionaries       
        if t in storage_dict_Y:
            if not any(res[0] == model for res in storage_dict_Y[t]):
                # storage_dict_Y[t].append((model, field_t[:, direction_index[0]]))
                storage_dict_Y[t].append((model, tilupy.read.StaticResults1D(name=field_name,
                                                                             d=field_t[:, direction_index[0]],
                                                                             coords=self._y,
                                                                             coords_name='y')))
        else:
            # storage_dict_Y[t] = [(model, field_t[:, direction_index[0]])]
            storage_dict_Y[t] = [(model, tilupy.read.StaticResults1D(name=field_name,
                                                                     d=field_t[:, direction_index[0]],
                                                                     coords=self._y,
                                                                     coords_name='y'))]
        if t in storage_dict_X:
            if not any(res[0] == model for res in storage_dict_X[t]):
                # storage_dict_X[t].append((model, field_t[direction_index[1], :]))
                storage_dict_X[t].append((model, tilupy.read.StaticResults1D(name=field_name,
                                                                             d=field_t[direction_index[1], :],
                                                                             coords=self._x,
                                                                             coords_name='x')))
        else:
            # storage_dict_X[t] = [(model, field_t[direction_index[1], :])]
            storage_dict_X[t] = [(model, tilupy.read.StaticResults1D(name=field_name,
                                                                     d=field_t[direction_index[1], :],
                                                                     coords=self._x,
                                                                     coords_name='x'))]

        # Calculate front positions
        idx_max_X = np.argmax(field_t[direction_index[1], :])
        idx_max_Y = np.argmax(field_t[:, direction_index[0]])

        idx_r = get_front_index(field_t[direction_index[1], idx_max_X:]) + idx_max_X
        idx_l = get_back_index(field_t[direction_index[1], :idx_max_X])
        idx_u = get_front_index(field_t[idx_max_Y:, direction_index[0]]) + idx_max_Y
        idx_d = get_back_index(field_t[:idx_max_Y, direction_index[0]])

        
        # Save front positions
        if t in storage_params:
            if not any(res[0] == model for res in storage_params[t]):
                storage_params[t].append((model, direction_index, [idx_l, idx_r, idx_d, idx_u]))
        else:
            storage_params[t] = [(model, direction_index, [idx_l, idx_r, idx_d, idx_u])]

        if show_message:
            print(self._current_model, f" -> {field_name}")
            print(f"Selected index:\n   X -> {direction_index[0]}\n   Y -> {direction_index[1]}")
            print(f"Front positions:\n  Right -> {idx_r}, {self._x[idx_r]}m\n  Left -> {idx_l}, {self._x[idx_l]}m\n  Up -> {idx_u}, {self._y[idx_u]}m\n  Down -> {idx_d}, {self._y[idx_d]}m")

        return direction_index


    def extract_profiles(self,
                         output: str,
                         model: str = None,
                         t: float = None,
                         direction_index: list[float] | str = None,
                         flow_threshold: float = 1e-3,
                         show_message: bool = False,
                         ) -> None:
        """Extract 1D profiles along X and Y axis at a specific time and store it for future use.

        Parameters
        ----------
        output : str
            Wanted data profile. Can be: "h", "u", "ux", "uy".
        model : str, optional
            Wanted model to extract profiles, if None take the currently loaded model. By default None.
        t : float, optional
            Time index to extract. If None, uses the last available time step.
        direction_index : list[float] or str, optional
            Index along each axis to extract the profile from: (X, Y). Depending on the information given, the extract method change:
                - list[float]: position (in meter) of the wanted profiles.
                - str: 'max' for finding the best profiles, based on the farthest flow front and the position of maximum fluid height.
                - None: select the medians.
        flow_threshold : float, optional
            Minimum height to consider as part of the flow, by default 1e-3.
        show_message : bool, optional
            If True, print the indexes and the front positions saved. 

        Raises
        ------
        ValueError
            If the model has not been loaded.
        ValueError
            If the output asked is not allowed.
        """
        if model is None:
            model = self._current_model

        if model not in self._loaded_results.keys():
            raise ValueError(" -> First load model using load_numerical_result.")
        
        if output not in self._allowed_extracting_outputs:
            raise ValueError(" -> Invalid output. See _allowed_extracting_outputs.")
        
        if direction_index is None:
            direction_index = [None, None]            
        
        outputs_dict = {"h" : [self._h_num_1d_X, self._h_num_1d_Y, self._h_num_1d_params],
                        "u" : [self._u_num_1d_X, self._u_num_1d_Y, self._u_num_1d_params],
                        "ux" : [self._ux_num_1d_X, self._ux_num_1d_Y, self._ux_num_1d_params],
                        "uy" : [self._uy_num_1d_X, self._uy_num_1d_Y, self._uy_num_1d_params]}
        
        self.process_data_field(model=model, 
                                field_name=output, 
                                storage_dict_X=outputs_dict[output][0], 
                                storage_dict_Y=outputs_dict[output][1], 
                                storage_params=outputs_dict[output][2], 
                                t=t,
                                direction_index=direction_index, 
                                flow_threshold=flow_threshold, 
                                show_message=show_message)
 

    def extract_field(self,
                      output: str,
                      model: str = None,
                      t: float = None
                      ):
        if model is None:
            model = self._current_model

        if model not in self._loaded_results:
            raise ValueError(" -> Model not load, use first load_numerical_result(model).")
        
        if output not in self._allowed_extracting_outputs:
            raise ValueError(" -> Invalid output. See _allowed_extracting_outputs.")
        
        outputs_dict = {"h" : self._h_num_2d,
                        "u" : self._u_num_2d,
                        "ux" : self._ux_num_2d,
                        "uy" : self._uy_num_2d}
        
        field_2d_all = self._loaded_results[model].get_output(output)
        tim = self._loaded_results[model].tim
        t_idx = None
        
        # If no data extracted, stop the program
        if isinstance(field_2d_all, tilupy.read.AbstractResults) and not isinstance(field_2d_all, tilupy.read.TemporalResults):
            raise UserWarning(f"{output} is not accessible in {model}.")
        
        # Find the index of the wanted time (or the closest)
        if t is not None:
            for idx in range(len(tim)):
                if tim[idx] == t:
                    t_idx = idx
                    break
                else:
                    if abs(tim[idx] - t) < 0.01:
                        t_idx = idx
                        break
            if t_idx is None:
                raise ValueError(f"Invalid time step. Recorded time steps are : {tim}")
        else:
            t_idx = field_2d_all.d.shape[2]-1

        t_idx = field_2d_all.d.shape[2]-1 if t_idx is None or t_idx >= field_2d_all.d.shape[2] or isinstance(t_idx, float) else t_idx
        field_2d_t = field_2d_all.d[:, :, t_idx]
    
        if t in outputs_dict[output]:
            if not any(res[0] == model for res in outputs_dict[output][t]):
                # outputs_dict[output][t].append((model, field_2d_t))
                outputs_dict[output][t].append((model, tilupy.read.StaticResults2D(name=output,
                                                                                   d=field_2d_t,
                                                                                   x=self._x,
                                                                                   y=self._y,
                                                                                   z=self._z)))
        else:
            outputs_dict[output][t] = [(model, tilupy.read.StaticResults2D(name=output,
                                                                           d=field_2d_t,
                                                                           x=self._x,
                                                                           y=self._y,
                                                                           z=self._z))]


    def show_profile(self,
                     output: str,
                     model: str,
                     axis: str = "X",
                     time_steps: float | list[float] = None,
                     direction_index: list[float] | str = None,
                     flow_threshold: float = 0.001,
                     ax: matplotlib.axes._axes.Axes = None,
                     figsize: tuple[float] = None,
                     linestyles: list[str] = None,
                     plot_multiples: bool = False,
                     multiples_highlight: bool = False,
                     plot_kwargs: dict = None,
                     show_plot: bool = True,
                     )-> matplotlib.axes._axes.Axes:
        """Plot 1D profiles for a given model or the analytic solution.

        Parameters
        ----------
        output : str
            Wanted data profile. Can be: "h", "u", "ux", "uy".
        model : str
            Wanted model to show the profile. Can be 'as' for plotting the analytical solution.
        axis : str, optional
            Allows to choose the profile according to the desired axis, by default "X".
        time_steps : float | list[float], optional
            Value or list of time steps required to extract and display profiles. 
            If None displays the profiles already extracted. Only available for
            models. By default None.
        direction_index : list[float] | str, optional
            Index along each axis to extract the profile from: (X, Y). Depending on the information given, the extract method change:
                - list[float]: position (in meter) of the wanted profiles.
                - str: 'max' for finding the best profiles, based on the farthest flow front and the position of maximum fluid height.
                - None: select the medians.
            By default None.
        flow_threshold : float, optional
            Minimum value to consider as part of the flow, by default 0.001.
        ax : matplotlib.axes._axes.Axes, optional
            Existing matplotlib window, by default None.
        figsize : tuple[float], optional
            Size of the plotted figure if no ax is given (Width, Height; in inch). By default None.
        linestyles : list[str], optional
            Custom linestyles for each time step. If None, colors and styles are auto-assigned. By default None.
        plot_multiples : bool, optional
            If True, plot graph in small multiples. Only if :data:`time_steps` is list. By default False.
        multiples_highlight : bool, optional
            If True, display all time steps on each graph of the multiples and highlight the curve corresponding 
            to the time step of the subgraph. Only if :data:`plot_multiples` is True. By default False.
        plot_kwargs : dict, optional
            Additional arguments for the plot function. By default None.
        show_plot : bool, optional
            If True, show the plot, by default True.

        Returns
        -------
        matplotlib.axes._axes.Axes
            _description_

        Raises
        ------
        ValueError
            If the output asked is not allowed.
        ValueError
            If model is not loaded or no analytical solution asked.
        ValueError
            If no analytical solution computed.
        ValueError
            If the profiles are not extracted. 
        ValueError
            If invalid axis.
        """
        if output not in self._allowed_extracting_outputs:
            raise ValueError(" -> Invalid output. See _allowed_extracting_outputs.")
        
        if model not in self._loaded_results.keys() and model != "as":
            raise ValueError(" -> First load model using load_numerical_result or compute analytical solution.")

        plot = None
        profil_idx = None
        profil_positions = []
        
        axis = axis.upper()
                
        outputs_dict = {"h" : [self._h_num_1d_X, self._h_num_1d_Y, self._h_num_1d_params],
                        "u" : [self._u_num_1d_X, self._u_num_1d_Y, self._u_num_1d_params],
                        "ux" : [self._ux_num_1d_X, self._ux_num_1d_Y, self._ux_num_1d_params],
                        "uy" : [self._uy_num_1d_X, self._uy_num_1d_Y, self._uy_num_1d_params]}
        
        as_dict = {"h" : self._h_as_1d,
                   "u" : self._u_as_1d}
        
        # For analytic solution profile
        if model == 'as' and axis == 'X' and (output == "h" or output == "u"):            
            if len(as_dict[output]) == 0:
                raise ValueError(" -> No analytic solution computed.")
            else:
                plot = as_dict[output][:]
        
        # For models profile
        elif model in self._allowed_models:
            # If time_steps is given, first extract the wanted profiles at specified time
            if isinstance(time_steps, float):
                    self.extract_profiles(output, 
                                          model, 
                                          time_steps, 
                                          direction_index, 
                                          flow_threshold)
                
            elif isinstance(time_steps, list):
                for t in time_steps:
                    self.extract_profiles(output,
                                          model, 
                                          t, 
                                          direction_index, 
                                          flow_threshold)

            # Then extract the profile depending on the axis 
            if axis == "X":
                if any(label == model for lst in outputs_dict[output][0].values() for label, _ in lst):
                    plot = [(t, val) for t, lst in outputs_dict[output][0].items() for label, val in lst if label == model]
                    profil_idx = [idx[1] for t, lst in outputs_dict[output][2].items() for label, idx, front in lst if label == model]
                else:
                    raise ValueError(" -> Solution not extracted.")
            elif axis == "Y":
                if any(label == model for lst in outputs_dict[output][1].values() for label, _ in lst):
                    plot = [(t, val) for t, lst in outputs_dict[output][1].items() for label, val in lst if label == model]
                    profil_idx = [idx[0] for t, lst in outputs_dict[output][2].items() for label, idx, front in lst if label == model]
                else:
                    raise ValueError(" -> Solution not extracted.")
            else:
                raise ValueError(" -> Incorrect axis: 'X' or 'Y'.")
        
        # Create fig if not given
        if ax is None and not plot_multiples:
            fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        
        # Recover the position of the extracted profiles
        absci = self._y if axis == 'Y' else self._x
        inv_axis = 'X' if axis == 'Y' else 'Y'
        profil_coord = self._x if axis == 'Y' else self._y
        if profil_idx:
            profil_positions = [float(profil_coord[i]) for i in profil_idx]
            # print(f"Profiles' position: {profil_positions}m")
    
    
        # Plot the profiles
        if len(plot) == 1: # Only one time to plot
            plot[0][1].plot(ax=ax, 
                            color='black', 
                            **plot_kwargs)
        
        else: # Multiple times to plot
            t = [s[0] for s in plot]
            all_values = [s[1].d for s in plot]     
            
            stack_values = np.stack(all_values, axis=1)
            temporal_output = tilupy.read.TemporalResults1D(name=output,
                                                            d=stack_values,
                                                            t=t,
                                                            coords=plot[0][1].coords,
                                                            coords_name=plot[0][1].coords_name)
            if plot_multiples:
                axes = temporal_output.plot(plot_type="multiples",
                                            figsize=figsize,
                                            highlighted_curve=multiples_highlight)
                

                for i in range(len(profil_positions)):
                    axes[i].set_title(f"{inv_axis}={profil_positions[i]}m", loc="right")
                
                if show_plot:
                    plt.show()
                
                return axes 
                
            temporal_output.plot(plot_type="simple", 
                                 ax=ax,
                                 linestyles=linestyles,
                                 **plot_kwargs)
                
        # Formatting the fig
        if len(plot) == 1:
            ax.set_title(f"{inv_axis}={profil_positions[0]}m  |  t={plot[0][0]}s")
        else:
            ax.set_title(f"{inv_axis}={profil_positions[0]}m")
            ax.legend(loc='upper right')
                
        if show_plot:
            plt.show()
            
        return ax       
    
    
    def show_field(self,
                   output: str,
                   model: str,
                   t: float | list[float],
                   ax: matplotlib.axes._axes.Axes = None,
                   figsize: tuple[float] = None,
                   show_plot: bool = True,
                   **plot_kwargs,
                   ) -> matplotlib.axes._axes.Axes:
        if output not in self._allowed_extracting_outputs:
            raise ValueError(" -> Invalid output. See _allowed_extracting_outputs.")
        
        if model not in self._loaded_results.keys():
            raise ValueError(" -> First load model using load_numerical_result.")
        
        outputs_dict = {"h" : self._h_num_2d,
                        "u" : self._u_num_2d,
                        "ux" : self._ux_num_2d,
                        "uy" : self._uy_num_2d}
        
        # Extract output if not already done
        if isinstance(t, float):
            if t not in outputs_dict[output]:
                self.extract_field(output=output,
                                   model=model, 
                                   t=t)
                    
            if not any(res[0] == model for res in outputs_dict[output][t]):
                self.extract_field(output=output,
                                  model=model, 
                                  t=t)
        
        elif isinstance(t, list):
            for T in t:
                if T not in outputs_dict[output]:
                    self.extract_field(output=output,
                                       model=model, 
                                       t=T)
                        
                if not any(res[0] == model for res in outputs_dict[output][T]):
                    self.extract_field(output=output,
                                       model=model, 
                                       t=T)
        
        if isinstance(t, float):
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize, layout="constrained")
            
            for lst in outputs_dict[output][t]:
                if lst[0] == model:
                    lst[1].plot(ax = ax,
                                **plot_kwargs)
                    break
            ax.set_title(f"t={t}s")
            
            if show_plot:
                plt.show()

            return ax

        else:
            if any(label == model for lst in outputs_dict[output].values() for label, _ in lst):
                data = [(t, val) for t, lst in outputs_dict[output].items() for label, val in lst if label == model]
            else:
                raise ValueError(" -> Solution not extracted.")
            
            t = [s[0] for s in data]
            all_values = [s[1].d for s in data]     
            
            stack_values = np.stack(all_values, axis=2)

            temporal_output = tilupy.read.TemporalResults2D(name=output,
                                                            d=stack_values,
                                                            t=t,
                                                            x=self._x,
                                                            y=self._y,
                                                            z=self._z)
            
            axes = temporal_output.plot(plot_multiples=True,
                                        **plot_kwargs)
            
            if show_plot:
                plt.show()

            return axes

'''
    def show_height_profile_comparison(self,
                                       models_to_plot: list[str],
                                       time_step: float,
                                       axis: str = "X",
                                       plot_as: bool = False,
                                    #    nbr_point: int = 20,
                                       colors: list = None,
                                       linestyles: list[str] = None,
                                       ax: matplotlib.axes._axes.Axes = None,
                                       x_unit:str = "m",
                                       h_unit:str = "m",
                                       cmap: str = 'plasma',
                                       show_plot: bool = True,
                                       ) -> matplotlib.axes._axes.Axes:
        """
        Plot multiple height profiles and optionally compare them to the analytic solution.

        Parameters
        ----------
        models_to_plot : list[str]
            List of model names to compare (in :attr:`_allowed_models`).
        time_step : float
            Time step for the plot.
        axis : str, optional
            Allows to choose the profile according to the desired axis, by default 'X'.
        plot_as : bool, optional
            Whether to include the analytic solution, by default False.
        nbr_point : int, optional
            Number of points to plot for each curve, by default 20.
        colors
        
        linestyles
        
        ax : matplotlib.axes._axes.Axes, optional
            Existing matplotlib window.
        x_unit : str, optional
            Unit for the x-axis, by default "m".
        h_unit : str, optional
            Unit for the height axis, by default "m".
        cmap : str, optional
            Cmap for the model_to_plot curves, be default 'plasma'. 
        show_plot : bool, optional
            If True, show the plot, by default True.

        Returns
        -------
        matplotlib.axes._axes.Axes
            The axis with the plotted profiles.
        
        Raises
        ------
        ValueError
            If no result has been loaded.
        ValueError
            If no result computed at the specified time step for the analytical solution.
        ValueError
            If the axis is incorrect.
        """
        # marker_list = ["o", "s", "^", "p", "D", "h", "v", "*"]
        # cmap = plt.get_cmap(cmap, len(models_to_plot))
        
        if self._x is None:
            raise ValueError(" -> No solution extracted, first use load_numerical_result.")
        
        axis = axis.upper()
        
        if axis == 'X':
            # step = len(self._x) // nbr_point
            
            # If no profile for the time_step, extract it
            if time_step not in self._h_num_1d_X.keys():
                for model in models_to_plot:
                    self.extract_height_profiles(model=model,
                                                 t=time_step)
            
            if ax is None:
                fig, ax = plt.subplots()
            
           # Plot analytic solution
            if plot_as:
                if any(res[0] == time_step for res in self._h_as_1d):
                    for t, h in self._h_as_1d:
                        if t == time_step:
                            # ax.plot(self._x, h, linestyle="-", color='black', label="AS")
                            ax.plot(self._x, h, linestyle=":", color='red', alpha=0.9, linewidth=1.5, label="AS")
                            break
                else:
                    raise ValueError(" -> Time step not computed in analytical solution.") 
            
            # Plot models
            for i in range(len(models_to_plot)):
                if any(res[0] == models_to_plot[i] for res in self._h_num_1d_X[time_step]):
                    for m, h in self._h_num_1d_X[time_step]:
                        if models_to_plot[i] == m:
                            # j = i
                            # while j >= len(marker_list):
                            #     j -= len(marker_list)
                            #     if j < 0:
                            #         j = 0
                            # ax.plot(self._x[::step], h[::step], marker=marker_list[j], markeredgecolor="black", markeredgewidth=0.2, linestyle='None', color=cmap(i), label=models_to_plot[i])
                            
                            if colors is not None and len(colors) > i:
                                color = colors[i]
                            else:
                                color = "black"
                            if linestyles is not None and len(linestyles) > i:
                                linestyle = linestyles[i]
                            else:
                                linestyle = None
                                
                            ax.plot(self._x, h, color=color, linestyle=linestyle, label=models_to_plot[i], alpha=0.8, linewidth=2)
                            
                            
                            # plot_params = importlib.import_module("tilupy.models." + model + ".plot_params")
                            # ax.plot(self._x[::step], h[::step], marker=plot_params.marker, linestyle='None', color=plot_params.color, label=model)

            ax.set_xlim(left=min(self._x), right=max(self._x))
            ax.set_xlabel(f"x [{x_unit}]")
        
        elif axis == 'Y':
            # step = len(self._y) // nbr_point
            
            # If no profile for the time_step, extract it
            if time_step not in self._h_num_1d_Y.keys():
                for model in models_to_plot:
                    self.extract_height_profiles(model=model,
                                                 t=time_step)
            
            if ax is None:
                fig, ax = plt.subplots()
            
            """NO ANALYTIC SOLUTION ALONG AXIS Y
            if plot_as:
                if any(res[0] == time_step for res in self._h_as_1d):
                    for t, h in self._h_as_1d:
                        if t == time_step:
                            ax.plot(self._x, h, linestyle="-", color='black', label="AS")
                            break
                else:
                    raise ValueError(" -> Time step not computed in analytical solution.")
            """

            # Plot models
            for i in range(len(models_to_plot)):
                if any(res[0] == models_to_plot[i] for res in self._h_num_1d_Y[time_step]):
                    for m, h in self._h_num_1d_Y[time_step]:
                        if models_to_plot[i] == m:
                            # j = i
                            # while j >= len(marker_list):
                            #     j -= len(marker_list)
                            #     if j < 0:
                            #         j = 0
                            # ax.plot(self._x[::step], h[::step], marker=marker_list[j], markeredgecolor="black", markeredgewidth=0.2, linestyle='None', color=cmap(i), label=models_to_plot[i])

                            if colors is not None and len(colors) > i:
                                color = colors[i]
                            else:
                                color = "black"
                            if linestyles is not None and len(linestyles) > i:
                                linestyle = linestyles[i]
                            else:
                                linestyle = None
    
                            ax.plot(self._y, h, color=color, linestyle=linestyle, label=models_to_plot[i], alpha=0.8, linewidth=2)
                            
            ax.set_xlim(left=min(self._y), right=max(self._y))
            ax.set_xlabel(f"y [{x_unit}]")

        else:
            raise ValueError(" -> Incorrect axis: 'Y' or 'X'.")
        
        
        # Formatting fig  
        ax.grid(which='major')
        ax.grid(which='minor', alpha=0.5)
        
        ax.set_title(f"Height comparison along {axis} for t={time_step}s")
        
        ax.set_ylabel(f"h [{h_unit}]")
        ax.legend(loc='upper right')
        
        if show_plot:
            plt.show()
        
        return ax
'''
'''
    def show_velocity_profile_comparison(self,
                                         models_to_plot: list[str],
                                         time_step: float,
                                         velocity_axis: str = "u",
                                         axis: str = "X",
                                         velocity_threshold: float = 1e-6,
                                         plot_as: bool = False,
                                         nbr_point: int = 20,
                                         ax: matplotlib.axes._axes.Axes = None,
                                         x_unit:str = "m",
                                         u_unit:str = "m/s",
                                         cmap: str = 'plasma',
                                         show_plot: bool = True,
                                         ) -> matplotlib.axes._axes.Axes:
        """
        Plot multiple velocity profiles and optionally compare them to the analytic solution.

        Parameters
        ----------
        models_to_plot : list[str]
            List of model names to compare (in :attr:`_allowed_models`).
        time_step : float
            Time step for the plot.
        velocity_axis : str, optional
            Velocity direction to use for the plot ("u", "ux" or "uy"), by default 'u' (norm).
        axis : str, optional
            Allows to choose the profile according to the desired axis, by default 'X'.
        velocity_threshold : float, optional
            Threshold value where lower values will be replaced by Nan, by default 1e-6.
        plot_as : bool, optional
            Whether to include the analytic solution, by default False.
        nbr_point : int, optional
            Number of points to plot for each curve, by default 20.
        ax : matplotlib.axes._axes.Axes, optional
            Existing matplotlib window.
        x_unit : str, optional
            Unit for the x-axis, by default "m".
        u_unit : str, optional
            Unit for the height axis, by default "m".
        cmap : str, optional
            Cmap for the model_to_plot curves, be default 'plasma'. 
        show_plot : bool, optional
            If True, show the plot, by default True.

        Returns
        -------
        matplotlib.axes._axes.Axes
            The axis with the plotted profiles.
        
        Raises
        ------
        ValueError
            If no result has been loaded.
        ValueError
            If no result computed at the specified time step for the analytical solution.
        ValueError
            If the axis is incorrect.
        ValueError
            If the velocity axis is incorrect.
        """
        marker_list = ["o", "s", "^", "p", "D", "h", "v", "*"]
        cmap = plt.get_cmap(cmap, len(models_to_plot))
        
        if self._x is None:
            raise ValueError(" -> No solution extracted, first use load_numerical_result.")
        
        axis = axis.upper()
        
        if axis == 'X':
            step = len(self._x) // nbr_point
            
            if velocity_axis == 'u':
                if time_step not in self._u_num_1d_X.keys():
                    for model in models_to_plot:
                        self.extract_velocity_profiles(model=model,
                                                       t=time_step)
                
                if ax is None:
                    fig, ax = plt.subplots()
                
                if plot_as:
                    if any(res[0] == time_step for res in self._u_as_1d):
                        for t, h in self._u_as_1d:
                            if t == time_step:
                                h[h <= velocity_threshold] = np.nan
                                ax.plot(self._x, h, linestyle="-", color='black', label="AS")
                                break
                    else:
                        raise ValueError(" -> Time step not computed in analytical solution.")
            
                for i in range(len(models_to_plot)):
                    if any(res[0] == models_to_plot[i] for res in self._u_num_1d_X[time_step]):
                        for m, h in self._u_num_1d_X[time_step]:
                            if models_to_plot[i] == m:
                                h[h <= velocity_threshold] = np.nan
                                j = i
                                while j >= len(marker_list):
                                    j -= len(marker_list)
                                    if j < 0:
                                        j = 0
                                ax.plot(self._x[::step], h[::step], marker=marker_list[j], markeredgecolor="black", markeredgewidth=0.2, linestyle='None', color=cmap(i), label=models_to_plot[i])

            
            elif velocity_axis == 'ux':
                if time_step not in self._ux_num_1d_X.keys():
                    for model in models_to_plot:
                        self.extract_velocity_profiles(model=model,
                                                       t=time_step)
                
                if ax is None:
                    fig, ax = plt.subplots()

                for i in range(len(models_to_plot)):
                    if any(res[0] == models_to_plot[i] for res in self._ux_num_1d_X[time_step]):
                        for m, h in self._ux_num_1d_X[time_step]:
                            if models_to_plot[i] == m:
                                h[h <= velocity_threshold] = np.nan
                                j = i
                                while j >= len(marker_list):
                                    j -= len(marker_list)
                                    if j < 0:
                                        j = 0
                                ax.plot(self._x[::step], h[::step], marker=marker_list[j], markeredgecolor="black", markeredgewidth=0.2, linestyle='None', color=cmap(i), label=models_to_plot[i])


            elif velocity_axis == 'uy':
                if time_step not in self._uy_num_1d_X.keys():
                    for model in models_to_plot:
                        self.extract_velocity_profiles(model=model,
                                                       t=time_step)
                
                if ax is None:
                    fig, ax = plt.subplots()
            
                for i in range(len(models_to_plot)):
                    if any(res[0] == models_to_plot[i] for res in self._uy_num_1d_X[time_step]):
                        for m, h in self._uy_num_1d_X[time_step]:
                            if models_to_plot[i] == m:
                                h[h <= velocity_threshold] = np.nan
                                j = i
                                while j >= len(marker_list):
                                    j -= len(marker_list)
                                    if j < 0:
                                        j = 0
                                ax.plot(self._x[::step], h[::step], marker=marker_list[j], markeredgecolor="black", markeredgewidth=0.2, linestyle='None', color=cmap(i), label=models_to_plot[i])


            else:
                raise ValueError(" -> Incorrect velocity axis: 'u', 'ux' or 'uy")
            
            ax.set_xlim(left=min(self._x), right=max(self._x))
            ax.set_xlabel(f"x [{x_unit}]")
            
        
        elif axis == 'Y':
            step = len(self._y) // nbr_point

            if velocity_axis == 'u':
                if time_step not in self._u_num_1d_Y.keys():
                    for model in models_to_plot:
                        self.extract_velocity_profiles(model=model,
                                                       t=time_step)
                
                if ax is None:
                    fig, ax = plt.subplots()
            
                for i in range(len(models_to_plot)):
                    if any(res[0] == models_to_plot[i] for res in self._u_num_1d_Y[time_step]):
                        for m, h in self._u_num_1d_Y[time_step]:
                            if models_to_plot[i] == m:
                                h[h <= velocity_threshold] = np.nan
                                j = i
                                while j >= len(marker_list):
                                    j -= len(marker_list)
                                    if j < 0:
                                        j = 0
                                ax.plot(self._x[::step], h[::step], marker=marker_list[j], markeredgecolor="black", markeredgewidth=0.2, linestyle='None', color=cmap(i), label=models_to_plot[i])


            elif velocity_axis == 'ux':
                if time_step not in self._ux_num_1d_Y.keys():
                    for model in models_to_plot:
                        self.extract_velocity_profiles(model=model,
                                                       t=time_step)
                
                if ax is None:
                    fig, ax = plt.subplots()
            
                for i in range(len(models_to_plot)):
                    if any(res[0] == models_to_plot[i] for res in self._ux_num_1d_Y[time_step]):
                        for m, h in self._ux_num_1d_Y[time_step]:
                            if models_to_plot[i] == m:
                                h[h <= velocity_threshold] = np.nan
                                j = i
                                while j >= len(marker_list):
                                    j -= len(marker_list)
                                    if j < 0:
                                        j = 0
                                ax.plot(self._x[::step], h[::step], marker=marker_list[j], markeredgecolor="black", markeredgewidth=0.2, linestyle='None', color=cmap(i), label=models_to_plot[i])


            elif velocity_axis == 'uy':
                if time_step not in self._uy_num_1d_Y.keys():
                    for model in models_to_plot:
                        self.extract_velocity_profiles(model=model,
                                                       t=time_step)
                
                if ax is None:
                    fig, ax = plt.subplots()
            
                for i in range(len(models_to_plot)):
                    if any(res[0] == models_to_plot[i] for res in self._uy_num_1d_Y[time_step]):
                        for m, h in self._uy_num_1d_Y[time_step]:
                            if models_to_plot[i] == m:
                                h[h <= velocity_threshold] = np.nan
                                j = i
                                while j >= len(marker_list):
                                    j -= len(marker_list)
                                    if j < 0:
                                        j = 0
                                ax.plot(self._x[::step], h[::step], marker=marker_list[j], markeredgecolor="black", markeredgewidth=0.2, linestyle='None', color=cmap(i), label=models_to_plot[i])


            else:
                raise ValueError(" -> Incorrect velocity axis: 'u', 'ux' or 'uy.")
            
            ax.set_xlim(left=min(self._y), right=max(self._y))
            ax.set_xlabel(f"y [{x_unit}]")

        else:
            raise ValueError(" -> Incorrect axis: 'Y' or 'X'.")
             
        ax.grid(which='major')
        ax.grid(which='minor', alpha=0.5)
        
        ax.set_title(f"Velocity comparison along {axis} for t={time_step}s")
        
        ax.set_ylabel(f"u [{u_unit}]")
        ax.legend(loc='upper right')
        
        if show_plot:
            plt.show()
        
        return ax
'''

'''
    def show_height_profile_with_coussot_morpho(self, 
                                                model_to_plot: str,
                                                rho: float, 
                                                tau: float, 
                                                theta: int=0, 
                                                H_size: int=100, 
                                                direction: str = "right",
                                                direction_index: list[float] | str = None,
                                                flow_threshold: float = 1e-3,
                                                front_position: float = None,
                                                lateral_position: list[float] = None,
                                                h_init: float = None, 
                                                h_final: float = None, 
                                                axes: matplotlib.axes._axes.Axes = None, 
                                                x_unit: str = "m", 
                                                h_unit: str = "m",
                                                show_plot: bool = True,
                                                ) -> list[matplotlib.axes._axes.Axes] :
        """
        Compare the final morphological profile (along X and Y axis) of the numerical result to the expected Coussot theoretical profile.

        Parameters
        ----------
        model_to_plot : str
            Model name to plot ('shaltop', 'lave2D', 'saval2D').
        rho : float
            Density of the material.
        tau : float
            Yield stress.
        theta : int, optional
            Slope angle (in degrees), by default 0.
        H_size : int, optional
            Number of discretization points, by default 100.
        direction : str, optional
            Direction of the flow ("right", "left", "down", "up"), by default "right".
        direction_index : list[float] or str, optional
            Index along each axis to extract the profile from: (X, Y). Depending on the information given, the extract method change:
                - list[float]: position (in meter) of the wanted profiles.
                - str: 'max' for finding the best profiles, based on the farthest flow front and the position of maximum fluid height.
                - None: select the medians.
        front_position : float, optional
            Location to align the theoretical shape for the flow front. If None, detected automatically.
        lateral_position : float, optional
            Locations to align the theoretical shape for the lateral flow fronts: [min, max]. If None, detected automatically.
        h_init : float, optional
            Initial flow height. Used to compute the expected shape.
        h_final : float, optional
            Final flow height. Used to compute the expected shape. If None, computed automatically using h_init.
        axes : matplotlib.axes._axes.Axes, optional
            Existing matplotlib window.
        x_unit : str, optional
            Unit for the x-axis, by default "m".
        h_unit : str, optional
            Unit for the height axis, by default "m".
        show_plot: bool, optional
            If True, show the plot, by default True.

        Returns
        -------
        list[matplotlib.axes._axes.Axes]
            The axis with the plotted comparison : 
                axes[0] contains the leading front of the flow.
                axes[1] Contains the lateral fronts of the flow

        Raises
        ------
        ValueError
            If invalid direction.
        ValueError
            If the model is not recognized.
        ValueError
            If no numerical result has been loaded.
        """
        if direction not in ["right", "left", "up", "down"]:
            raise ValueError(' -> Invalid direction: "right", "left", "up", "down"')
        
        if model_to_plot not in self._allowed_models:
            raise ValueError(f" -> No allowed model selected, choose between:\n       {self._allowed_models}")
                
        # Extract the profiles at the latest time
        self.extract_height_profiles(model=model_to_plot,
                                     t=None,
                                     direction_index=direction_index,
                                     flow_height_threshold=flow_threshold)
        
        # Extract profile along X
        if direction == "right" or direction == "left":
            max_time = max(self._h_num_1d_X.keys())
            saved_profiles = self._h_num_1d_X[max_time]
            
            if any(res[0] == model_to_plot for res in saved_profiles):
                for res in saved_profiles:
                    if res[0] == model_to_plot:
                        front_profile = res[1]
                        break    
            else:
                raise ValueError(" -> Model not loaded, first extract profiles.")

            # Find the equivalent Y axis profile
            for lat_res in self._h_num_1d_Y[max_time]:
                if lat_res[0] == model_to_plot:
                    lat_profile = lat_res[1]
                    break  
        
        
        # Do for axis Y
        else:
            max_time = max(self._h_num_1d_Y.keys())
            saved_profiles = self._h_num_1d_Y[max_time]
            
            if any(res[0] == model_to_plot for res in saved_profiles):
                for res in saved_profiles:
                    if res[0] == model_to_plot:
                        front_profile = res[1]
                        break    
            else:
                raise ValueError(" -> Model not loaded, first extract profiles.")

            for lat_res in self._h_num_1d_X[max_time]:
                if lat_res[0] == model_to_plot:
                    lat_profile = lat_res[1]
                    break  
        
        
        # Extract front positions
        for params in self._h_num_1d_params[max_time]:
            if params[0] == model_to_plot:
                # idx = params[1]
                fronts = params[2]
        
        
        # Compute Coussot's solution and positions it in the right place for the front flow
        morpho = AS.Coussot_shape(rho=rho, tau=tau, theta=theta, H_size=H_size)
        morpho.compute_rheological_test_front_morpho(h_init=h_init, h_final=h_final)
        
        if direction == "right" or direction == "up" :
            morpho.change_orientation_flow()
        
        if front_position is None:
            if direction == "right":
                morpho.translate_front(self._x[fronts[1]])
            elif direction == "left":
                morpho.translate_front(self._x[fronts[0]])
            elif direction == "down":
                morpho.translate_front(self._y[fronts[2]])
            elif direction == "up":
                morpho.translate_front(self._y[fronts[3]])
        else:
            morpho.translate_front(front_position)

        
        # Create fig
        if axes is None:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
        
        
        # Formatting axes[0] for the front flow 
        if direction == "up" or direction == "down" :
            axes[0].plot(self._y, front_profile, linestyle='-', color='black', label=model_to_plot)
            axes[0].set_xlim(left=min(self._y), right=max(self._y))
            axes[0].set_xlabel(f"y [{x_unit}]")
 
        else:
            axes[0].plot(self._x, front_profile, linestyle='-', color='black', label=model_to_plot)
            axes[0].set_xlim(left=min(self._x), right=max(self._x))
            axes[0].set_xlabel(f"x [{x_unit}]")
        
        axes[0].plot(morpho.x, morpho.h, linestyle="--", color='red', label="Coussot shape")
        axes[0].legend(loc='upper right')
        
        axes[0].grid(which='major')
        axes[0].grid(which='minor', alpha=0.5)
        
        axes[0].set_ylabel(f"h [{h_unit}]")
        axes[0].legend(loc='upper right')
        
        axes[0].set_title("Flow front")

        
        # Compute Coussot's solution and positions it in the right place for the lateral flow
        if theta != 0:
            morpho.compute_rheological_test_lateral_morpho()
            lat_morpho_left = morpho.y
            lat_morpho_right = [-1*v for v in morpho.y]
        
        else:
            morpho.compute_rheological_test_front_morpho(h_init=h_init, h_final=h_final)
            lat_morpho_left = morpho.x
            lat_morpho_right = [-1*v for v in morpho.x]
        
        if lateral_position is None:
            if direction == "right":
                lat_morpho_left = [v+self._y[fronts[2]] for v in lat_morpho_left]
                lat_morpho_right = [v+self._y[fronts[3]] for v in lat_morpho_right]
            elif direction == "left":
                lat_morpho_left = [v+self._y[fronts[3]] for v in lat_morpho_left]
                lat_morpho_right = [v+self._y[fronts[2]] for v in lat_morpho_right]
            elif direction == "down":
                lat_morpho_left = [v+self._x[fronts[0]] for v in lat_morpho_left]
                lat_morpho_right = [v+self._x[fronts[1]] for v in lat_morpho_right]
            elif direction == "up":
                lat_morpho_left = [v+self._x[fronts[1]] for v in lat_morpho_left]
                lat_morpho_right = [v+self._x[fronts[0]] for v in lat_morpho_right]
        else:
            lat_morpho_left = [v+lateral_position[0] for v in lat_morpho_left]
            lat_morpho_right = [v+lateral_position[1] for v in lat_morpho_right]
        
        
        # Formatting axes[1] for the lateral flow 
        if direction == "up" or direction == "down" :
            axes[1].plot(self._x, lat_profile, linestyle='-', color='black', label=model_to_plot)
            axes[1].set_xlim(left=min(self._x), right=max(self._x))
            axes[1].set_xlabel(f"x [{x_unit}]")
 
        else:
            axes[1].plot(self._y, lat_profile, linestyle='-', color='black', label=model_to_plot)
            axes[1].set_xlim(left=min(self._y), right=max(self._y))
            axes[1].set_xlabel(f"y [{x_unit}]")
        
        axes[1].plot(lat_morpho_left, morpho.h, linestyle="--", color='red', label="Coussot shape")
        axes[1].plot(lat_morpho_right, morpho.h, linestyle="--", color='red', label="Coussot shape")
        axes[1].legend(loc='upper right')
        
        axes[1].grid(which='major')
        axes[1].grid(which='minor', alpha=0.5)
        
        axes[1].set_ylabel(f"h [{h_unit}]")
        axes[1].legend(loc='upper right')
        
        axes[1].set_title("Lateral flow front")
        
        if show_plot:
            plt.show()

        return axes
    '''