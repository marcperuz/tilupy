# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 12:09:41 2023

@author: peruzzetto
"""
import importlib

from typing import Callable

import math as math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tilupy.read
import tilupy.analytic_sol as AS
import pytopomap.plot as pyplt


class Benchmark:
    def __init__(self):
        self._allowed_model = ["shaltop", "lave2D", "saval2D"]
        
        self._current_model = None
        self._current_result = {}
        self._x, self._y, self._z = None, None, None
        
        self._h_num_1d_X = {}
        self._h_num_1d_Y = {}
        self._h_num_1d_params = {}
        
        self._h_as_1d = []
        
        self._u_num_1d_X = {}
        self._u_num_1d_Y = {}
        self._u_num_1d_params = {}
        
        self._ux_num_1d_X = {}
        self._ux_num_1d_Y = {}
        self._ux_num_1d_params = {}
        
        self._uy_num_1d_X = {}
        self._uy_num_1d_Y = {}
        self._uy_num_1d_params = {}
        
        self._u_as_1d = []
        
        self._h_num_2d = {}
        self._u_num_2d = {}
        self._ux_num_2d = {}
        self._uy_num_2d = {}
    
    
    def load_numerical_result(self, 
                              model: str, 
                              **kwargs,
                              ) -> None:
        """
        Load numerical simulation results for a given model.

        Parameters
        ----------
        model : str
            Name of the model to load. Must be one of the allowed models: ['shaltop', 'lave2D', 'saval2D'].
        **kwargs
            Keyword arguments passed to the result reader for the specific model.

        Raises
        ------
        ValueError
            If the provided model is not in the allowed list.
        """
        if model in self._allowed_model:
            self._current_model = model
            if model not in self._current_result:
                self._current_result[model] = tilupy.read.get_results(model, **kwargs)
            self._x, self._y, self._z = self._current_result[model].x, self._current_result[model].y, self._current_result[model].z
        else:
            raise ValueError(f" -> No correct model selected, choose between:\n       {self._allowed_model}")
    
    
    def process_data_field(self, 
                           model: str, 
                           field_name: str, 
                           storage_dict_X: dict, 
                           storage_dict_Y: dict, 
                           storage_params: dict, 
                           t: float, 
                           direction_index: str | list[float], 
                           flow_threshold: float = 1e-3,
                           show_message: bool = False,
                           ) -> str | list[float]:
        """Process data field to extract profiles.

        Parameters
        ----------
        model : str
            Model to extract the profile.
        field_name : str
            Data field. Can be: 'h', 'u', 'ux' or 'uy'.
        storage_dict_X : dict
            Dictionary to save the X axis profile. 
        storage_dict_Y : dict
            Dictionary to save the Y axis profile.
        storage_params : dict
            Dictionary to save the parameters of the profiles.
        t_val : float
            Value of the time to extract the profiles.
        direction_index : str | list[float]
            Index along each axis to extract the profile from: (X, Y). If None, it is detected automatically based on the farthest flow front 
            and the position of maximum fluid data.
        flow_threshold : float
            Minimum velocity to consider as part of the flow, by default 1e-3.
        show_message : bool, optional
            If True, print the indexes and the front positions saved, by default False

        Returns
        -------
        str | list[float]
            direction_index
        """
        # Extract data and time
        field_all = self._current_result[model].get_output(field_name)
        tim = self._current_result[model].tim
        t_idx = None
        
        # If no data extracted, stop the program
        if field_all is None:
            return direction_index
        
        
        # Find the index of the wanted time (or the closest)
        if t is not None:
            for idx in range(len(tim)):
                if tim[idx] == t:
                    t_idx = idx
                    break
                else:
                    if abs(tim[idx] - t) < 0.1:
                        t_idx = idx
                        break
        else:
            t_idx = field_all.d.shape[2]-1

        
        # Keep only the data field at the wanted time
        field_t = field_all.d[:, :, t_idx]
        field_t[field_t <= flow_threshold] = 0

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

        y_size, x_size = field_t.shape


        # Different cases depending on the type of direction_index 
        if isinstance(direction_index[0], float):
            x_val = direction_index[0]
            y_val = direction_index[1]
            
            x_index, y_index = None, None
            for x in range(x_size):
                if abs(x_val - self._x[x]) <= 0.1:
                    x_index = x
            for y in range(y_size):
                if abs(y_val - self._y[y]) <= 0.1:
                    y_index = y
            
            direction_index[0] = x_index if x_index is not None else None
            direction_index[1] = y_index if y_index is not None else None
           
        elif isinstance(direction_index, str):
            direction_index = [None, None]
            direction_index[0] = find_max_index_along_axis(field_t, axis=0, threshold=flow_threshold)
            direction_index[1] = find_max_index_along_axis(field_t, axis=1, threshold=flow_threshold)

        else:
            direction_index[0] = x_size//2
            direction_index[1] = y_size//2
        
        if direction_index[0] is None or direction_index[1] is None:
            direction_index[0] = x_size//2
            direction_index[1] = y_size//2
        
        
        # Save profiles in the dictionaries       
        if t in storage_dict_Y:
            if not any(res[0] == model for res in storage_dict_Y[t]):
                storage_dict_Y[t].append((model, field_t[:, direction_index[0]]))
        else:
            storage_dict_Y[t] = [(model, field_t[:, direction_index[0]])]

        if t in storage_dict_X:
            if not any(res[0] == model for res in storage_dict_X[t]):
                storage_dict_X[t].append((model, field_t[direction_index[1], :]))
        else:
            storage_dict_X[t] = [(model, field_t[direction_index[1], :])]


        # Calculate front positions
        idx_max_X = np.argmax(field_t[direction_index[1], :])
        idx_max_Y = np.argmax(field_t[:, direction_index[0]])

        def get_front_index(profile):
            idx = np.where(profile <= flow_threshold)[0]
            return (idx[0] - 1) if len(idx) else len(profile) - 1

        def get_back_index(profile):
            idx = np.where(profile <= flow_threshold)[0]
            return (idx[-1] + 1) if len(idx) else 0

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


    def extract_height_profiles(self,
                                model: str = None,
                                t: float = None,
                                direction_index: list[float] | str = None,
                                flow_height_threshold: float = 1e-3,
                                show_message: bool = False,
                                ) -> None:
        """
        Extract 1D height profiles along X and Y axis at a specific time and store it for future use.

        Parameters
        ----------
        model : str, optional
            Wanted model to extract profiles, if None take the currently loaded model.
        t : float, optional
            Time index to extract. If None, uses the last available time step.
        direction_index : list[float] or str, optional
            Index along each axis to extract the profile from: (X, Y). Depending on the information given, the extract method change:
                - list[float]: position (in meter) of the wanted profiles.
                - str: 'max' for finding the best profiles, based on the farthest flow front and the position of maximum fluid height.
                - None: select the medians.
        flow_height_threshold : float, optional
            Minimum height to consider as part of the flow, by default 1e-3.
        show_message : bool, optional
            If True, print the indexes and the front positions saved. 

        Raises
        ------
        ValueError
            If the model has not been loaded.
        """
        if model is None:
            model = self._current_model

        if model not in self._current_result:
            raise ValueError(" -> Model not load, use first load_numerical_result(model).")
        
        if direction_index is None:
            direction_index = [None, None]            
                
        self.process_data_field(model=model, 
                                field_name='h', 
                                storage_dict_X=self._h_num_1d_X, 
                                storage_dict_Y=self._h_num_1d_Y, 
                                storage_params=self._h_num_1d_params, 
                                t=t,
                                direction_index=direction_index, 
                                flow_threshold=flow_height_threshold, 
                                show_message=show_message)


    def extract_velocity_profiles(self,
                                  model: str = None,
                                  t: float = None,
                                  direction_index: list[int] = None,
                                  flow_velocity_threshold: float = 1e-3,
                                  show_message: bool = False
                                  ) -> None:
        """
        Extract 1D velocity profiles along X and Y axis at a specific time and store it for future use.

        Parameters
        ----------
        model : str, optional
            Wanted model to extract profiles, if None take the currently loaded model.
        t : float, optional
            Time index to extract. If None, uses the last available time step.
        direction_index : tuple[int], optional
            Index along each axis to extract the profile from: (X, Y). If None, it is detected automatically based on the farthest flow front 
            and the position of maximum fluid velocity.
        flow_velocity_threshold : float, optional
            Minimum velocity to consider as part of the flow, by default 1e-3.
        show_message : bool, optional
            If True, print the indexes and the front positions saved. 

        Raises
        ------
        ValueError
            If the model has not been loaded.
        """
        if model is None:
            model = self._current_model
        
        if model not in self._current_result:
            raise ValueError(" -> Model not load, use first load_numerical_result(model).")
                
        if direction_index is None:
            direction_index = [None, None]
        
        self.process_data_field(model=model,
                                field_name='u',
                                storage_dict_X=self._u_num_1d_X,
                                storage_dict_Y=self._u_num_1d_Y,
                                storage_params=self._u_num_1d_params,
                                t=t,
                                direction_index=direction_index,
                                flow_threshold=flow_velocity_threshold,
                                show_message=show_message)
        
        self.process_data_field(model=model,
                                field_name='ux',
                                storage_dict_X=self._ux_num_1d_X,
                                storage_dict_Y=self._ux_num_1d_Y,
                                storage_params=self._ux_num_1d_params,
                                t=t,
                                direction_index=direction_index,
                                flow_threshold=flow_velocity_threshold,
                                show_message=show_message)
        
        self.process_data_field(model=model,
                                field_name='uy',
                                storage_dict_X=self._uy_num_1d_X,
                                storage_dict_Y=self._uy_num_1d_Y,
                                storage_params=self._uy_num_1d_params,
                                t=t,
                                direction_index=direction_index,
                                flow_threshold=flow_velocity_threshold,
                                show_message=show_message)
        

    def extract_height_field(self,
                             model: str = None,
                             t: float = None,
                             ) -> None:
        """
        Extract and store the full 2D height field at a given time step and store it for future use.

        Parameters
        ----------
        model : str, optional
            Wanted model to extract profiles, if None take the currently loaded model.
        t : float, optional
            Time index to extract. If None or invalid, uses the last available time step.

        Raises
        ------
        ValueError
            If the model has not been loaded.
        """
        if model is None:
            model = self._current_model

        if model not in self._current_result:
            raise ValueError(" -> Model not load, use first load_numerical_result(model).")

        h_2d_all = self._current_result[model].get_output("h")
        tim = self._current_result[model].tim
        t_idx = None
        
        # Find the index of the wanted time (or the closest)
        if t is not None:
            for idx in range(len(tim)):
                if tim[idx] == t:
                    t_idx = idx
                    break
                else:
                    if abs(tim[idx] - t) < 0.1:
                        t_idx = idx
                        break
        else:
            t_idx = h_2d_all.d.shape[2]-1
        
        t_idx = h_2d_all.d.shape[2]-1 if t_idx is None or t_idx >= h_2d_all.d.shape[2] or isinstance(t_idx, float) else t_idx
        h_2d_t = h_2d_all.d[:, :, t_idx]
                    
        if t in self._h_num_2d:
            if not any(res[0] == model for res in self._h_num_2d[t]):
                self._h_num_2d[t].append((model, h_2d_t))
        else:
            self._h_num_2d[t] = [(model, h_2d_t)]
            

    def extract_velocity_field(self,
                               model: str = None,
                               t: float = None,
                               ) -> None:
        """
        Extract and store the full 2D velocity field at a given time step and store it for future use.

        Parameters
        ----------
        model : str, optional
            Wanted model to extract profiles, if None take the currently loaded model.
        t : float, optional
            Time index to extract. If None or invalid, uses the last available time step.

        Raises
        ------
        ValueError
            If the model has not been loaded.
        """
        if model is None:
            model = self._current_model

        if model not in self._current_result:
            raise ValueError(" -> Model not load, use first load_numerical_result(model).")
        
        u_2d_all = self._current_result[model].get_output('u')
        ux_2d_all = self._current_result[model].get_output('ux')
        uy_2d_all = self._current_result[model].get_output('uy')
        tim = self._current_result[model].tim
        
        # Find the index of the wanted time (or the closest)
        if t is not None:
            for idx in range(len(tim)):
                if tim[idx] == t:
                    t = idx
                    break
                else:
                    if abs(tim[idx] - t) < 0.1:
                        t = idx
                        break
        else:
            t = u_2d_all.d.shape[2]-1
            # print("oui")
        
        if u_2d_all is not None:
            t = u_2d_all.d.shape[2]-1 if t is None or t >= u_2d_all.d.shape[2] or isinstance(t, float) else t
            u_2d_t = u_2d_all.d[:, :, t]
            
            if t in self._u_num_2d:
                if not any(res[0] == self._current_model for res in self._u_num_2d[t]):
                    self._u_num_2d[t].append((model, u_2d_t))
            else:
                self._u_num_2d[t] = [(model, u_2d_t)]
        
        if ux_2d_all is not None:
            t = ux_2d_all.d.shape[2]-1 if t is None or t >= ux_2d_all.d.shape[2] or isinstance(t, float) else t
            u_2d_t = ux_2d_all.d[:, :, t]
            
            if t in self._ux_num_2d:
                if not any(res[0] == self._current_model for res in self._ux_num_2d[t]):
                    self._ux_num_2d[t].append((model, u_2d_t))
            else:
                self._ux_num_2d[t] = [(model, u_2d_t)]
        
        if uy_2d_all is not None:
            t = uy_2d_all.d.shape[2]-1 if t is None or t >= uy_2d_all.d.shape[2] or isinstance(t, float) else t
            u_2d_t = uy_2d_all.d[:, :, t]
            
            if t in self._uy_num_2d:
                if not any(res[0] == self._current_model for res in self._uy_num_2d[t]):
                    self._uy_num_2d[t].append((model, u_2d_t))
            else:
                self._uy_num_2d[t] = [(model, u_2d_t)]
         
    
    def compute_analytic_solution(self, 
                                  model: Callable, 
                                  T: float | list[float], 
                                  **kwargs
                                  ) -> None:
        """
        Compute the analytic solution for the given time steps using the provided model.

        Parameters
        ----------
        model : Callable
            Callable object representing the analytic solution model (model from tilupy.analytic_sol)
        T : float | list[float]
            Time or list of times at which to compute the analytic solution.
        **kwargs
            Keyword arguments passed to the analytic solution for the specific model.
        """
        solution = model(**kwargs)
        
        if isinstance(T, float):
            T = [T]
        
        solution.compute_h(self._x, T)
        solution.compute_u(self._x, T)
        
        if solution.h is not None:
            for i in range(len(T)):
                self._h_as_1d.append((T[i], solution.h[i]))    
        if solution.u is not None:
            for i in range(len(T)):
                self._u_as_1d.append((T[i], solution.u[i]))
    

    def show_height_profile(self,
                            model_to_plot: str,
                            axis: str = "X",
                            time_steps: float | list[float] = None,
                            direction_index = None,
                            flow_height_threshold = 0.001,
                            linestyles: list[str] = None,
                            ax: matplotlib.axes._axes.Axes = None,
                            x_unit:str = "m",
                            h_unit:str = "m",
                            show_plot: bool = True,
                            ) -> matplotlib.axes._axes.Axes:
        """
        Plot 1D height profiles for a given model or the analytic solution.

        Parameters
        ----------
        model_to_plot : str
            Model name to plot ('shaltop', 'lave2D', 'saval2D', or 'as' for analytic solution).
        axis : str, optional
            Allows to choose the profile according to the desired axis, by default 'X'.
        time_steps : float or list[float], optional
            List of time steps required to extract and display profiles. If None displays the profiles already extracted. Only available for
            models.
        linestyles : list[str], optional
            Custom linestyles for each time step. If None, colors and styles are auto-assigned.
        ax : matplotlib.axes._axes.Axes, optional
            Existing matplotlib window.
        x_unit : str, optional
            Unit for the x-axis, by default "m".
        h_unit : str, optional
            Unit for the height axis, by default "m".
        show_plot : bool, optional
            If True, show the plot, by default True.
        
        Returns
        -------
        matplotlib.axes._axes.Axes
            The axis with the plotted profiles.

        Raises
        ------
        ValueError
            If the model is not recognized.
        ValueError
            If no analytic solution was computed (when 'as' is requested).
        ValueError
            If the requested model has no stored data.
        ValueError
            If the axis is incorrect.
        """
        h_plot = None
        profil_idx = None
        profil_positions = []
        
        axis = axis.upper()
                
        # For analytic solution profile
        if model_to_plot == 'as' and axis == 'X':            
            if len(self._h_as_1d) == 0:
                raise ValueError(" -> No analytic solution computed.")
            else:
                h_plot = self._h_as_1d[:]
        
        
        # For models profile
        elif model_to_plot in self._allowed_model:
            # If time_steps is given, first extract the wanted profiles at specified time
            if isinstance(time_steps, float):
                    self.extract_height_profiles(model_to_plot, time_steps, direction_index, flow_height_threshold)
                
            elif isinstance(time_steps, list):
                for t in time_steps:
                    self.extract_height_profiles(model_to_plot, t, direction_index, flow_height_threshold)
            
            
            # Then extract the profile depending on the axis 
            if axis == "Y":
                if any(label == model_to_plot for lst in self._h_num_1d_Y.values() for label, _ in lst):
                    h_plot = [(t, val) for t, lst in self._h_num_1d_Y.items() for label, val in lst if label == model_to_plot]
                    profil_idx = [idx[0] for t, lst in self._h_num_1d_params.items() for label, idx, front in lst if label == model_to_plot]
                else:
                    raise ValueError(" -> Solution not extracted.")
            elif axis == "X":
                if any(label == model_to_plot for lst in self._h_num_1d_X.values() for label, _ in lst):
                    h_plot = [(t, val) for t, lst in self._h_num_1d_X.items() for label, val in lst if label == model_to_plot]
                    profil_idx = [idx[1] for t, lst in self._h_num_1d_params.items() for label, idx, front in lst if label == model_to_plot]
                else:
                    raise ValueError(" -> Solution not extracted.")
            else:
                raise ValueError(" -> Incorrect axis: 'Y' or 'X'.")
        
        else:
            raise ValueError(f" -> No allowed model selected, choose between:\n       {self._allowed_model.append('as')}")

        
        # Create fig if not given
        if ax is None:
            fig, ax = plt.subplots()
        
        
        # Print the position of the plotted profiles
        absci = self._y if axis == 'Y' else self._x
        profil_coord = self._x if axis == 'Y' else self._y
        if profil_idx:
            profil_positions = [float(profil_coord[i]) for i in profil_idx]
            print(f"Profiles' position: {profil_positions}m")
        
        
        # Plot the profiles
        if len(h_plot) == 1: # Only one time to plot
            ax.plot(absci, h_plot[0][1], color='black', linewidth=1, label=f"t={h_plot[0][0]}s")
        else: # Multiple times to plot
            if linestyles is None or len(linestyles)!=(len(h_plot)):
                norm = plt.Normalize(vmin=min(t for t, _ in h_plot), vmax=max(t for t, _ in h_plot))
                cmap = plt.cm.copper
                
            for sol_idx, sol_val in enumerate(h_plot):
                t_val = h_plot[sol_idx][0]
                if linestyles is None or len(linestyles)!=(len(h_plot)):
                    color = cmap(norm(t_val)) if t_val != 0 else "red"
                    l_style = "-" if t_val != 0 else (0, (1, 4))
                else:
                    color = "black" if t_val != 0 else "red"
                    l_style = linestyles[sol_idx] if t_val != 0 else (0, (1, 4))    
                ax.plot(absci, sol_val[1], color=color, linestyle=l_style, label=f"t={t_val}s")


        # Formatting the fig
        ax.grid(which='major')
        ax.grid(which='minor', alpha=0.5)
        ax.set_xlim(left=min(absci), right=max(absci))
        
        ax.set_title(f"Flow height profile along {axis}")
        ax.set_xlabel(f"x [{x_unit}]")
        ax.set_ylabel(f"h [{h_unit}]")
        ax.legend(loc='upper right')
        
        if show_plot:
            plt.show()
            
        return ax        


    def show_velocity_profile(self,
                              model_to_plot: str,
                              velocity_axis: str='ux',
                              axis: str="X",
                              time_steps: float | list[float] = None,
                              direction_index = None,
                              flow_velocity_threshold = 0.001,
                              velocity_threshold: float=1e-6,
                              linestyles: list[str]=None,
                              ax: matplotlib.axes._axes.Axes = None,
                              x_unit:str = "m",
                              h_unit:str = "m",
                              show_plot: bool = True,
                              ) -> matplotlib.axes._axes.Axes:
        """
        Plot 1D velocity profiles for a given model or the analytic solution.

        Parameters
        ----------
        model_to_plot : str
            Model name to plot ('shaltop', 'lave2D', 'saval2D', or 'as' for analytic solution).
        velocity_axis : str, optional
            Velocity direction to use for the plot, by default 'U' (along X axis).
        axis : str, optional
            Allows to choose the profile according to the desired axis, by default 'X'.
        time_steps : float or list[float], optional
            List of time steps required to extract and display profiles. If None displays the profiles already extracted. Only available for
            models.
        velocity_threshold : float, optional
            Threshold value where lower values will be replaced by Nan, by default 1e-6.
        linestyles : list[str], optional
            Custom linestyles for each time step. If None, colors and styles are auto-assigned.
        ax : matplotlib.axes._axes.Axes, optional
            Existing matplotlib window.
        x_unit : str, optional
            Unit for the x-axis, by default "m".
        h_unit : str, optional
            Unit for the height axis, by default "m".
        show_plot : bool, optional
            If True, show the plot, by default True.
            
        Returns
        -------
        matplotlib.axes._axes.Axes
            The axis with the plotted profiles.   
        
        Raises
        ------
        ValueError
            If the model is not recognized.
        ValueError
            If no analytic solution was computed (when 'as' is requested).
        ValueError
            If the requested model has no stored data.
        ValueError
            If the axis is incorrect.
        """
        def get_profile_data(velocity_axis, axis, model):
            """Helper to extract profile data and index from appropriate dictionaries.

            Parameters
            ----------
            velocity_axis : str
                Axis 'ux' or 'uy'. Can also be 'u'.
            axis : str
                Acis 'X' or 'Y'
            model : str
                Model to plot

            Returns
            -------
            u_plot : list[np.ndarray] 
                List of profiles to plot.
            profil_idx : list[int]
                Profiles indexes.    

            Raises
            ------
            ValueError
                If incorrect axis.
            ValueError
                If no solution extracted.
            """
            axis = axis.upper()
            valid_axes = ['X', 'Y']
            if axis not in valid_axes or velocity_axis not in ['u', 'ux', 'uy']:
                raise ValueError(" -> Incorrect axis or velocity axis. Use axis in ['X', 'Y'] and velocity in ['u', 'ux', 'uy'].")

            dicts = {
                ('u', 'X'): self._u_num_1d_X,
                ('u', 'Y'): self._u_num_1d_Y,
                ('ux', 'X'): self._ux_num_1d_X,
                ('ux', 'Y'): self._ux_num_1d_Y,
                ('uy', 'X'): self._uy_num_1d_X,
                ('uy', 'Y'): self._uy_num_1d_Y,
            }
            params = {
                'u': self._u_num_1d_params,
                'ux': self._ux_num_1d_params,
                'uy': self._uy_num_1d_params,
            }

            data_dict = dicts[(velocity_axis, axis)]
            param_dict = params[velocity_axis]

            if not any(label == model for lst in data_dict.values() for label, _ in lst):
                raise ValueError(" -> Solution not extracted.")

            u_plot = [(t, val) for t, lst in data_dict.items() for label, val in lst if label == model]
            profil_idx = [idx[0 if axis == 'Y' else 1] for t, lst in param_dict.items() for label, idx, _ in lst if label == model]
            return u_plot, profil_idx

        u_plot = None
        profil_idx = None
        profil_positions = []
        
        axis = axis.upper()

        
        # For analytic solution profile
        if model_to_plot == 'as':
            if not self._u_as_1d:
                raise ValueError(" -> No analytic solution computed.")
            u_plot = self._u_as_1d[:]
            profil_idx = None
        
        
        # For models profile  
        elif model_to_plot in self._allowed_model:
            # If time_steps is given, first extract the wanted profiles at specified time
            if isinstance(time_steps, float):
                    self.extract_velocity_profiles(model_to_plot, time_steps, direction_index, flow_velocity_threshold)
                
            elif isinstance(time_steps, list):
                for t in time_steps:
                    self.extract_velocity_profiles(model_to_plot, t, direction_index, flow_velocity_threshold)
            
            
            # Extract the profiles
            u_plot, profil_idx = get_profile_data(velocity_axis, axis, model_to_plot)
        else:
            raise ValueError(f" -> Invalid model. Choose from: {self._allowed_model + ['as']}")


        # Create the fig
        if ax is None:
            _, ax = plt.subplots()

        
        # Print the position of the plotted profiles
        axis = axis.upper()
        absci = self._y if axis == 'Y' else self._x
        profil_coord = self._x if axis == 'Y' else self._y
        if profil_idx:
            profil_positions = [float(profil_coord[i]) for i in profil_idx]
            print(f"Profiles' position: {profil_positions}m")
        

        # Plot the profiles
        if isinstance(u_plot[0], tuple):
            if linestyles is None or len(linestyles) != len(u_plot):
                cmap = plt.cm.copper
                norm = plt.Normalize(vmin=min(t for t, _ in u_plot), vmax=max(t for t, _ in u_plot))
            for idx, (t_val, profile) in enumerate(u_plot):
                profile[profile <= velocity_threshold] = np.nan
                if linestyles and len(linestyles) == len(u_plot):
                    l_style = linestyles[idx]
                    color = "black" if t_val != 0 else "red"
                else:
                    color = cmap(norm(t_val)) if t_val != 0 else "red"
                    l_style = "-" if t_val != 0 else (0, (1, 4))
                ax.plot(absci, profile, color=color, linestyle=l_style, label=f"t={t_val}s")
        else:
            u_plot[u_plot <= velocity_threshold] = np.nan
            ax.plot(absci, u_plot, color='black', linewidth=1, label=f"t={t_val}s")


        # Formatting the fig
        ax.grid(which='both', alpha=0.5)
        ax.set_xlim(min(absci), max(absci))
        ax.set_title(f"Flow velocity ({velocity_axis}) profile along {axis}")
        ax.set_xlabel(f"x [{x_unit}]")
        ax.set_ylabel(f"h [{h_unit}]")
        ax.legend(loc='upper right')

        if show_plot:
            plt.show()

        return ax


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
        
        if model_to_plot not in self._allowed_model:
            raise ValueError(f" -> No allowed model selected, choose between:\n       {self._allowed_model}")
                
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


    def show_height_profile_comparison(self,
                                       models_to_plot: list[str],
                                       time_step: float,
                                       axis: str = "X",
                                       plot_as: bool = False,
                                       nbr_point: int = 20,
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
            List of model names to compare ('shaltop', 'lave2D', 'saval2D').
        time_step : float
            Time step for the plot.
        axis : str, optional
            Allows to choose the profile according to the desired axis, by default 'X'.
        plot_as : bool, optional
            Whether to include the analytic solution, by default False.
        nbr_point : int, optional
            Number of points to plot for each curve, by default 20.
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
        marker_list = ["o", "s", "^", "p", "D", "h", "v", "*"]
        cmap = cm.get_cmap(cmap, len(models_to_plot))
        
        if self._x is None:
            raise ValueError(" -> No solution extracted, first use load_numerical_result.")
        
        axis = axis.upper()
        
        if axis == 'X':
            step = len(self._x) // nbr_point
            
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
                            ax.plot(self._x, h, linestyle="-", color='black', label="AS")
                            break
                else:
                    raise ValueError(" -> Time step not computed in analytical solution.")


            # Plot models
            for i in range(len(models_to_plot)):
                if any(res[0] == models_to_plot[i] for res in self._h_num_1d_X[time_step]):
                    for m, h in self._h_num_1d_X[time_step]:
                        if models_to_plot[i] == m:
                            j = i
                            while j >= len(marker_list):
                                j -= len(marker_list)
                                if j < 0:
                                    j = 0
                            ax.plot(self._x[::step], h[::step], marker=marker_list[j], markeredgecolor="black", markeredgewidth=0.2, linestyle='None', color=cmap(i), label=models_to_plot[i])

                            # plot_params = importlib.import_module("tilupy.models." + model + ".plot_params")
                            # ax.plot(self._x[::step], h[::step], marker=plot_params.marker, linestyle='None', color=plot_params.color, label=model)

            ax.set_xlim(left=min(self._x), right=max(self._x))
            ax.set_xlabel(f"x [{x_unit}]")
        
        elif axis == 'Y':
            step = len(self._y) // nbr_point
            
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
                            j = i
                            while j >= len(marker_list):
                                j -= len(marker_list)
                                if j < 0:
                                    j = 0
                            ax.plot(self._x[::step], h[::step], marker=marker_list[j], markeredgecolor="black", markeredgewidth=0.2, linestyle='None', color=cmap(i), label=models_to_plot[i])

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
            List of model names to compare ('shaltop', 'lave2D', 'saval2D').
        time_step : float
            Time step for the plot.
        velocity_axis : str, optional
            Velocity direction to use for the plot, by default 'U' (along X axis).
        axis : str, optional
            Allows to choose the profile according to the desired axis, by default 'X'.
        velocity_threshold : float, optional
            Threshold value where lower values ​​will be replaced by Nan, by default 1e-6.
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
        cmap = cm.get_cmap(cmap, len(models_to_plot))
        
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


    def show_height_field(self,
                          model_to_plot: str,
                          t: float,
                          ax: matplotlib.axes._axes.Axes = None,
                          show_plot: bool = True,
                          **kwargs,
                          ) -> matplotlib.axes._axes.Axes:
        """
        Visualize the 2D height field at a specified time steps.

        Parameters
        ----------
        model_to_plot : str
            Model name to plot ('shaltop', 'lave2D', 'saval2D').
        t : float
            Time step to plot.
        ax : matplotlib.axes._axes.Axes, optional
            Existing matplotlib window.
        show_plot : bool, optional
            If True, show the plot, by default True.
        
        Returns
        -------
        matplotlib.axes._axes.Axes
            The axis with the plotted profiles.
        """
        if t not in self._h_num_2d:
            self.extract_height_field(model=model_to_plot, t=t)
                
        if not any(res[0] == model_to_plot for res in self._h_num_2d[t]):
            self.extract_height_field(model=model_to_plot, t=t)
        
        if ax is None:
            fig, ax = plt.subplots()
        
        for lst in self._h_num_2d[t]:
            if lst[0] == model_to_plot:
                pyplt.plot_data_on_topo(self._x, self._y, self._z, lst[1], axe=ax, **kwargs)
                break
            
        if show_plot:
            plt.show()
        
        return ax
  

    def show_velocity_field(self,
                            model_to_plot: str,
                            t: float,
                            velocity_axis: str = 'u',
                            ax: matplotlib.axes._axes.Axes = None,
                            show_plot: bool = True,
                            ) -> matplotlib.axes._axes.Axes:
        """
        Visualize the 2D height field at a specified time steps.

        Parameters
        ----------
        model_to_plot : str
            Model name to plot ('shaltop', 'lave2D', 'saval2D').
        t : float
            Time step to plot.
        velocity_axis : str, optional
            Velocity direction to use for the plot, by default 'U' (along X axis).
        ax : matplotlib.axes._axes.Axes, optional
            Existing matplotlib window.
        show_plot : bool, optional
            If True, show the plot, by default True.
        
        Returns
        -------
        matplotlib.axes._axes.Axes
            The axis with the plotted profiles.

        Raises
        ------
        ValueError
            If the velocity axis is incorrect.
        """
        if velocity_axis == 'u':
            if t not in self._u_num_2d:
                self.extract_velocity_field(model=model_to_plot, t=t)
            
            if not any(res[0] == model_to_plot for res in self._u_num_2d[t]):
                self.extract_velocity_field(model=model_to_plot, t=t)
            
            if ax is None:
                fig, ax = plt.subplots()
            
            for lst in self._u_num_2d[t]:
                if lst[0] == model_to_plot:
                    pyplt.plot_data_on_topo(self._x, self._y, self._z, lst[1], axe=ax)
                    break
                
        elif velocity_axis == 'ux':
            if t not in self._ux_num_2d:
                self.extract_velocity_field(model=model_to_plot, t=t)
            
            if not any(res[0] == model_to_plot for res in self._ux_num_2d[t]):
                self.extract_velocity_field(model=model_to_plot, t=t)
            
            if ax is None:
                fig, ax = plt.subplots()
            
            for lst in self._ux_num_2d[t]:
                if lst[0] == model_to_plot:
                    pyplt.plot_data_on_topo(self._x, self._y, self._z, lst[1], axe=ax)
                    break
                
        elif velocity_axis == 'uy':
            if t not in self._uy_num_2d:
                self.extract_velocity_field(model=model_to_plot, t=t)
            
            if not any(res[0] == model_to_plot for res in self._uy_num_2d[t]):
                self.extract_velocity_field(model=model_to_plot, t=t)
            
            if ax is None:
                fig, ax = plt.subplots()
            
            for lst in self._uy_num_2d[t]:
                if lst[0] == model_to_plot:
                    pyplt.plot_data_on_topo(self._x, self._y, self._z, lst[1], axe=ax)
                    break
        
        else:
            raise ValueError(" -> Incorrect velocity axis: 'u', 'ux' or 'uy'")
        
        if show_plot:
            plt.show()
        
        return ax
      

    '''def show_comparison_1D(self,
                           model_to_plot: list,
                           plot_as: bool = False,
                           x_unit:str = "m",
                           h_unit:str = "m",
                           cols: int = 4,
                           fig_size: tuple = None,
                           nbr_point: int = 20,
                           ) -> None:
        """
        Plot multiple 1D numerical solutions and optionally compare them to the analytic solution.

        Parameters
        ----------
        model_to_plot : list
            List of model names to compare ('shaltop', 'lave2D', 'saval2D').
        plot_as : bool, optional
            Whether to include the analytic solution, by default False.
        x_unit : str, optional
            Unit for the x-axis, by default "m".
        h_unit : str, optional
            Unit for the height axis, by default "m".
        cols : int, optional
            Number of columns in subplot grid, by default 4.
        fig_size : tuple, optional
            Figure size for the plots.
        nbr_point : int, optional
            Number of points to plot for each curve, by default 20.

        Raises
        ------
        ValueError
            If no spatial data has been loaded or extracted.
        """
        if self._x is None:
            raise ValueError("Solution not extracted.")
            
        n_graph = len(self._h_num_1d.keys())            
        rows = (n_graph // cols)
        
        step = max(1, len(self._x) // nbr_point)

        fig, axes = plt.subplots(rows, cols, figsize=fig_size)
        axes = axes.flatten()

        t_list = sorted(self._h_num_1d.keys())

        for i in range(n_graph):
            current_t = t_list[i]
            
            if plot_as:
                if any(res[0] == current_t for res in self._h_as_1d):
                    for t, h in self._h_as_1d:
                        if t == current_t:
                            axes[i].plot(self._x, h, linestyle="-", color='black', label="AS")
                            break
            
            for model in model_to_plot:
                if any(res[0] == model for res in self._h_num_1d[current_t]):
                    for m, h in self._h_num_1d[current_t]:
                        if model == "shaltop" and model == m:
                            axes[i].plot(self._x[::step], h[::step], marker='.', linestyle='None', color='red', label=model)
                        elif model == "lave2D" and model == m:
                            axes[i].plot(self._x[::step], h[::step], marker='x', linestyle='None', color='green', label=model)
                        elif model == "saval2D" and model == m:
                            axes[i].plot(self._x[::step], h[::step], marker='*', linestyle='None', color='blue', label=model)

            axes[i].set_title(f"t={current_t}")
            axes[i].grid(which='major')
            axes[i].grid(which='minor', alpha=0.5)
            axes[i].set_xlim(left=min(self._x), right=max(self._x))
            
            # plt.title(f"Flow height comparison")
            axes[i].set_xlabel(f"x [{x_unit}]")
            axes[i].set_ylabel(f"h [{h_unit}]")
            axes[i].legend(loc='upper right')
            
        for i in range(n_graph, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        plt.show()

    '''

    
    '''def show_difference_1D(self, 
                           first_models: str, 
                           second_models: str, 
                           cols: int=4, 
                           fig_size: tuple=None, 
                           x_unit: str="m") -> None:
        """
        Plot the difference between two models or between a model and the analytic solution.

        Parameters
        ----------
        first_models : str
            First model name or 'as' for analytic solution.
        second_models : str
            Second model name or 'as' for analytic solution.
        cols : int, optional
            Number of subplot columns, by default 4.
        fig_size : tuple, optional
            Figure size for the plots.
        x_unit : str, optional
            Unit for the x-axis, by default "m".

        Raises
        ------
        ValueError
            If one of the requested models is not implemented or missing in the stored results.
        """
        n_graph = len(self._h_num_1d.keys())            
        rows = (n_graph // cols)
        
        # step = max(1, len(self._x) // nbr_point)

        fig, axes = plt.subplots(rows, cols, figsize=fig_size)
        axes = axes.flatten()

        t_list = sorted(self._h_num_1d.keys())

        for i in range(n_graph):
            first_profil = None
            second_profil = None
            as_profil = None
            current_t = t_list[i]
            
            if "as" in first_models or "as" in second_models:
                if any(res[0] == current_t for res in self._h_as_1d):
                    for t, h in self._h_as_1d:
                        if t == current_t:
                            as_profil = h
                            break
            
            if first_models not in ["shaltop", "lave2D", "saval2D"]:
                if as_profil is None:
                    raise ValueError("First model not implemented in show_difference or synthaxe error.")            
            else:
                for m, h in self._h_num_1d[current_t]:
                    if first_models == m:
                        first_profil = h

            if second_models not in ["shaltop", "lave2D", "saval2D"]:
                if as_profil is None:
                    raise ValueError("Second model not implemented in show_difference or synthaxe error.")
            else:
                for m, h in self._h_num_1d[current_t]:
                    if second_models == m:
                        second_profil = h
            
            if as_profil is not None:
                if first_profil is not None:
                    diff = np.array(first_profil) - np.array(as_profil)
                    rms = np.sqrt(np.mean((np.array(first_profil) - np.array(as_profil))**2))
                else:
                    diff = np.array(second_profil) - np.array(as_profil)
                    rms = np.sqrt(np.mean((np.array(second_profil) - np.array(as_profil))**2))
            
                axes[i].plot(self._x, diff, linestyle='-', color='red', label="RMS " + "{:01.2f}".format(rms))
                plt.title(f"{first_models} - AS")
            else:
                diff = np.array(first_profil) - np.array(second_profil)
                rms = np.sqrt(np.mean((np.array(first_profil) - np.array(second_profil))**2))

                axes[i].plot(self._x, diff, linestyle='-', color='red', label=f"RMS {rms}")
                plt.title(f"{first_models} - {second_models}")
                        
            axes[i].set_title(f"t={current_t}")
            axes[i].grid(which='major')
            axes[i].grid(which='minor', alpha=0.5)
            axes[i].set_xlim(left=min(self._x), right=max(self._x))
            axes[i].set_ylim(bottom=min(diff)+min(diff)*0.05, top=max(diff)+max(diff)*0.05)

            axes[i].set_xlabel(f"x [{x_unit}]")
            axes[i].set_ylabel(f"Error")
            axes[i].legend(loc='upper center')
            
        for i in range(n_graph, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        plt.show()
    '''

        
    '''def show_difference_2D(self, 
                           first_models:str, 
                           second_models:str, 
                           cols: int=4, 
                           fig_size: tuple=None, 
                           x_unit: str="m") -> None:
        """
        Plot the difference between two models in 2D.

        Parameters
        ----------
        first_models : str
            First model name.
        second_models : str
            Second model name.
        cols : int, optional
            Number of subplot columns, by default 4
        fig_size : tuple, optional
            Figure size for the plots.
        x_unit : str, optional
            Unit for the x-axis, by default "m"

        Raises
        ------
        ValueError
            If one of the requested models is not implemented or missing in the stored results.
        """
        n_graph = len(self._h_num_2d.keys())            
        rows = (n_graph // cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=fig_size)
        axes = axes.flatten()

        t_list = sorted(self._h_num_2d.keys())

        for i in range(n_graph):
            first_data = None
            second_data = None
            current_t = t_list[i]
            

            if first_models not in ["shaltop", "lave2D", "saval2D"]:
                raise ValueError("First model not implemented in show_difference or synthaxe error.")            
            else:
                for m, h in self._h_num_2d[current_t]:
                    print(m)
                    if first_models == m:
                        print('oui')
                        first_data = h

            if second_models not in ["shaltop", "lave2D", "saval2D"]:
                raise ValueError("Second model not implemented in show_difference or synthaxe error.")
            else:
                print(0)
                for m, h in self._h_num_2d[current_t]:
                    if second_models == m:
                        print('non')
                        second_data = h
            
            
            diff = np.array(first_data) - np.array(second_data)
            rms = np.sqrt(np.mean((np.array(first_data) - np.array(second_data))**2))

            
            pyplt.plot_data_on_topo(self._x, self._y, self._z, diff, axe=axes[i], cmap="seismic")
            # plt.title(f"{first_models} - {second_models}")
                    
            axes[i].set_title(f"t={current_t}")
            axes[i].grid(which='major')
            axes[i].grid(which='minor', alpha=0.5)

            axes[i].set_xlabel(f"x [{x_unit}]")
            axes[i].set_ylabel(f"Error")
            axes[i].legend(loc='upper center')
            
        for i in range(n_graph, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        plt.show()
    '''

