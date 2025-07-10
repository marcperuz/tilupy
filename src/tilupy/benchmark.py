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
import tilupy.analytic_sol as AS
import pytopomap.plot as pyplt


class Benchmark:
    def __init__(self):
        self._allowed_model = ["shaltop", "lave2D", "saval2D"]
        
        self._current_model = None
        self._current_result = None
        self._x, self._y, self._z = None, None, None
        
        self._h_num_1d_X = {}
        self._h_num_1d_Y = {}
        self._h_num_1d_params = {}
        
        self._h_as_1d = []
        
        self._u_num_1d_X = {}
        self._u_num_1d_Y = {}
        self._u_num_1d_params = {}
        
        self._v_num_1d_X = {}
        self._v_num_1d_Y = {}
        self._v_num_1d_params = {}
        
        self._u_as_1d = []
        
        self._h_num_2d = {}
        self._u_num_2d = {}
        self._v_num_2d = {}
    
    
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
            self._current_result = tilupy.read.get_results(model, **kwargs)
            self._x, self._y, self._z = self._current_result.x, self._current_result.y, self._current_result.z
        else:
            raise ValueError(f" -> No correct model selected, choose between:\n       {self._allowed_model}")
    
    
    def extract_height_profiles(self,
                                t: float=None,
                                direction_index: list[int] = None,
                                flow_height_threshold: float = 1e-3,
                                show_message: bool = False,
                                ) -> None:
        """
        Extract 1D height profiles along X and Y axis at a specific time and store it for future use.

        Parameters
        ----------
        t : float, optional
            Time index to extract. If None, uses the last available time step.
        direction_index : list[int], optional
            Index along each axis to extract the profile from: (X, Y). If None, it is detected automatically based on the farthest flow front 
            and the position of maximum fluid height.
        flow_height_threshold : float, optional
            Minimum height to consider as part of the flow, by default 1e-3.
        show_message : bool, optional
            If True, print the indexes and the front positions saved. 

        Raises
        ------
        ValueError
            If no numerical result has been loaded.
        """
        if self._current_result is None:
            raise ValueError(" -> No result load, use first load_numerical_result.")
                
        if direction_index is None:
            direction_index = [None, None]
        
        h_2d_all = self._current_result.get_output('h')

        t = h_2d_all.d.shape[2]-1 if t is None or t >= h_2d_all.d.shape[2] else t
        h_2d_t = h_2d_all.d[:, :, t]
        h_2d_t[h_2d_t <= flow_height_threshold] = 0
        
        y_size, x_size = h_2d_t.shape[0], h_2d_t.shape[1]
           
        # ALONG Y
        if direction_index[0] is None:
            max_front_idx = 0
            list_column_with_front = []
            
            for x in range(x_size):
                idx_max = np.argmax(h_2d_t[:, x])
                profile = h_2d_t[idx_max:, x]
                idx = np.where(profile < flow_height_threshold)[0]
                                
                if len(idx) == 0:
                    max_front_idx = 0
                    list_column_with_front = [0]
                
                else:
                    if (idx[0] + idx_max) > max_front_idx :
                        max_front_idx = (idx[0] + idx_max)
                        list_column_with_front = [x]
                    
                    elif (idx[0] + idx_max) == max_front_idx :
                        list_column_with_front.append(x)
                                    
            max_front_value = 0
            list_column_with_value = []
            for idx in list_column_with_front:
                val = h_2d_t[max_front_idx - 1, idx]
                if val > max_front_value:
                    max_front_value = val
                    list_column_with_value = [idx]
                elif val == max_front_value:
                    list_column_with_value.append(idx)
                            
            if len(list_column_with_value) > 1:
                max_index = np.unravel_index(np.argmax(h_2d_t[:, list_column_with_value]), h_2d_t.shape)
                direction_index[0] = list_column_with_value[max_index[1]]
            else:
                direction_index[0] = list_column_with_value[0]
            
        if t in self._h_num_1d_Y:
            if not any(res[0] == self._current_model for res in self._h_num_1d_Y[t]):
                self._h_num_1d_Y[t].append((self._current_model, h_2d_t[:, direction_index[0]]))
        else:
            self._h_num_1d_Y[t] = [(self._current_model, h_2d_t[:, direction_index[0]])]
        
        # ALONG X
        if direction_index[1] is None:
            max_front_idx = 0
            list_row_with_front = []
            
            for y in range(y_size):
                idx_max = np.argmax(h_2d_t[y, :])
                profile = h_2d_t[y, idx_max:]
                idx = np.where(profile <= flow_height_threshold)[0]
                
                if len(idx) == 0:
                    max_front_idx = 0
                    list_row_with_front = [0]
                
                else:
                    if (idx[0] + idx_max) > max_front_idx :
                        max_front_idx = (idx[0] + idx_max)
                        list_row_with_front = [y]

                    elif (idx[0] + idx_max) == max_front_idx :
                        list_row_with_front.append(y)
            
            max_front_value = 0
            list_row_with_value = []
            for idx in list_row_with_front:
                val = h_2d_t[idx, max_front_idx - 1]
                if val > max_front_value:
                    max_front_value = val
                    list_row_with_value = [idx]
                elif val == max_front_value:
                    list_row_with_value.append(idx)
                        
            if len(list_row_with_value) > 1:
                max_index = np.unravel_index(np.argmax(h_2d_t[list_row_with_value, :]), h_2d_t.shape)
                direction_index[1] = list_row_with_value[max_index[0]]
            else:
                direction_index[1] = list_row_with_value[0]            

        if t in self._h_num_1d_X:
            if not any(res[0] == self._current_model for res in self._h_num_1d_X[t]):
                self._h_num_1d_X[t].append((self._current_model, h_2d_t[direction_index[1], :]))
        else:
            self._h_num_1d_X[t] = [(self._current_model, h_2d_t[direction_index[1], :])]
        
        # FRONT POSITIONS
        idx_max_X = np.argmax(h_2d_t[direction_index[1], :])
        idx_max_Y = np.argmax(h_2d_t[:, direction_index[0]])
        
        # RIGHT
        profile = h_2d_t[direction_index[1], idx_max_X:]
        idx = np.where(profile <= flow_height_threshold)[0]
        
        if len(idx) == 0:
            idx_r = x_size - 1
        else:
            idx_r = idx[0] + idx_max_X - 1
        
        # LEFT
        profile = h_2d_t[direction_index[1], :idx_max_X]
        idx = np.where(profile <= flow_height_threshold)[0]
        
        if len(idx) == 0:
            idx_l = 0
        else:
            idx_l = idx[-1] + 1
        
        # UP
        profile = h_2d_t[idx_max_Y:, direction_index[0]]
        idx = np.where(profile <= flow_height_threshold)[0]
        
        if len(idx) == 0:
            idx_u = y_size - 1
        else:
            idx_u = idx[0] + idx_max_Y - 1
        
        # LEFT
        profile = h_2d_t[:idx_max_Y, direction_index[0]]
        idx = np.where(profile <= flow_height_threshold)[0]
        
        if len(idx) == 0:
            idx_d = 0
        else:
            idx_d = idx[-1] + 1
        
        
        if t in self._h_num_1d_params:
            if not any(res[0] == self._current_model for res in self._h_num_1d_params[t]):
                self._h_num_1d_params[t].append((self._current_model, direction_index, [idx_l, idx_r, idx_d, idx_u]))
        else:
            self._h_num_1d_params[t] = [(self._current_model, direction_index, [idx_l, idx_r, idx_d, idx_u])]
        
        if show_message:
            print(self._current_model)
            print(f"Selected index:\n   X -> {direction_index[0]}\n   Y -> {direction_index[1]}")
            print(f"Front positions:\n  Right -> {idx_r}, {self._x[idx_r]}m\n  Left -> {idx_l}, {self._x[idx_l]}m\n  Up -> {idx_u}, {self._y[idx_u]}m\n  Down -> {idx_d}, {self._y[idx_d]}m")


    def extract_velocity_profiles(self,
                                  t: float = None,
                                  direction_index: list[int] = None,
                                  flow_velocity_threshold: float = 1e-3,
                                  show_message: bool = False
                                  ) -> None:
        """
        Extract 1D velocity profiles along X and Y axis at a specific time and store it for future use.

        Parameters
        ----------
        t : float, optional
            Time index to extract. If None, uses the last available time step.
        direction_index : tuple[int], optional
            Index along each axis to extract the profile from: (X, Y). If None, it is detected automatically based on the farthest flow front 
            and the position of maximum fluid velocity.
        flow_velocity_threshold : float, optional
            Minimum height to consider as part of the flow, by default 1e-3.
        show_message : bool, optional
            If True, print the indexes and the front positions saved. 

        Raises
        ------
        ValueError
            If no numerical result has been loaded.
        """
        if self._current_result is None:
            raise ValueError(" -> No result load, use first load_numerical_result.")
                
        if direction_index is None:
            direction_index = [None, None]
        
        if self._current_model == "shaltop":
            u_2d_all = self._current_result.get_output('ux')
            v_2d_all = self._current_result.get_output('uy')
        elif self._current_model == "lave2D":
            u_2d_all = self._current_result.get_output('u')
            v_2d_all = None
        elif self._current_model == "saval2D":
            u_2d_all = self._current_result.get_output('u')
            v_2d_all = self._current_result.get_output('v')
            
        t = u_2d_all.d.shape[2]-1 if t is None or t >= u_2d_all.d.shape[2] else t
        u_2d_t = u_2d_all.d[:, :, t]
        u_2d_t[u_2d_t <= flow_velocity_threshold] = 0
        
        y_size, x_size = u_2d_t.shape[0], u_2d_t.shape[1]
        
        # ALONG Y
        if direction_index[0] is None:
            max_front_idx = 0
            list_column_with_front = []
            
            for x in range(x_size):
                idx_max = np.argmax(u_2d_t[:, x])
                profile = u_2d_t[idx_max:, x]
                idx = np.where(profile < flow_velocity_threshold)[0]
                                
                if len(idx) == 0:
                    max_front_idx = 0
                    list_column_with_front = [0]
                
                else:
                    if (idx[0] + idx_max) > max_front_idx :
                        max_front_idx = (idx[0] + idx_max)
                        list_column_with_front = [x]
                    
                    elif (idx[0] + idx_max) == max_front_idx :
                        list_column_with_front.append(x)
                                    
            max_front_value = 0
            list_column_with_value = []
            for idx in list_column_with_front:
                val = u_2d_t[max_front_idx - 1, idx]
                if val > max_front_value:
                    max_front_value = val
                    list_column_with_value = [idx]
                elif val == max_front_value:
                    list_column_with_value.append(idx)
                            
            if len(list_column_with_value) > 1:
                max_index = np.unravel_index(np.argmax(u_2d_t[:, list_column_with_value]), u_2d_t.shape)
                direction_index[0] = list_column_with_value[max_index[1]]
            else:
                direction_index[0] = list_column_with_value[0]
            
        if t in self._u_num_1d_Y:
            if not any(res[0] == self._current_model for res in self._u_num_1d_Y[t]):
                self._u_num_1d_Y[t].append((self._current_model, u_2d_t[:, direction_index[0]]))
        else:
            self._u_num_1d_Y[t] = [(self._current_model, u_2d_t[:, direction_index[0]])]
        
        # ALONG X
        if direction_index[1] is None:
            max_front_idx = 0
            list_row_with_front = []
            
            for y in range(y_size):
                idx_max = np.argmax(u_2d_t[y, :])
                profile = u_2d_t[y, idx_max:]
                idx = np.where(profile <= flow_velocity_threshold)[0]
                
                if len(idx) == 0:
                    max_front_idx = 0
                    list_row_with_front = [0]
                
                else:
                    if (idx[0] + idx_max) > max_front_idx :
                        max_front_idx = (idx[0] + idx_max)
                        list_row_with_front = [y]

                    elif (idx[0] + idx_max) == max_front_idx :
                        list_row_with_front.append(y)
            
            max_front_value = 0
            list_row_with_value = []
            for idx in list_row_with_front:
                val = u_2d_t[idx, max_front_idx - 1]
                if val > max_front_value:
                    max_front_value = val
                    list_row_with_value = [idx]
                elif val == max_front_value:
                    list_row_with_value.append(idx)
                        
            if len(list_row_with_value) > 1:
                max_index = np.unravel_index(np.argmax(u_2d_t[list_row_with_value, :]), u_2d_t.shape)
                direction_index[1] = list_row_with_value[max_index[0]]
            else:
                direction_index[1] = list_row_with_value[0]            

        if t in self._u_num_1d_X:
            if not any(res[0] == self._current_model for res in self._u_num_1d_X[t]):
                self._u_num_1d_X[t].append((self._current_model, u_2d_t[direction_index[1], :]))
        else:
            self._u_num_1d_X[t] = [(self._current_model, u_2d_t[direction_index[1], :])]
        
        # FRONT POSITIONS
        idx_max_X = np.argmax(u_2d_t[direction_index[1], :])
        idx_max_Y = np.argmax(u_2d_t[:, direction_index[0]])
        
        # RIGHT
        profile = u_2d_t[direction_index[1], idx_max_X:]
        idx = np.where(profile <= flow_velocity_threshold)[0]
        
        if len(idx) == 0:
            idx_r = x_size - 1
        else:
            idx_r = idx[0] + idx_max_X - 1
        
        # LEFT
        profile = u_2d_t[direction_index[1], :idx_max_X]
        idx = np.where(profile <= flow_velocity_threshold)[0]
        
        if len(idx) == 0:
            idx_l = 0
        else:
            idx_l = idx[-1] + 1
        
        # UP
        profile = u_2d_t[idx_max_Y:, direction_index[0]]
        idx = np.where(profile <= flow_velocity_threshold)[0]
        
        if len(idx) == 0:
            idx_u = y_size - 1
        else:
            idx_u = idx[0] + idx_max_Y - 1
        
        # LEFT
        profile = u_2d_t[:idx_max_Y, direction_index[0]]
        idx = np.where(profile <= flow_velocity_threshold)[0]
        
        if len(idx) == 0:
            idx_d = 0
        else:
            idx_d = idx[-1] + 1
        
        if t in self._u_num_1d_params:
            if not any(res[0] == self._current_model for res in self._u_num_1d_params[t]):
                self._u_num_1d_params[t].append((self._current_model, direction_index, [idx_l, idx_r, idx_d, idx_u]))
        else:
            self._u_num_1d_params[t] = [(self._current_model, direction_index, [idx_l, idx_r, idx_d, idx_u])]
        
        if show_message:
            print(self._current_model, " -> U")
            print(f"Selected index:\n   X -> {direction_index[0]}\n   Y -> {direction_index[1]}")
            print(f"Front positions:\n  Right -> {idx_r}, {self._x[idx_r]}m\n  Left -> {idx_l}, {self._x[idx_l]}m\n  Up -> {idx_u}, {self._y[idx_u]}m\n  Down -> {idx_d}, {self._y[idx_d]}m")


        if v_2d_all is not None:
            v_2d_t = v_2d_all.d[:, :, t]
            v_2d_t[v_2d_t <= flow_velocity_threshold] = 0
        
            # ALONG Y
            if direction_index[0] is None:
                max_front_idx = 0
                list_column_with_front = []
                
                for x in range(x_size):
                    idx_max = np.argmax(v_2d_t[:, x])
                    profile = v_2d_t[idx_max:, x]
                    idx = np.where(profile < flow_velocity_threshold)[0]
                                    
                    if len(idx) == 0:
                        max_front_idx = 0
                        list_column_with_front = [0]
                    
                    else:
                        if (idx[0] + idx_max) > max_front_idx :
                            max_front_idx = (idx[0] + idx_max)
                            list_column_with_front = [x]
                        
                        elif (idx[0] + idx_max) == max_front_idx :
                            list_column_with_front.append(x)
                                        
                max_front_value = 0
                list_column_with_value = []
                for idx in list_column_with_front:
                    val = v_2d_t[max_front_idx - 1, idx]
                    if val > max_front_value:
                        max_front_value = val
                        list_column_with_value = [idx]
                    elif val == max_front_value:
                        list_column_with_value.append(idx)
                                
                if len(list_column_with_value) > 1:
                    max_index = np.unravel_index(np.argmax(v_2d_t[:, list_column_with_value]), v_2d_t.shape)
                    direction_index[0] = list_column_with_value[max_index[1]]
                else:
                    direction_index[0] = list_column_with_value[0]
                
            if t in self._v_num_1d_Y:
                if not any(res[0] == self._current_model for res in self._v_num_1d_Y[t]):
                    self._v_num_1d_Y[t].append((self._current_model, v_2d_t[:, direction_index[0]]))
            else:
                self._v_num_1d_Y[t] = [(self._current_model, v_2d_t[:, direction_index[0]])]
            
            # ALONG X
            if direction_index[1] is None:
                max_front_idx = 0
                list_row_with_front = []
                
                for y in range(y_size):
                    idx_max = np.argmax(v_2d_t[y, :])
                    profile = v_2d_t[y, idx_max:]
                    idx = np.where(profile <= flow_velocity_threshold)[0]
                    
                    if len(idx) == 0:
                        max_front_idx = 0
                        list_row_with_front = [0]
                    
                    else:
                        if (idx[0] + idx_max) > max_front_idx :
                            max_front_idx = (idx[0] + idx_max)
                            list_row_with_front = [y]

                        elif (idx[0] + idx_max) == max_front_idx :
                            list_row_with_front.append(y)
                
                max_front_value = 0
                list_row_with_value = []
                for idx in list_row_with_front:
                    val = v_2d_t[idx, max_front_idx - 1]
                    if val > max_front_value:
                        max_front_value = val
                        list_row_with_value = [idx]
                    elif val == max_front_value:
                        list_row_with_value.append(idx)
                            
                if len(list_row_with_value) > 1:
                    max_index = np.unravel_index(np.argmax(v_2d_t[list_row_with_value, :]), v_2d_t.shape)
                    direction_index[1] = list_row_with_value[max_index[0]]
                else:
                    direction_index[1] = list_row_with_value[0]            

            if t in self._v_num_1d_X:
                if not any(res[0] == self._current_model for res in self._v_num_1d_X[t]):
                    self._v_num_1d_X[t].append((self._current_model, v_2d_t[direction_index[1], :]))
            else:
                self._v_num_1d_X[t] = [(self._current_model, v_2d_t[direction_index[1], :])]
            
            # FRONT POSITIONS
            idx_max_X = np.argmax(v_2d_t[direction_index[1], :])
            idx_max_Y = np.argmax(v_2d_t[:, direction_index[0]])
            
            # RIGHT
            profile = v_2d_t[direction_index[1], idx_max_X:]
            idx = np.where(profile <= flow_velocity_threshold)[0]
            
            if len(idx) == 0:
                idx_r = x_size - 1
            else:
                idx_r = idx[0] + idx_max_X - 1
            
            # LEFT
            profile = v_2d_t[direction_index[1], :idx_max_X]
            idx = np.where(profile <= flow_velocity_threshold)[0]
            
            if len(idx) == 0:
                idx_l = 0
            else:
                idx_l = idx[-1] + 1
            
            # UP
            profile = v_2d_t[idx_max_Y:, direction_index[0]]
            idx = np.where(profile <= flow_velocity_threshold)[0]
            
            if len(idx) == 0:
                idx_u = y_size - 1
            else:
                idx_u = idx[0] + idx_max_Y - 1
            
            # LEFT
            profile = v_2d_t[:idx_max_Y, direction_index[0]]
            idx = np.where(profile <= flow_velocity_threshold)[0]
            
            if len(idx) == 0:
                idx_d = 0
            else:
                idx_d = idx[-1] + 1
            
            if t in self._v_num_1d_params:
                if not any(res[0] == self._current_model for res in self._v_num_1d_params[t]):
                    self._v_num_1d_params[t].append((self._current_model, direction_index, [idx_l, idx_r, idx_d, idx_u]))
            else:
                self._v_num_1d_params[t] = [(self._current_model, direction_index, [idx_l, idx_r, idx_d, idx_u])]
            
            if show_message:
                print(self._current_model, " -> V")
                print(f"Selected index:\n   X -> {direction_index[0]}\n   Y -> {direction_index[1]}")
                print(f"Front positions:\n  Right -> {idx_r}, {self._x[idx_r]}m\n  Left -> {idx_l}, {self._x[idx_l]}m\n  Up -> {idx_u}, {self._y[idx_u]}m\n  Down -> {idx_d}, {self._y[idx_d]}m")


    def extract_height_field(self,
                               t: float = None
                               ) -> None:
        """
        Extract and store the full 2D height field at a given time step and store it for future use.

        Parameters
        ----------
        t : float, optional
            Time index to extract. If None or invalid, uses the last available time step.

        Raises
        ------
        ValueError
            If no numerical result has been loaded.
        """
        if self._current_result is None:
            raise ValueError(" -> No result load, use first load_numerical_result.")

        h_2d_all = self._current_result.get_output("h")

        t = h_2d_all.d.shape[2]-1 if t is None or t >= h_2d_all.d.shape[2] else t
        h_2d_t = h_2d_all.d[:, :, t]
                    
        if t in self._h_num_2d:
            if not any(res[0] == self._current_model for res in self._h_num_2d[t]):
                self._h_num_2d[t].append((self._current_model, h_2d_t))
        else:
            self._h_num_2d[t] = [(self._current_model, h_2d_t)]
            

    def extract_velocity_field(self,
                               t: float = None
                               ) -> None:
        """
        Extract and store the full 2D velocity field at a given time step and store it for future use.

        Parameters
        ----------
        t : float, optional
            Time index to extract. If None or invalid, uses the last available time step.

        Raises
        ------
        ValueError
            If no numerical result has been loaded.
        """
        if self._current_result is None:
            raise ValueError(" -> No result load, use first load_numerical_result.")

        if self._current_model == "shaltop":
            u_2d_all = self._current_result.get_output('ux')
            v_2d_all = self._current_result.get_output('uy')
        elif self._current_model == "lave2D":
            u_2d_all = self._current_result.get_output('u')
            v_2d_all = None
        elif self._current_model == "saval2D":
            u_2d_all = self._current_result.get_output('u')
            v_2d_all = self._current_result.get_output('v')
        
        t = u_2d_all.d.shape[2]-1 if t is None or t >= u_2d_all.d.shape[2] else t
        u_2d_t = u_2d_all.d[:, :, t]
        
        if t in self._u_num_2d:
            if not any(res[0] == self._current_model for res in self._u_num_2d[t]):
                self._u_num_2d[t].append((self._current_model, u_2d_t))
        else:
            self._u_num_2d[t] = [(self._current_model, u_2d_t)]
        
        if v_2d_all is not None:
            v_2d_t = v_2d_all.d[:, :, t]
                    
            if t in self._v_num_2d:
                if not any(res[0] == self._current_model for res in self._v_num_2d[t]):
                    self._v_num_2d[t].append((self._current_model, v_2d_t))
            else:
                self._v_num_2d[t] = [(self._current_model, v_2d_t)]
      
    
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
        for i in range(len(T)):
            self._h_as_1d.append((T[i], solution.h[i]))
            self._u_as_1d.append((T[i], solution.u[i]))
    

    def show_height_profile(self,
                            model_to_plot: str,
                            axis: str = "X",
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
        
        if model_to_plot == 'as' and (axis == 'X' or axis == 'x'):
            if len(self._h_as_1d) == 0:
                raise ValueError(" -> No analytic solution computed.")
            else:
                h_plot = self._h_as_1d[:]
        
        elif model_to_plot in self._allowed_model:
            if axis == "Y" or axis == "y":
                if any(label == model_to_plot for lst in self._h_num_1d_Y.values() for label, _ in lst):
                    h_plot = [(t, val) for t, lst in self._h_num_1d_Y.items() for label, val in lst if label == model_to_plot]
                    profil_idx = [idx[0] for t, lst in self._h_num_1d_params.items() for label, idx, front in lst if label == model_to_plot]
                else:
                    raise ValueError(" -> Solution not extracted.")
            elif axis == "X" or axis == "x":
                if any(label == model_to_plot for lst in self._h_num_1d_X.values() for label, _ in lst):
                    h_plot = [(t, val) for t, lst in self._h_num_1d_X.items() for label, val in lst if label == model_to_plot]
                    profil_idx = [idx[1] for t, lst in self._h_num_1d_params.items() for label, idx, front in lst if label == model_to_plot]
                else:
                    raise ValueError(" -> Solution not extracted.")
            else:
                raise ValueError(" -> Incorrect axis: 'Y' or 'X'.")
        else:
            raise ValueError(f" -> No allowed model selected, choose between:\n       {self._allowed_model.append("as")}")

        if h_plot is not None:
            if ax is None:
                fig, ax = plt.subplots()
            
            if axis == "Y" or axis == "y":
                absci = self._y
                if profil_idx is not None:
                    for i in profil_idx:
                        profil_positions.append(float(self._x[i]))
                    print(f"Profiles' position: {profil_positions}m")

            elif axis == "X" or axis == "x":
                absci = self._x
                if profil_idx is not None:
                    for i in profil_idx:
                        profil_positions.append(float(self._y[i]))
                    print(f"Profiles' position: {profil_positions}m")

            if len(h_plot) == 1:
                ax.plot(absci, h_plot[0][1], color='black', linewidth=1, label=f"t={h_plot[0][0]}s")
            else:
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
                              velocity_axis: str='U',
                              axis: str="X",
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
        velocity_threshold : float, optional
            Threshold value where lower values ​​will be replaced by Nan, by default 1e-6.
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
        u_plot = None
        profil_idx = None
        profil_positions = []
        
        if model_to_plot == 'as':
            if len(self._u_as_1d) == 0:
                raise ValueError(" -> No analytic solution computed.")
            else:
                u_plot = self._u_as_1d[:]
        
        elif model_to_plot in self._allowed_model:
            if velocity_axis == 'U' or velocity_axis == 'u':
                if axis == "Y" or axis == "y":
                    if any(label == model_to_plot for lst in self._u_num_1d_Y.values() for label, _ in lst):
                        u_plot = [(t, val) for t, lst in self._u_num_1d_Y.items() for label, val in lst if label == model_to_plot]
                        profil_idx = [idx[0] for t, lst in self._u_num_1d_params.items() for label, idx, front in lst if label == model_to_plot]
                    else:
                        raise ValueError(" -> Solution not extracted.")
                elif axis == "X" or axis == "x":
                    # print(0)
                    # print(self._u_num_1d_X)
                    # print(self._u_num_1d_Y)
                    # print(self._v_num_1d_X)
                    # print(self._v_num_1d_Y)
                    if any(label == model_to_plot for lst in self._u_num_1d_X.values() for label, _ in lst):
                        u_plot = [(t, val) for t, lst in self._u_num_1d_X.items() for label, val in lst if label == model_to_plot]
                        profil_idx = [idx[1] for t, lst in self._u_num_1d_params.items() for label, idx, front in lst if label == model_to_plot]
                    else:
                        raise ValueError(" -> Solution not extracted.")
                else:
                    raise ValueError(" -> Incorrect axis: 'Y' or 'X'.")
            
            elif velocity_axis == 'V' or velocity_axis == 'v':
                if axis == "Y" or axis == "y":
                    if any(label == model_to_plot for lst in self._v_num_1d_Y.values() for label, _ in lst):
                        u_plot = [(t, val) for t, lst in self._v_num_1d_Y.items() for label, val in lst if label == model_to_plot]
                        profil_idx = [idx[0] for t, lst in self._v_num_1d_params.items() for label, idx, front in lst if label == model_to_plot]
                    else:
                        raise ValueError(" -> Solution not extracted.")
                elif axis == "X" or axis == "x":
                    if any(label == model_to_plot for lst in self._v_num_1d_X.values() for label, _ in lst):
                        u_plot = [(t, val) for t, lst in self._v_num_1d_X.items() for label, val in lst if label == model_to_plot]
                        profil_idx = [idx[1] for t, lst in self._v_num_1d_params.items() for label, idx, front in lst if label == model_to_plot]
                    else:
                        raise ValueError(" -> Solution not extracted.")
                else:
                    raise ValueError(" -> Incorrect axis: 'Y' or 'X'.")
            else:
                raise ValueError(" -> Incorrect velocity axis: 'U' or 'V'.")
        else:
            raise ValueError(f" -> No allowed model selected, choose between:\n       {self._allowed_model[:].append("as")}")

        if u_plot is not None:
            if ax is None:
                fig, ax = plt.subplots()
                
            if axis == "Y" or axis == "y":
                absci = self._y
                if profil_idx is not None:
                    for i in profil_idx:
                        profil_positions.append(float(self._x[i]))
                    print(f"Profiles' position: {profil_positions}m")

            elif axis == "X" or axis == "x":
                absci = self._x
                if profil_idx is not None:
                    print(len(self._x), len(self._y))
                    print(profil_idx)
                    for i in profil_idx:
                        profil_positions.append(float(self._y[i]))
                    print(f"Profiles' position: {profil_positions}m")

            if len(u_plot) == 1:
                u_plot[0][1][u_plot[0][1] <= velocity_threshold] = np.nan
                ax.plot(absci, u_plot[0][1], color='black', linewidth=1, label=f"t={u_plot[0][0]}s")
            else:
                if linestyles is None or len(linestyles)!=(len(u_plot)):
                    norm = plt.Normalize(vmin=min(t for t, _ in u_plot), vmax=max(t for t, _ in u_plot))
                    cmap = plt.cm.copper
                    
                for sol_idx, sol_val in enumerate(u_plot):
                    t_val = u_plot[sol_idx][0]
                    if linestyles is None or len(linestyles)!=(len(u_plot)):
                        color = cmap(norm(t_val)) if t_val != 0 else "red"
                        l_style = "-" if t_val != 0 else (0, (1, 4))
                    else:
                        color = "black" if t_val != 0 else "red"
                        l_style = linestyles[sol_idx] if t_val != 0 else (0, (1, 4))
                    sol_val[1][sol_val[1] <= velocity_threshold] = np.nan
                    ax.plot(absci, sol_val[1], color=color, linestyle=l_style, label=f"t={t_val}s")

            ax.grid(which='major')
            ax.grid(which='minor', alpha=0.5)
            ax.set_xlim(left=min(absci), right=max(absci))
            
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
            raise ValueError(f' -> Invalid direction: "right", "left", "up", "down"')
        
        if model_to_plot not in self._allowed_model:
            raise ValueError(f" -> No allowed model selected, choose between:\n       {self._allowed_model}")

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

            for lat_res in self._h_num_1d_Y[max_time]:
                if lat_res[0] == model_to_plot:
                    lat_profile = lat_res[1]
                    break  
        
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
        
        for params in self._h_num_1d_params[max_time]:
            if params[0] == model_to_plot:
                idx = params[1]
                fronts = params[2]
        
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

        if axes is None:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
            
        if direction == "up" or direction == "down" :
            axes[0].plot(self._y, front_profile, linestyle='-', color='black', label=self._current_model)
            axes[0].set_xlim(left=min(self._y), right=max(self._y))
            axes[0].set_xlabel(f"y [{x_unit}]")
 
        else:
            axes[0].plot(self._x, front_profile, linestyle='-', color='black', label=self._current_model)
            axes[0].set_xlim(left=min(self._x), right=max(self._x))
            axes[0].set_xlabel(f"x [{x_unit}]")
        
        axes[0].plot(morpho.x, morpho.h, linestyle="--", color='red', label="Coussot shape")
        axes[0].legend(loc='upper right')
        
        axes[0].grid(which='major')
        axes[0].grid(which='minor', alpha=0.5)
        
        axes[0].set_ylabel(f"h [{h_unit}]")
        axes[0].legend(loc='upper right')
        
        axes[0].set_title("Flow front")
        
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
        
        
        if direction == "up" or direction == "down" :
            axes[1].plot(self._x, lat_profile, linestyle='-', color='black', label=self._current_model)
            axes[1].set_xlim(left=min(self._x), right=max(self._x))
            axes[1].set_xlabel(f"x [{x_unit}]")
 
        else:
            axes[1].plot(self._y, lat_profile, linestyle='-', color='black', label=self._current_model)
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
            If no result extracted at the specified time step.
        ValueError
            If no result computed at the specified time step for the analytical solution.
        ValueError
            If the axis is incorrect.
        """
        if self._x is None:
            raise ValueError(" -> No solution extracted, first use load_numerical_result.")
        
        if axis == 'X' or axis == 'x':
            step = len(self._x) // nbr_point
            
            if time_step not in self._h_num_1d_X.keys():
                raise ValueError(" -> Time step not extracted.")
            
            if ax is None:
                fig, ax = plt.subplots()
            
            if plot_as:
                if any(res[0] == time_step for res in self._h_as_1d):
                    for t, h in self._h_as_1d:
                        if t == time_step:
                            ax.plot(self._x, h, linestyle="-", color='black', label="AS")
                            break
                else:
                    raise ValueError(" -> Time step not computed in analytical solution.")
        
            for model in models_to_plot:
                if any(res[0] == model for res in self._h_num_1d_X[time_step]):
                    for m, h in self._h_num_1d_X[time_step]:
                        if model == "shaltop" and model == m:
                            ax.plot(self._x[::step], h[::step], marker='.', linestyle='None', color='red', label=model)
                        elif model == "lave2D" and model == m:
                            ax.plot(self._x[::step], h[::step], marker='x', linestyle='None', color='green', label=model)
                        elif model == "saval2D" and model == m:
                            ax.plot(self._x[::step], h[::step], marker='*', linestyle='None', color='blue', label=model)

            ax.set_xlim(left=min(self._x), right=max(self._x))
            ax.set_xlabel(f"x [{x_unit}]")
        
        elif axis == 'Y' or axis == 'y':
            step = len(self._y) // nbr_point
            
            if time_step not in self._h_num_1d_Y.keys():
                raise ValueError(" -> Time step not extracted.")
            
            if ax is None:
                fig, ax = plt.subplots()
            
            # # NO SOLUTION ALONG AXIS Y
            # if plot_as:
            #     if any(res[0] == time_step for res in self._h_as_1d):
            #         for t, h in self._h_as_1d:
            #             if t == time_step:
            #                 ax.plot(self._x, h, linestyle="-", color='black', label="AS")
            #                 break
            #     else:
            #         raise ValueError(" -> Time step not computed in analytical solution.")
        
            for model in models_to_plot:
                if any(res[0] == model for res in self._h_num_1d_Y[time_step]):
                    for m, h in self._h_num_1d_Y[time_step]:
                        if model == "shaltop" and model == m:
                            ax.plot(self._y[::step], h[::step], marker='.', linestyle='None', color='red', label=model)
                        elif model == "lave2D" and model == m:
                            ax.plot(self._y[::step], h[::step], marker='x', linestyle='None', color='green', label=model)
                        elif model == "saval2D" and model == m:
                            ax.plot(self._y[::step], h[::step], marker='*', linestyle='None', color='blue', label=model)

            ax.set_xlim(left=min(self._y), right=max(self._y))
            ax.set_xlabel(f"y [{x_unit}]")

        else:
            raise ValueError(" -> Incorrect axis: 'Y' or 'X'.")
             
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
                                         velocity_axis: str = "U",
                                         axis: str = "X",
                                         velocity_threshold: float = 1e-6,
                                         plot_as: bool = False,
                                         nbr_point: int = 20,
                                         ax: matplotlib.axes._axes.Axes = None,
                                         x_unit:str = "m",
                                         h_unit:str = "m",
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
            If no result has been loaded.
        ValueError
            If no result extracted at the specified time step.
        ValueError
            If no result computed at the specified time step for the analytical solution.
        ValueError
            If the axis is incorrect.
        ValueError
            If the velocity axis is incorrect.
        """
        if self._x is None:
            raise ValueError(" -> No solution extracted, first use load_numerical_result.")
        
        if axis == 'X' or axis == 'x':
            step = len(self._x) // nbr_point
            
            if velocity_axis == 'U' or velocity_axis == 'u':
                if time_step not in self._u_num_1d_X.keys():
                    raise ValueError(" -> Time step not extracted.")
                
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
            
                for model in models_to_plot:
                    if any(res[0] == model for res in self._u_num_1d_X[time_step]):
                        for m, h in self._u_num_1d_X[time_step]:
                            if model == "shaltop" and model == m:
                                h[h <= velocity_threshold] = np.nan
                                ax.plot(self._x[::step], h[::step], marker='.', linestyle='None', color='red', label=model)
                            elif model == "lave2D" and model == m:
                                h[h <= velocity_threshold] = np.nan
                                ax.plot(self._x[::step], h[::step], marker='x', linestyle='None', color='green', label=model)
                            elif model == "saval2D" and model == m:
                                h[h <= velocity_threshold] = np.nan
                                ax.plot(self._x[::step], h[::step], marker='*', linestyle='None', color='blue', label=model)
            
            elif velocity_axis == 'V' or velocity_axis == 'v':
                if time_step not in self._v_num_1d_X.keys():
                    raise ValueError(" -> Time step not extracted.")
                
                if ax is None:
                    fig, ax = plt.subplots()
            
                for model in models_to_plot:
                    if any(res[0] == model for res in self._v_num_1d_X[time_step]):
                        for m, h in self._v_num_1d_X[time_step]:
                            if model == "shaltop" and model == m:
                                h[h <= velocity_threshold] = np.nan
                                ax.plot(self._x[::step], h[::step], marker='.', linestyle='None', color='red', label=model)
                            elif model == "lave2D" and model == m:
                                h[h <= velocity_threshold] = np.nan
                                ax.plot(self._x[::step], h[::step], marker='x', linestyle='None', color='green', label=model)
                            elif model == "saval2D" and model == m:
                                h[h <= velocity_threshold] = np.nan
                                ax.plot(self._x[::step], h[::step], marker='*', linestyle='None', color='blue', label=model)

            else:
                raise ValueError(" -> Incorrect velocity axis: 'U' or 'V'.")
            
            ax.set_xlim(left=min(self._x), right=max(self._x))
            ax.set_xlabel(f"x [{x_unit}]")
        
        elif axis == 'Y' or axis == 'y':
            step = len(self._y) // nbr_point

            if velocity_axis == 'U' or velocity_axis == 'u':
                if time_step not in self._u_num_1d_Y.keys():
                    raise ValueError(" -> Time step not extracted.")
                
                if ax is None:
                    fig, ax = plt.subplots()
            
                for model in models_to_plot:
                    if any(res[0] == model for res in self._u_num_1d_Y[time_step]):
                        for m, h in self._u_num_1d_Y[time_step]:
                            if model == "shaltop" and model == m:
                                h[h <= velocity_threshold] = np.nan
                                ax.plot(self._y[::step], h[::step], marker='.', linestyle='None', color='red', label=model)
                            elif model == "lave2D" and model == m:
                                h[h <= velocity_threshold] = np.nan
                                ax.plot(self._y[::step], h[::step], marker='x', linestyle='None', color='green', label=model)
                            elif model == "saval2D" and model == m:
                                h[h <= velocity_threshold] = np.nan
                                ax.plot(self._y[::step], h[::step], marker='*', linestyle='None', color='blue', label=model)

            elif velocity_axis == 'V' or velocity_axis == 'v':
                if time_step not in self._v_num_1d_Y.keys():
                    raise ValueError(" -> Time step not extracted.")
                
                if ax is None:
                    fig, ax = plt.subplots()
            
                for model in models_to_plot:
                    if any(res[0] == model for res in self._v_num_1d_Y[time_step]):
                        for m, h in self._v_num_1d_Y[time_step]:
                            if model == "shaltop" and model == m:
                                h[h <= velocity_threshold] = np.nan
                                ax.plot(self._y[::step], h[::step], marker='.', linestyle='None', color='red', label=model)
                            elif model == "lave2D" and model == m:
                                h[h <= velocity_threshold] = np.nan
                                ax.plot(self._y[::step], h[::step], marker='x', linestyle='None', color='green', label=model)
                            elif model == "saval2D" and model == m:
                                h[h <= velocity_threshold] = np.nan
                                ax.plot(self._y[::step], h[::step], marker='*', linestyle='None', color='blue', label=model)

            else:
                raise ValueError(" -> Incorrect velocity axis: 'U' or 'V'.")
            
            ax.set_xlim(left=min(self._y), right=max(self._y))
            ax.set_xlabel(f"y [{x_unit}]")

        else:
            raise ValueError(" -> Incorrect axis: 'Y' or 'X'.")
             
        ax.grid(which='major')
        ax.grid(which='minor', alpha=0.5)
        
        ax.set_title(f"Velocity comparison along {axis} for t={time_step}s")
        
        ax.set_ylabel(f"h [{h_unit}]")
        ax.legend(loc='upper right')
        
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


    def show_height_field(self,
                          model_to_plot: str,
                          t: float,
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
            If no numerical result has been extracted.
        """
        if t not in self._h_num_2d:
            raise ValueError(" -> No solution extracted at specified time, use extract_height_field")
        
        if not any(label == model_to_plot for lst in self._h_num_2d.values() for label, _ in lst):
            raise ValueError(" -> No solution extracted for this model, load and use extract_height_field")
        
        if ax is None:
            fig, ax = plt.subplots()
        
        for lst in self._h_num_2d[t]:
            if lst[0] == model_to_plot:
                pyplt.plot_data_on_topo(self._x, self._y, self._z, lst[1], axe=ax)
                break
            
        if show_plot:
            plt.show()
        
        return ax
  

    def show_velocity_field(self,
                            model_to_plot: str,
                            t: float,
                            velocity_axis: str = 'U',
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
            If no numerical result has been extracted.
        ValueError
            If the velocity axis is incorrect.
        """
        if velocity_axis == 'U' or velocity_axis == 'u':
            if t not in self._u_num_2d:
                raise ValueError(" -> No solution extracted, use extract_height_field")
            
            if not any(label == model_to_plot for lst in self._u_num_2d.values() for label, _ in lst):
                raise ValueError(" -> No solution extracted for this model, load and use extract_height_field")
            
            if ax is None:
                fig, ax = plt.subplots()
            
            for lst in self._u_num_2d[t]:
                if lst[0] == model_to_plot:
                    pyplt.plot_data_on_topo(self._x, self._y, self._z, lst[1], axe=ax)
                    break
                
        elif velocity_axis == 'V' or velocity_axis == 'v':
            if t not in self._v_num_2d:
                raise ValueError(" -> No solution extracted, use extract_height_field")
            
            if not any(label == model_to_plot for lst in self._v_num_2d.values() for label, _ in lst):
                raise ValueError(" -> No solution extracted for this model, load and use extract_height_field")
            
            if ax is None:
                fig, ax = plt.subplots()
            
            for lst in self._v_num_2d[t]:
                if lst[0] == model_to_plot:
                    pyplt.plot_data_on_topo(self._x, self._y, self._z, lst[1], axe=ax)
                    break
        
        else:
            raise ValueError(" -> Incorrect velocity axis: 'U' or 'V'.")
        
        if show_plot:
            plt.show()
        
        return ax
      
        
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

