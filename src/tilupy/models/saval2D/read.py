# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 11:07:44 2024

@author: peruzzetto
"""

import os

import numpy as np
import tilupy.read


STATES_OUTPUT = ["h", "ux", "uy", "hvert"]


def extract_saval_ascii(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    start_index = 6
    
    h_str = lines[start_index:]

    h = [list(map(float, line.split())) for line in h_str if line.strip()]
    return np.array(h)


class Results(tilupy.read.Results):
    def __init__(self, folder, raster_topo="mntsimulation.asc"):
        super().__init__()
        self._code = "saval2D"
        
        self._folder = folder
        self._folder_output = folder
        
        self._simu_inputs = os.path.join(folder, "inputs") if os.path.exists(os.path.join(folder, "inputs")) else None
        self._simu_outputs = os.path.join(folder, "outputs") if os.path.exists(os.path.join(folder, "outputs")) else None
        
        if not raster_topo.endswith(".asc"):
            raster_topo = raster_topo + ".asc"
        self._raster = raster_topo

        self._params = dict()

        if self._simu_inputs is not None:
            self._x, self._y, self._zinit = tilupy.raster.read_ascii(os.path.join(self._simu_inputs, 
                                                                                  self._raster))
        self._nx, self._ny = len(self._x), len(self._y)
        self._dx = self._x[1] - self._x[0]
        self._dy = self._y[1] - self._y[0]

        self._params["nx"] = len(self.x)
        self._params["ny"] = len(self.y)

        # Rheology
        rheol = self.get_rheological_parameters(os.path.join(self._simu_outputs, "log.txt"))
        self._params["tau/rho"] = rheol[0]
        self._params["mu"] = rheol[1]
        self._params["xi"] = rheol[2]
    

    def get_rheological_parameters(self, path_to_log):
        values = None
        with open(path_to_log, "r") as f:
            for line in f:
                line = line.strip()
                try:
                    parts = list(map(float, line.split()))
                    if len(parts) == 3:
                        values = parts
                        break
                except ValueError:
                    continue
        return values
        
    
    def get_times(self, path):
        t_list = []
        with open(path, 'r') as f:
            lines = f.readlines()
        
        start_index = 16
        for line in lines[start_index:]:
            if line.strip() == '':
                continue
            else:
                _, t, _, _, _ = line.split()
                t_list.append(float(t))
        
        return t_list            
    
    
    def _extract_output(self, name):
        # Read thicknesses or velocity components
        d = None
        t = None
        notation = None
        
        tim = [0]
        t_list = self.get_times(os.path.join(self._simu_outputs, "log.txt"))
        
        _, _, h_init = tilupy.raster.read_ascii(os.path.join(self._simu_inputs, 
                                                                "zdepsimulation.asc"))
        
        h_list = [h_init]
        
        ux_list = []
        uy_list = []
        u_list = []
        ux_list.append(np.zeros_like(h_init))
        uy_list.append(np.zeros_like(h_init))
        u_list.append(np.zeros_like(h_init))
        
        qx_list = []
        qy_list = []
        q_list = []
        qx_list.append(np.zeros_like(h_init))
        qy_list.append(np.zeros_like(h_init))
        q_list.append(np.zeros_like(h_init))
        
        for t in range(len(t_list)):
            h_t = extract_saval_ascii(os.path.join(self._simu_outputs, f"resuh{t+1}.asc"))
            qu_t = extract_saval_ascii(os.path.join(self._simu_outputs, f"resuqu{t+1}.asc"))
            qv_t = extract_saval_ascii(os.path.join(self._simu_outputs, f"resuqv{t+1}.asc"))

            h_t[h_t<0.0001] = 0
            
            ux_t = np.divide(qu_t, h_t, out=np.zeros_like(qu_t), where=h_t != 0)
            uy_t = np.divide(qv_t, h_t, out=np.zeros_like(qv_t), where=h_t != 0)
            
            u_t = np.sqrt(ux_t**2 + uy_t**2)
            q_t = np.sqrt(qu_t**2 + qv_t**2)
            
            h_list.append(h_t)
            
            ux_list.append(ux_t)
            uy_list.append(uy_t)
            u_list.append(u_t)
            
            qx_list.append(qu_t)
            qy_list.append(qv_t)
            q_list.append(q_t)
            
            tim.append(t_list[t])

        self._tim = tim
    
        if name == "h":
            d = np.stack(h_list, axis=-1)
            t = self._tim
        
        if name == "u":
            d = np.stack(u_list, axis=-1)
            t = self._tim
        
        if name == "q":
            d = np.stack(q_list, axis=-1)
            t = self._tim
        
        if name == "ux":
            d = np.stack(ux_list, axis=-1)
            t = self._tim
        
        if name == "uy":
            d = np.stack(uy_list, axis=-1)
            t = self._tim
            
        if name == "qx":
            d = np.stack(qx_list, axis=-1)
            t = self._tim
        
        if name == "qy":
            d = np.stack(qy_list, axis=-1)
            t = self._tim
            
        if t is None:
            return tilupy.read.AbstractResults(name, d, notation=notation)

        else:
            if d.ndim == 3:
                return tilupy.read.TemporalResults2D(name, 
                                                     d, 
                                                     t, 
                                                     notation=notation, 
                                                     x=self._x, 
                                                     y=self._y, 
                                                     z=self._zinit)
            if d.ndim == 1:
                return tilupy.read.TemporalResults0D(name, 
                                                     d, 
                                                     t, 
                                                     notation=notation)
        return None
        
    """
    
        h_list = []
        ux_list = []
        uy_list = []
        u_list = []
        
        # read initial mass
        _, _, h_init = tilupy.raster.read_ascii(os.path.join(self.folder_inputs, "zdepsimulation.asc"))
        h_list.append(h_init)
        ux_list.append(np.zeros_like(h_init))
        uy_list.append(np.zeros_like(h_init))
        u_list.append(np.zeros_like(h_init))
        
        # Read results
        self._tim = [0]
        t_list = self.extract_times(os.path.join(self.folder_outputs, "log.txt"))
                
        for t in range(len(t_list)):
            h_t = self.extract_saval_ascii(os.path.join(self.folder_outputs, f"resuh{t+1}.asc"))
            qu_t = self.extract_saval_ascii(os.path.join(self.folder_outputs, f"resuqu{t+1}.asc"))
            qv_t = self.extract_saval_ascii(os.path.join(self.folder_outputs, f"resuqv{t+1}.asc"))
            
            # qu_t[qu_t<0.001] = 0
            # qv_t[qv_t<0.001] = 0
            h_t[h_t<0.0001] = 0
            
            ux_t = np.divide(qu_t, h_t, out=np.zeros_like(qu_t), where=h_t != 0)
            uy_t = np.divide(qv_t, h_t, out=np.zeros_like(qv_t), where=h_t != 0)
            
            u_t = np.sqrt(ux_t**2 + uy_t**2)
            
            h_list.append(h_t)
            
            ux_list.append(ux_t)
            uy_list.append(uy_t)
            u_list.append(u_t)
            self._tim.append(t_list[t])
        
        self._h = np.stack(h_list, axis=-1)
        self._ux = np.stack(ux_list, axis=-1)
        self._uy = np.stack(uy_list, axis=-1)
        self._u = np.stack(u_list, axis=-1)

    """
        
    # def _get_output(self, name, **kwargs):
    #     d = None
    #     t = None
    #     notation = None

    #     if name in ["h", "u", "ux", "uy"]:
    #         d = getattr(self, name)
    #         t = self._tim
    #         return tilupy.read.TemporalResults2D(
    #             name, d, t, notation=notation, x=self.x, y=self.y, z=self.z
    #         )

    # @property
    # def h(self):
    #     if self._h is None:
    #         self.read_resfile()
    #     return self._h

    # @property
    # def u(self):
    #     if self._u is None:
    #         self.read_resfile()
    #     return self._u

    # @property
    # def ux(self):
    #     if self._ux is None:
    #         self.read_resfile()
    #     return self._ux

    # @property
    # def uy(self):
    #     if self._uy is None:
    #         self.read_resfile()
    #     return self._uy

    # @property
    # def tim(self):
    #     if self._tim is None:
    #         self.read_resfile()
    #     return self._tim

    # @tim.setter
    # def tim(self, value):
    #     self._tim = value

