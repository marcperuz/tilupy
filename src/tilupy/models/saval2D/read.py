# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 11:07:44 2024

@author: peruzzetto
"""

import os

import numpy as np
import tilupy.read


class Results(tilupy.read.Results):
    def __init__(self, folder, raster_topo="mntsimulation.asc"):
        super().__init__()

        self.folder = folder
        self.folder_output = folder
        self.folder_outputs = os.path.join(folder, "outputs") if os.path.exists(os.path.join(folder, "outputs")) else None
        self.folder_inputs = os.path.join(folder, "inputs") if os.path.exists(os.path.join(folder, "inputs")) else None
        
        self._u = None
        
        self._ux = None
        self._uy = None

        if not raster_topo.endswith(".asc"):
            raster_topo = raster_topo + ".asc"
        self.raster = raster_topo

        self.params = dict()

        if self.folder_inputs is not None:
            self.x, self.y, self._zinit = tilupy.raster.read_ascii(
                os.path.join(self.folder_inputs, self.raster)
            )
        
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

        self.params["nx"] = len(self.x)
        self.params["ny"] = len(self.y)

        # Rheology
        rheol = self.extract_rheological_parameters(os.path.join(self.folder_outputs, "log.txt"))
        self.params["tau/rho"] = rheol[0]
        self.params["mu"] = rheol[1]
        self.params["xi"] = rheol[2]
    

    def extract_rheological_parameters(self, path_to_log):
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
    
    
    def extract_saval_ascii(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
        start_index = 6
        
        # Extraire uniquement les lignes du tableau
        h_str = lines[start_index:]

        # Optionnel : convertir en liste de listes de floats
        h = [list(map(float, line.split())) for line in h_str if line.strip()]
        return np.array(h)
        
    
    def extract_times(self, path):
        t_list = []
        with open(path, 'r') as f:
            lines = f.readlines()
        
        start_index = 17
        for line in lines[start_index::2]:
            if line.strip() == '':
                continue
            else:
                _, t, _, _, _ = line.split()
                t_list.append(int(float(t)))
        
        return t_list            
    
    
    def read_resfile(
        self,
    ):  
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
        
        for t in t_list:
            h_t = self.extract_saval_ascii(os.path.join(self.folder_outputs, f"resuh{t}.asc"))
            qu_t = self.extract_saval_ascii(os.path.join(self.folder_outputs, f"resuqu{t}.asc"))
            qv_t = self.extract_saval_ascii(os.path.join(self.folder_outputs, f"resuqv{t}.asc"))
            
            ux_t = np.divide(qu_t, h_t, out=np.zeros_like(qu_t), where=h_t != 0)
            uy_t = np.divide(qv_t, h_t, out=np.zeros_like(qv_t), where=h_t != 0)
            
            u_t = np.sqrt(ux_t**2 + uy_t**2)
            
            h_list.append(h_t)
            
            ux_list.append(ux_t)
            uy_list.append(uy_t)
            u_list.append(u_t)
            self._tim.append(t)
        
        self._h = np.stack(h_list, axis=-1)
        self._ux = np.stack(ux_list, axis=-1)
        self._uy = np.stack(uy_list, axis=-1)
        self._u = np.stack(u_list, axis=-1)


    @property
    def h(self):
        if self._h is None:
            self.read_resfile()
        return self._h

    @property
    def u(self):
        if self._u is None:
            self.read_resfile()
        return self._u

    @property
    def ux(self):
        if self._ux is None:
            self.read_resfile()
        return self._ux

    @property
    def uy(self):
        if self._uy is None:
            self.read_resfile()
        return self._uy

    @property
    def tim(self):
        if self._tim is None:
            self.read_resfile()
        return self._tim

    @tim.setter
    def tim(self, value):
        self._tim = value

    def _get_output(self, name, **kwargs):
        d = None
        t = None
        notation = None

        if name in ["h", "u", "ux", "uy"]:
            d = getattr(self, name)
            t = self._tim
            return tilupy.read.TemporalResults2D(
                name, d, t, notation=notation, x=self.x, y=self.y, z=self.z
            )
