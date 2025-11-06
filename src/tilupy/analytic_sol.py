# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib

from scipy.optimize import fsolve

from abc import ABC, abstractmethod


class Depth_result(ABC):
    """Abstract base class representing simulation results for flow depth and velocity.

    This class defines a common interface for analytical solution that compute flow height 
    h(x,t) and flow velocity u(x,t).

    Parameters
    ----------
        theta : float, optional
            Angle of the surface, in radian, by default 0.
    
    Attributes
    ----------
        _g = 9.81 : float 
            Gravitational constant.
        _theta : float
            Angle of the surface, in radian.
        _x : float or np.ndarray
            Spatial coordinates.
        _t : float or np.ndarray
            Time instant.
        _h : np.ndarray
            Flow height depending on space at a moment.
        _u : np.ndarray
            Flow velocity depending on space at a moment.
    """
    def __init__(self, 
                 theta: float=None
                 ):
        self._g = 9.81
        self._theta = theta
        self._x = None
        self._t = None
        self._h = None
        self._u = None


    @abstractmethod
    def compute_h(self,
                  x: float | np.ndarray, 
                  t: float | np.ndarray
                  ) -> None:
        """Virtual function that compute the flow height :attr:`_h` at given space and time.

        Parameters
        ----------
        x : float or np.ndarray
            Spatial coordinates.
        t : float or np.ndarray
            Time instant.
        """
        pass


    @abstractmethod
    def compute_u(self,
                  x: float | np.ndarray,
                  t: float | np.ndarray
                  ) -> None:
        """Virtual function that compute the flow velocity :attr:`_u` at given space and time.

        Parameters
        ----------
        x : float or np.ndarray
            Spatial coordinates.
        t : float or np.ndarray
            Time instant.
        """
        pass


    @property
    def h(self):
        """Accessor of h(x,t) solution.
        
        Returns
        -------
        numpy.ndarray
            Attribute :attr:`_h`. If None, no solution computed.
        """
        return self._h


    @property
    def u(self):
        """Accessor of u(x,t) solution.
        
        Returns
        -------
        numpy.ndarray
            Attribute :attr:`_u`. If None, no solution computed.
        """
        return self._u


    @property
    def x(self):
        """Accessor of the spatial distribution of the computed solution.
        
        Returns
        -------
        numpy.ndarray
            Attribute :attr:`_x`. If None, no solution computed.
        """
        return self._x


    @property
    def t(self):
        """Accessor of the time instant of the computed solution.
        
        Returns
        -------
        float or numpy.ndarray
            Attribut :attr:`_t`. If None, no solution computed.
        """
        return self._t


    def plot(self, 
             show_h: bool=False, 
             show_u: bool=False,
             show_surface: bool=False,
             linestyles: list[str]=None,
             x_unit:str = "m",
             h_unit:str = "m",
             u_unit:str = "m/s",
             show_plot:bool = True,
             figsize:tuple = None,
             ) -> matplotlib.axes._axes.Axes:
        """Plot the simulation results.

        Parameters
        ----------
        show_h : bool, optional
            If True, plot the flow height (:attr:`_h`) curve.
        show_u : bool, optional
            If True, plot the flow velocity (:attr:`_u`) curve.
        show_surface : bool, optional
            If True, plot the slop of the surface.
        linestyles : list[str], optional
            List of linestyle to applie to the graph, must have the same since as the numbre of curve to plot or it 
            will not be taken into account (-1), by default None. If None, copper colormap will be applied.
        x_unit: str
            Space unit.
        h_unit: str
            Height unit.
        u_unit: str
            Velocity unit.
        show_plot: bool, optional
            If True, show the resulting plot. By default True.
        figsize: tuple, optional
            Size of the wanted plot, by default None.
        
        Return
        ------
        matplotlib.axes._axes.Axes
            Resulting plot.
        
        Raises
        ------
        ValueError
            If no solution computed (:attr:`_h` and :attr:`_u` are None).
        """
        z_surf = [0, 0]
        
        if self._h is None and self._u is None:
            raise ValueError("No solution computed.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if show_h and self._h is not None:
            if self._h.ndim == 1:
                ax.plot(self._x, self._h, color='black', linewidth=1)
            else:
                if linestyles is None or len(linestyles)!=(len(self._t)):
                    norm = plt.Normalize(vmin=min(self._t), vmax=max(self._t))
                    cmap = plt.cm.copper
                    
                for h_idx, h_val in enumerate(self._h):
                    t_val = self._t[h_idx]
                    if linestyles is None or len(linestyles)!=(len(self._t)):
                        color = cmap(norm(t_val)) if t_val != 0 else "red"
                        l_style = "-" if t_val != 0 else (0, (1, 4))
                    else:
                        color = "black" if t_val != 0 else "red"
                        l_style = linestyles[h_idx] if t_val != 0 else (0, (1, 4))    
                    ax.plot(self._x, h_val, color=color, linestyle=l_style, label=f"t={t_val}s")
            
            if show_surface:
                ax.plot([self._x[0], self._x[-1]], z_surf, color='black', linewidth=2)
            
            ax.grid(which='major')
            ax.grid(which='minor', alpha=0.5)
            ax.set_xlim(left=min(self._x), right=max(self._x))
            
            ax.set_title(f"Flow height for t={self._t}")
            ax.set_xlabel(f"x [{x_unit}]")
            ax.set_ylabel(f"h [{h_unit}]")
            ax.legend(loc='upper right')
            if show_plot:
                plt.show()
            
            return ax

        if show_u and self._u is not None:
            if self._u.ndim == 1:
                ax.plot(self._x, self._u, color='black', linewidth=1)
                
            else:
                if linestyles is None or len(linestyles)!=(len(self._t)):
                    norm = plt.Normalize(vmin=min(self._t), vmax=max(self._t))
                    cmap = plt.cm.copper
                    
                for u_idx, u_val in enumerate(self._u):
                    t_val = self._t[u_idx]
                    if t_val == 0:
                        continue
                    if linestyles is None or len(linestyles)!=(len(self._t)):
                        color = cmap(norm(t_val))
                        l_style = "-"
                    else:
                        color = "black"
                        l_style = linestyles[u_idx]
                    ax.plot(self._x, u_val, color=color, linestyle=l_style, label=f"t={t_val}s")

            ax.grid(which='major')
            ax.grid(which='minor', alpha=0.5)
            ax.set_xlim(left=min(self._x), right=max(self._x))
            
            ax.set_title(f"Flow velocity for t={self._t}")
            ax.set_xlabel(f"x [{x_unit}]")
            ax.set_ylabel(f"u [{u_unit}]")
            ax.legend(loc='best')
            if show_plot:
                plt.show()
            
            return ax


class Ritter_dry(Depth_result):
    r"""Dam-break solution on a dry domain using shallow water theory.

    This class implements the 1D analytical Ritter's solution of an ideal dam break on a dry domain.
    The dam break is instantaneous, over an horizontal and flat surface with no friction.
    It computes the flow height (took verticaly) and velocity over space and time, based on the equation implemanted
    in SWASHES, based on Ritter's equation.
    
    Ritter, A., 1892, Die Fortpflanzung der Wasserwellen, Zeitschrift des Vereines Deutscher Ingenieure, vol. 36(33), p. 947-954.

    Parameters
    ----------
        x_0 : float, optional
            Initial dam location (position along x-axis), by default 0.
        h_0 : float
            Initial water depth to the left of the dam.

    Attributes
    ----------
        _x0 : float 
            Initial dam location (position along x-axis).
        _h0 : float
            Initial water depth to the left of the dam.
    """
    def __init__(self,
                 h_0: float,  
                 x_0: float=0, 
                 ):
        super().__init__()
        self._x0 = x_0
        self._h0 = h_0


    def xa(self, t: float) -> float:
        r"""
        Position of the rarefaction wave front (left-most edge) :
        
        .. math::
            x_A(t) = x_0 - t \sqrt{g h_0}

        Parameters
        ----------
        t : float
            Time instant.

        Returns
        -------
        float
            Position of the front edge of the rarefaction wave.
        """
        return self._x0 - (t * np.sqrt(self._g*self._h0))


    def xb(self, t: float) -> float:
        r"""
        Position of the contact discontinuity:
        
        .. math::
            x_B(t) = x_0 + 2 t \sqrt{g h_0}

        Parameters
        ----------
        t : float
            Time instant.

        Returns
        -------
        float
            Position of the contact wave (end of rarefaction).
        """
        return self._x0 + (2 * t * np.sqrt(self._g*self._h0))


    def compute_h(self, 
                  x: float | np.ndarray, 
                  T: float | np.ndarray
                  ) -> None:
        r"""Compute the flow height h(x, t) at given time and positions.

        .. math::
                h(x, t) = 
                \begin{cases}
                    h_0 & \text{if } x \leq x_A(t), \\\\
                    \frac{4}{9g} \left( \sqrt{g h_0} - \frac{x - x_0}{2t} \right)^2 & \text{if } x_A(t) < x \leq x_B(t), \\\\
                    0 & \text{if } x_B(t) < x,
                \end{cases}

        Parameters
        ----------
        x : float or np.ndarray
            Spatial positions.
        T : float or nd.ndarray
            Time instant.

        Notes
        -----
        Updates the internal :attr:`tilupy.analytic_sol.Depth_result._h`, :attr:`tilupy.analytic_sol.Depth_result._x`, :attr:`tilupy.analytic_sol.Depth_result._t` attributes with the computed result.
        """
        if isinstance(x, float):
            x = [x]
        if isinstance(T, float):
            T = [T]

        self._x = x
        self._t = T
        
        h = []
        for t in T:
            sub_h = []
            for i in x:
                if i <= self.xa(t):
                    sub_h.append(self._h0)
                elif self.xa(t) < i <= self.xb(t):
                    sub_h.append((4/(9*self._g)) *
                            (np.sqrt(self._g*self._h0)-((i-self._x0)/(2*t)))**2)
                else:
                    sub_h.append(0)
            h.append(sub_h)
        self._h = np.array(h)


    def compute_u(self, 
                  x: float | np.ndarray, 
                  T: float | np.ndarray
                  ) -> None:
        r"""Compute the flow velocity u(x, t) at given time and positions.

        .. math::
                u(x,t) = 
                \begin{cases}
                    0 & \text{if } x \leq x_A(t), \\\\
                    \frac{2}{3} \left( \frac{x - x_0}{t} + \sqrt{g h_0} \right) & \text{if } x_A(t) < x \leq x_B(t), \\\\
                    0 & \text{if } x_B(t) < x,
                \end{cases}

        Parameters
        ----------
        x : float or np.ndarray
            Spatial positions.
        T : float or np.ndarray
            Time instant.

        Notes
        -----
        Updates the internal :attr:`tilupy.analytic_sol.Depth_result._u`, :attr:`tilupy.analytic_sol.Depth_result._x`, :attr:`tilupy.analytic_sol.Depth_result._t` attributes with the computed result.
        """
        if isinstance(x, float):
            x = [x]
        if isinstance(T, float):
            T = [T]

        self._x = x
        self._t = T
        
        u = []
        for t in T:
            sub_u = []
            for i in x:
                if i <= self.xa(t):
                    sub_u.append(np.nan)
                elif i > self.xa(t) and i <= self.xb(t):
                    sub_u.append((2/3)*(((i-self._x0)/t) + np.sqrt(self._g*self._h0)))
                else:
                    sub_u.append(np.nan)
            u.append(sub_u)
        self._u = np.array(u)


class Stoker_SWASHES_wet(Depth_result):
    r"""Dam-break solution on a wet domain using shallow water theory.

    This class implements the 1D analytical Stoker's solution of an ideal dam break on a wet domain.
    The dam break is instantaneous, over an horizontal and flat surface with no friction.
    It computes the flow height (took verticaly) and velocity over space and time, based on the equation implemanted
    in SWASHES, based on Stoker's equation.
    
    Delestre, O., Lucas, C., Ksinant, P.-A., Darboux, F., Laguerre, C., Vo, T.-N.-T., James, F. & Cordier, S., 2013, SWASHES: a compilation of shallow water 
    analytic solutions for hydraulic and environmental studies, International Journal for Numerical Methods in Fluids, v. 72(3), p. 269-300, doi:10.1002/fld.3741.

    Stoker, J.J., 1957, Water Waves: The Mathematical Theory with Applications, Pure and Applied Mathematics, vol. 4, Interscience Publishers, New York, USA.

    Parameters
    ----------
        x_0 : float
            Initial dam location (position along x-axis).
        h_0 : float
            Water depth to the left of the dam.
        h_r : float
            Water depth to the right of the dam.
        h_m : float, optional
            Intermediate height used to compute the critical speed cm. If not provided,
            it will be computed numerically via the 'compute_cm()' method.
    
    Attributes
    ----------
        _x0 : float 
            Initial dam location (position along x-axis).
        _h0 : float
            Water depth to the left of the dam.
        _hr : float
            Water depth to the right of the dam.
        _cm : float
            Critical velocity. 
    """
    def __init__(self, 
                 x_0: float, 
                 h_0: float, 
                 h_r: float, 
                 h_m: float=None
                 ):
        super().__init__()
        self._x0 = x_0
        self._h0 = h_0
        self._hr = h_r
        self._cm = None
        self.compute_cm()

        if h_m is not None:
            self._cm = np.sqrt(self._g * h_m)


    def xa(self, t: float) -> float:
        r"""
        Position of the rarefaction wave front (left-most edge) :
        
        .. math::
            x_A(t) = x_0 - t \sqrt{g h_0}

        Parameters
        ----------
        t : float
            Time instant.

        Returns
        -------
        float
            Position of the front edge of the rarefaction wave.
        """
        return self._x0 - (t * np.sqrt(self._g*self._h0))


    def xb(self, t: float) -> float:
        r"""
        Position of the contact discontinuity:
        
        .. math::
            x_B(t) = x_0 + t \left( 2 \sqrt{g h_0} - 3 c_m \right)

        Parameters
        ----------
        t : float
            Time instant.

        Returns
        -------
        float
            Position of the contact wave (end of rarefaction).
        """
        return self._x0 + (t * ((2 * np.sqrt(self._g*self._h0)) - (3*self._cm)))


    def xc(self, t: float) -> float:
        r"""
        Position of the shock wave front (right-most wave):
        
        .. math::
            x_C(t) = x_0 + t \cdot \frac{2 c_m^2 \left( \sqrt{g h_0} - c_m \right)}{c_m^2 - g h_r}

        Parameters
        ----------
        t : float
            Time instant.

        Returns
        -------
        float
            Position of the shock front.
        """
        return self._x0 + (t * (((2*self._cm**2)*(np.sqrt(self._g*self._h0)-self._cm)) / ((self._cm**2) - (self._g*self._hr))))


    def equation_cm(self, cm) -> float:
        r"""Equation of the critical velocity cm:
        
        .. math::
            -8.g.hr.cm^{2}.(g.h0 - cm^{2})^{2} + (cm^{2} - g.hr)^{2} . (cm^{2} + g.hr) = 0

        Parameters
        ----------
        cm : float
            Trial value for :attr:`_cm`.

        Returns
        -------
        float
            Residual of the equation. Zero when :attr:`_cm` satisfies the system.
        """
        return -8 * self._g * self._hr * cm**2 * (self._g * self._h0 - cm**2)**2 + (cm**2 - self._g * self._hr)**2 * (cm**2 + self._g * self._hr)


    def compute_cm(self) -> None:
        r"""Solves the non-linear equation to compute the critical velocity :attr:`_cm`.

        Uses numerical root-finding to find a valid value of cm that separates
        the flow regimes. Sets :attr:`_cm` if a valid solution is found.
        """
        guesses = np.linspace(0.01, 1000, 1000)
        solutions = []

        for guess in guesses:
            sol = fsolve(self.equation_cm, guess)[0]

            if abs(self.equation_cm(sol)) < 1e-6 and not any(np.isclose(sol, s, atol=1e-6) for s in solutions):
                solutions.append(sol)

        for sol in solutions:
            hm = sol**2 / self._g
            if hm < self._h0 and hm > self._hr:
                find = True
                self._cm = sol
                break
            else:
                find = False

        if find:
            print(f"Find cm: {self._cm}\nhm:{self._cm**2 / self._g}")
        else:
            print(
                f"Didn't find cm, try with greater range of value. Default value: {None}")


    def compute_h(self, 
                  x: float | np.ndarray, 
                  T: float | np.ndarray
                  ) -> None:
        r"""Compute the flow height h(x, t) at given time and positions.

        .. math::
                h(x, t) = 
                \begin{cases}
                    h_0 & \text{if } x \leq x_A(t), \\\\
                    \frac{4}{9g} \left( \sqrt{g h_0} - \frac{x - x_0}{2t} \right)^2 & \text{if } x_A(t) < x \leq x_B(t), \\\\
                    \frac{c_m^2}{g} & \text{if } x_B(t) < x \leq x_C(t), \\\\
                    h_r & \text{if } x_C(t) < x,
                \end{cases}

        Parameters
        ----------
        x : float or np.ndarray
            Spatial positions.
        T : float or np.ndarray
            Time instant.

        Notes
        -----
        Updates the internal :attr:`tilupy.analytic_sol.Depth_result._h`, :attr:`tilupy.analytic_sol.Depth_result._x`, :attr:`tilupy.analytic_sol.Depth_result._t` attributes with the computed result.
        """
        if self._cm is not None:
            if isinstance(x, float):
                x = [x]
            self._x = x
            self._t = T

            if isinstance(T, float):
                h = []
                for i in x:
                    if i <= self.xa(T):
                        h.append(self._h0)
                    # elif i > self.xa(t) and i <= self.xb(t):
                    elif self.xa(T) < i <= self.xb(T):
                        h.append((4/(9*self._g))*(np.sqrt(self._g *
                                self._h0)-((i-self._x0)/(2*T)))**2)   # i-x0 and not i to recenter the breach of the dam at x=0.
                    elif self.xb(T) < i <= self.xc(T):
                        h.append((self._cm**2)/self._g)
                    else:
                        h.append(self._hr)
                self._h = np.array(h)
            else:
                h = []
                for t in T:
                    sub_h = []
                    for i in x:
                        if i <= self.xa(t):
                            sub_h.append(self._h0)
                        elif self.xa(t) < i <= self.xb(t):
                            sub_h.append((4/(9*self._g))*(np.sqrt(self._g *
                                    self._h0)-((i-self._x0)/(2*t)))**2)
                        elif self.xb(t) < i <= self.xc(t):
                            sub_h.append((self._cm**2)/self._g)
                        else:
                            sub_h.append(self._hr)
                    h.append(sub_h)
                self._h = np.array(h)

        else:
            print("No critical velocity found")


    def compute_u(self, 
                  x: float | np.ndarray, 
                  T: float | np.ndarray
                  ) -> None:
        r"""Compute the flow velocity u(x, t) at given time and positions.

        .. math::
                u(x,t) = 
                \begin{cases}
                    0 & \text{if } x \leq x_A(t), \\\\
                    \frac{2}{3} \left( \frac{x - x_0}{t} + \sqrt{g h_0} \right) & \text{if } x_A(t) < x \leq x_B(t), \\\\
                    2 \left( \sqrt{g h_0} - c_m \right) & \text{if } x_B(t) < x \leq x_C(t), \\\\
                    0 & \text{if } x_C(t) < x,
                \end{cases}

        Parameters
        ----------
        x : float or np.ndarray
            Spatial positions.
        T : float or np.ndarray
            Time instant.

        Notes
        -----
        Updates the internal :attr:`tilupy.analytic_sol.Depth_result._u`, :attr:`tilupy.analytic_sol.Depth_result._x`, :attr:`tilupy.analytic_sol.Depth_result._t` attributes with the computed result.
        """
        if self._cm is not None:
            if isinstance(x, float):
                x = [x]
            self._x = x
            self._t = T

            if isinstance(T, float):
                u = []
                for i in x:
                    if i <= self.xa(T):
                        u.append(0)
                    elif i > self.xa(T) and i <= self.xb(T):
                        u.append((2/3)*(((i-self._x0)/T) +
                                np.sqrt(self._g*self._h0)))
                    elif i > self.xb(T) and i <= self.xc(T):
                        u.append(2*(np.sqrt(self._g*self._h0) - self._cm))
                    else:
                        u.append(0)
                self._u = np.array(u)
            
            else:
                u = []
                for t in T:
                    sub_u = []
                    for i in x:
                        if i <= self.xa(t):
                            sub_u.append(0)
                        elif i > self.xa(t) and i <= self.xb(t):
                            sub_u.append((2/3)*(((i-self._x0)/t) +
                                    np.sqrt(self._g*self._h0)))
                        elif i > self.xb(t) and i <= self.xc(t):
                            sub_u.append(2*(np.sqrt(self._g*self._h0) - self._cm))
                        else:
                            sub_u.append(0)
                    u.append(sub_u)
                self._u = np.array(u)

        else:
            print("First define cm")


class Stoker_SARKHOSH_wet(Depth_result):
    r"""Dam-break solution on a wet domain using shallow water theory.

    This class implements the 1D analytical Stoker's solution of an ideal dam break on a wet domain.
    The dam break is instantaneous, over an horizontal and flat surface with no friction.
    It computes the flow height (took verticaly) and velocity over space and time, based on the equation implemanted
    in SWASHES, based on Stoker's equation.
    
    Sarkhosh, P., 2021, Stoker solution package, version 1.0.0, Zenodo. https://doi.org/10.5281/zenodo.5598374
    
    Stoker, J.J., 1957, Water Waves: The Mathematical Theory with Applications, Pure and Applied Mathematics, vol. 4, Interscience Publishers, New York, USA.
    
    Parameters
    ----------
        h_0 : float
            Initial water depth to the left of the dam.
        h_r : float
            Initial water depth to the right of the dam.

    Attributes
    ----------
        _h0 : float
            Water depth to the left of the dam.
        _hr : float
            Water depth to the right of the dam.
        _cm : float
            Shock front speed.
        _hm : float
            Height of the shock front.
    """
    def __init__(self, 
                 h_0: float, 
                 h_r: float, 
                 ):
        super().__init__()
        self._h0 = h_0
        self._hr = h_r
        
        if self._hr == 0:
            self._cm = 0
            self._hm = 0
        else:
            self._cm = self.compute_cm()
            self._hm = 0.5 * self._hr * (np.sqrt(1 + 8 * self._cm**2 / np.sqrt(self._g * self._hr)**2) - 1)


    def xa(self, t: float) -> float:
        r"""
        Position of the rarefaction wave front (left-most edge) :
        
        .. math::
            x_A(t) = x_0 - t \sqrt{g h_0}

        Parameters
        ----------
        t : float
            Time instant.

        Returns
        -------
        float
            Position of the front edge of the rarefaction wave.
        """
        return -(t * np.sqrt(self._g*self._h0))


    def xb(self, hm: float, t: float) -> float:
        r"""
        Position of the contact discontinuity:
        
        .. math::
            x_B(t) = t \left( 2 \sqrt{g h_0} - 3 \sqrt{g h_m} \right)

        Parameters
        ----------
        hm : float
            Height of the shock front.
        t : float
            Time instant.

        Returns
        -------
        float
            Position of the contact wave (end of rarefaction).
        """
        return (2 * np.sqrt(self._g * self._h0) - 3 * np.sqrt(self._g * hm)) * t


    def xc(self, cm: float, t: float) -> float:
        r"""
        Position of the shock wave front (right-most wave):
        
        .. math::
            x_C(t) = c_m t

        Parameters
        ----------
        cm : float
            Shock front speed.
        t : float
            Time instant.

        Returns
        -------
        float
            Position of the shock front.
        """
        return cm * t


    def compute_cm(self) -> float:
        r"""Compute the shock front speed using Newton-Raphson's method to find the solution of:
        
        .. math::
            c_m h_r - h_r \left( \sqrt{1 + \frac{8 c_m^2}{g h_r}} - 1 \right) \left( \frac{c_m}{2} - \sqrt{g h_0} + \sqrt{\frac{g h_r}{2} \left( \sqrt{1 + \frac{8 c_m^2}{g h_r}} - 1 \right)} \right) = 0
        
        Returns
        -------
        float
            Speed of the shock front.
        """
        f_cm = 1
        df_cm = 1
        cm = 10 * self._h0
        
        while abs(f_cm / cm) > 1e-10:
            root_term = np.sqrt(8 * cm**2 / np.sqrt(self._g * self._hr)**2 + 1)
            inner_sqrt = np.sqrt(self._g * self._hr * (root_term - 1) / 2)
            
            f_cm = cm * self._hr - self._hr * (root_term - 1) * (cm / 2 - np.sqrt(self._g * self._h0) + inner_sqrt)
            df_cm = (self._hr 
                     - self._hr * ((2 * cm * self._g * self._hr) / (np.sqrt(self._g * self._hr)**2 * root_term * inner_sqrt) + 0.5) * (root_term - 1)
                     - (8 * cm * self._hr * (cm / 2 - np.sqrt(self._g * self._h0) + inner_sqrt)) / (np.sqrt(self._g * self._hr)**2 * root_term))
            
            cm -= f_cm / df_cm

        return cm


    def compute_h(self, 
                  x: float | np.ndarray, 
                  T: float | np.ndarray
                  ) -> None:
        r"""Compute the flow height h(x, t) at given time and positions.

        .. math::
                h(x, t) = 
                \begin{cases}
                    h_0 & \text{if } x \leq x_A(t), \\\\
                    \frac{\left( 2 \sqrt{g h_0} - \frac{x}{t} \right)^2}{9 g} & \text{if } x_A(t) < x \leq x_B(t), \\\\
                    h_m = \frac{1}{2} h_r \left( \sqrt{1 + \frac{8 c_m^2}{g h_r}} - 1 \right) & \text{if } x_B(t) < x \leq x_C(t), \\\\
                    h_r & \text{if } x_C(t) < x
                \end{cases}

        Parameters
        ----------
        x : float or np.ndarray
            Spatial positions.
        T : float or np.ndarray
            Time instant.

        Notes
        -----
        Updates the internal :attr:`tilupy.analytic_sol.Depth_result._h`, :attr:`tilupy.analytic_sol.Depth_result._x`, :attr:`tilupy.analytic_sol.Depth_result._t` attributes with the computed result.
        """
        if isinstance(x, float):
            x = [x]
        if isinstance(T, float):
            T = [T]

        self._x = x
        self._t = T
        
        h = []
        for t in T:
            sub_h = []
            for i in x:
                if t == 0:
                    h_val = (2 * np.sqrt(self._g * self._h0) - 1e18) ** 2 / (9 * self._g)
                else:
                    h_val = (2 * np.sqrt(self._g * self._h0) - (i/t)) ** 2 / (9 * self._g)
                
                if i < self.xa(t):
                # if h_val >= self._h0:
                    h_val = self._h0
                
                if self._hm == 0 and h_val > sub_h[-1]:
                    h_val = 0
                else:
                    if (self.xb(self._hm, t) <  i <= self.xc(self._cm, t)) and h_val <= self._hm:
                        h_val = self._hm
                    elif i > self.xc(self._cm, t):
                        h_val = self._hr
                
                sub_h.append(h_val)
            h.append(sub_h)
        self._h = np.array(h)


    def compute_u(self, 
                  x: float | np.ndarray, 
                  T: float | np.ndarray
                  ) -> None:
        r"""Compute the flow velocity u(x, t) at given time and positions.

        .. math::
                u(x,t) = 
                \begin{cases}
                    0 & \text{if } x \leq x_A(t), \\\\
                    \frac{2}{3} \left( \frac{x}{t} + \sqrt{g h_0} \right) & \text{if } x_A(t) < x \leq x_B(t), \\\\
                    2 \sqrt{g h_0} - 2 \sqrt{g h_m} & \text{if } x_B(t) < x \leq x_C(t), \\\\
                    0 & \text{if } x_C(t) < x,
                \end{cases}

        Parameters
        ----------
        x : float or np.ndarray
            Spatial positions.
        T : float or np.ndarray
            Time instant.

        Notes
        -----
        Updates the internal :attr:`tilupy.analytic_sol.Depth_result._u`, :attr:`tilupy.analytic_sol.Depth_result._x`, :attr:`tilupy.analytic_sol.Depth_result._t` attributes with the computed result.
        """
        if isinstance(x, float):
            x = [x]
        if isinstance(T, float):
            T = [T]

        self._x = x
        self._t = T
        
        um = 2 * np.sqrt(self._g * self._h0) - 2 * np.sqrt(self._g * self._hm)
                
        u = []
        for t in T:
            sub_u = []
            for i in x:
                if t == 0:
                    u_val = 2 * (1e18 + np.sqrt(self._g * self._h0)) / 3
                else:
                    u_val = 2 * ((i/t) + np.sqrt(self._g * self._h0)) / 3
                
                if i < self.xa(t):
                # if h_val >= self._h0:
                    u_val = np.nan
                
                if self._hm == 0 and u_val > sub_u[-1]:
                    u_val = np.nan
                else:
                    if (self.xb(self._hm, t) <  i <= self.xc(self._cm, t)):
                        u_val = um
                    elif i > self.xc(self._cm, t):
                        u_val = np.nan
                
                sub_u.append(u_val)
            u.append(sub_u)
        self._u = np.array(u)


class Mangeney_dry(Depth_result):
    r"""Dam-break solution on an inclined dry domain with friction using shallow water theory.

    This class implements the 1D analytical Stoker's solution of an ideal dam break on a dry domain.
    The dam break is instantaneous, over an inclined and flat surface with friction.
    It computes the flow height (took normal to the surface) and velocity over space and time with an 
    infinitely-long fluid mass on an infinite surface.
    
    Mangeney, A., Heinrich, P., & Roche, R., 2000, Analytical solution for testing debris avalanche numerical models, 
    Pure and Applied Geophysics, vol. 157, p. 1081-1096.

    Parameters
    ----------
        x_0 : float
            Initial dam location (position along x-axis), by default 0.
        h_0 : float
            Initial water depth.
        theta : float
            Angle of the surface, in degree.
        delta : float
            Dynamic friction angle (20°-40° for debris avalanche), in degree.

    Attributes
    ----------
        _x0 : float 
            Initial dam location (position along x-axis).
        _h0 : float
            Initial water depth.
        _delta : float 
            Dynamic friction angle, in radian.
        _c0 : float
            Initial wave propagation speed.
        _m : float
            Constant horizontal acceleration of the front.
    """    
    def __init__(self,
                 x_0: float,
                 h_0: float,
                 theta: float,
                 delta: float,
                 ):
        super().__init__(theta=np.radians(theta))
        self._delta = np.radians(delta)
        self._h0 = h_0
        self._x0 = x_0
        self._c0 = self.compute_c0()
        self._m = self.compute_m()
        
        # print(f"delta: {self._delta}, theta: {self._theta}, m: {self._m}, c0: {self._c0}")


    def xa(self, t: float) -> float:
        r"""
        Edge of the quiet area:
        
        .. math::
            x_A(t) = x_0 + \frac{1}{2}mt^2 - c_0 t

        Parameters
        ----------
        t : float
            Time instant.

        Returns
        -------
        float
            Position of the edge of the quiet region.
        """
        return self._x0 + 0.5*self._m*t**2 - (self._c0*t)
    
    
    def xb(self, t: float) -> float:
        r"""
        Front of the flow:
        
        .. math::
            x_B(t) = x_0 + \frac{1}{2}mt^2 + 2 c_0 t

        Parameters
        ----------
        t : float
            Time instant.

        Returns
        -------
        float
            Position of the front edge of the fluid.
        """
        return self._x0 + 0.5*self._m*t**2 + (2*self._c0*t)


    def compute_c0(self) -> float:
        r"""Compute the initial wave propagation speed defined by:
        
        .. math::
            c_0 = \sqrt{g h_0 \cos{\theta}}
            
        Returns
        -------
        float
            Value of the initial wave propagation speed.
        """
        return np.sqrt(self._g * self._h0 * np.cos(self._theta))


    def compute_m(self) -> float:
        r"""Compute the constant horizontal acceleration of the front defined by:
        
        .. math::
            m = g \sin{\theta} - g \cos{\theta} \tan{\delta}
            
        Returns
        -------
        float
            Value of the constant horizontal acceleration of the front.
        """
        return (self._g * np.sin(self._theta)) - (self._g * np.cos(self._theta) * np.tan(self._delta))


    def compute_h(self, 
                  x: float | np.ndarray, 
                  T: float | np.ndarray) -> None:
        r"""Compute the flow height h(x, t) at given time and positions.

        .. math::
                h(x, t) = 
                \begin{cases}
                    h_0 & \text{if } x \leq x_A(t), \\\\
                    \frac{1}{9g cos(\theta)} \left(2 c_0 - \frac{x-x_0}{t}  + \frac{1}{2} m t \right)^2 & \text{if } x_A(t) < x \leq x_B(t), \\\\
                    0 & \text{if } x_B(t) < x,
                \end{cases}

        Parameters
        ----------
        x : float or np.ndarray
            Spatial positions.
        T : float or np.ndarray
            Time instant.

        Notes
        -----
        Updates the internal :attr:`tilupy.analytic_sol.Depth_result._h`, :attr:`tilupy.analytic_sol.Depth_result._x`, :attr:`tilupy.analytic_sol.Depth_result._t` attributes with the computed result.
        """
        if isinstance(x, float):
            x = [x]
        if isinstance(T, float):
            T = [T]

        self._x = x
        self._t = T
                
        h = []
        for t in T:
            sub_h = []
            
            for i in x:
                if i <= self.xa(t):
                    sub_h.append(self._h0)
                    
                elif self.xa(t) < i < self.xb(t):
                    sub_h.append( (1/(9*self._g*np.cos(self._theta))) * ( (-(i-self._x0)/t) + (2 * self._c0) + (0.5*t*self._m))**2 )

                else:
                    sub_h.append(0)
            h.append(sub_h)
            
        self._h = np.array(h)


    def compute_u(self, 
                  x: float | np.ndarray, 
                  T: float | np.ndarray) -> None:
        r"""Compute the flow velocity u(x, t) at given time and positions.

        .. math::
                u(x,t) = 
                \begin{cases}
                    0 & \text{if } x \leq x_A(t), \\\\
                    \frac{2}{3} \left( \frac{x-x_0}{t} + c_0 + mt \right) & \text{if } x_A(t) < x \leq x_B(t), \\\\
                    0 & \text{if } x_B(t) < x,
                \end{cases}

        Parameters
        ----------
        x : float or np.ndarray
            Spatial positions.
        T : float or np.ndarray
            Time instant.

        Notes
        -----
        Updates the internal :attr:`tilupy.analytic_sol.Depth_result._u`, :attr:`tilupy.analytic_sol.Depth_result._x`, :attr:`tilupy.analytic_sol.Depth_result._t` attributes with the computed result.
        """
        if isinstance(x, float):
            x = [x]
        if isinstance(T, float):
            T = [T]

        self._x = x
        self._t = T

        # x = [i - max(x) for i in x]

        u = []
        for t in T:
            sub_u = []
            for i in x:
                if i <= self.xa(t):
                    sub_u.append(np.nan)
                elif self.xa(t) < i <= self.xb(t):
                    u_val = (2/3) * ( ((i-self._x0)/t) + self._c0 + self._m * t )
                    sub_u.append(u_val)
                else:
                    sub_u.append(np.nan)
            u.append(sub_u)
        self._u = np.array(u)


class Dressler_dry(Depth_result):
    r"""Dam-break solution on a dry domain with friction using shallow water theory.

    This class implements the 1D analytical Dressler's solution of an ideal dam break on a dry domain with friction.
    The dam break is instantaneous, over an horizontal and flat surface with friction.
    It computes the flow height (took verticaly) and velocity over space and time, based on the equation implemanted
    in SWASHES, based on Dressler's equation.
    
    Dressler, R.F., 1952, Hydraulic resistance effect upon the dam‑break functions, Journal of Research of the National Bureau 
    of Standards, vol. 49(3), p. 217-225.

    Parameters
    ----------
        x_0 : float
            Initial dam location (position along x-axis).
        h_0 : float
            Water depth to the left of the dam.
        C : float, optional
            Chézy coefficient, by default 40.
    
    Attributes
    ----------
        _x0 : float 
            Initial dam location (position along x-axis).
        _h0 : float
            Water depth to the left of the dam.
        _c : float, optional
            Chézy coefficient, by default 40.
        _xt : float
            Position of the tip area, by default None.
        _ht : float, optional
            Depth of the tip area, by default None.
        _ut : float, optional
            Velocity of the tip area, by default None.
    """
    def __init__(self,
                 x_0: float, 
                 h_0: float, 
                 C: float=40
                 ):
        super().__init__()
        self._x0 = x_0
        self._h0 = h_0
        self._c = C
        self._xt = []
        

    def xa(self, t: float) -> float:
        r"""
        Position of the rarefaction wave front (left-most edge) :
        
        .. math::
            x_A(t) = x_0 - t \sqrt{g h_0}

        Parameters
        ----------
        t : float
            Time instant.

        Returns
        -------
        float
            Position of the front edge of the rarefaction wave.
        """
        return self._x0 - (t * np.sqrt(self._g*self._h0))


    def xb(self, t: float) -> float:
        r"""
        Position of the contact discontinuity:
        
        .. math::
            x_B(t) = x_0 + 2 t \sqrt{g h_0}

        Parameters
        ----------
        t : float
            Time instant.

        Returns
        -------
        float
            Position of the contact wave (end of rarefaction).
        """
        return self._x0 + (2 * t * np.sqrt(self._g*self._h0))


    def alpha1(self, x: float, t: float) -> float:
        r"""
        Correction coefficient for the height:
        
        .. math::
            \alpha_1(\xi) = \frac{6}{5(2-\xi)} - \frac{2}{3} + \frac{4 \sqrt{3}}{135} (2-\xi)^{3/2}), \\\\

        with :math:`\xi = \frac{x-x_0}{t\sqrt{g h_0}}`

        Parameters
        ----------
        x : float
            Spatial position.
        t : float
            Time instant.

        Returns
        -------
        float
            Correction coefficient.
        """
        xi = (x-self._x0)/(t*np.sqrt(self._g*self._h0))
        # return (6 / (5*(2 - (x/(t*np.sqrt(self._g*self._h0)))))) - (2/3) + (4*np.sqrt(3)/135)*((2 - (x/(t*np.sqrt(self._g*self._h0))))**(3/2))
        # return (6 / (5 * (2 - xi))) - (2 / 3) + (4 * np.sqrt(3) / 135) * (2 - xi) ** (3 / 2)
        if xi < 2:
            return (6 / (5 * (2 - xi))) - (2 / 3) + (4 * np.sqrt(3) / 135) * (2 - xi) ** (3 / 2)
        else:
            return 0
        

    def alpha2(self, x: float, t: float) -> float:
        r"""
        Correction coefficient for the velocity:
        
        .. math::
            \alpha_2(\xi) = \frac{12}{2-(2-\xi)} - \frac{8}{3} + \frac{8 \sqrt{3}}{189} (2-\xi)^{3/2}) - \frac{108}{7(2 - \xi)}, \\\\
        
        with :math:`\xi = \frac{x-x_0}{t\sqrt{g h_0}}`

        Parameters
        ----------
        x : float
            Spatial position.
        t : float
            Time instant.

        Returns
        -------
        float
            Correction coefficient.
        """
        xi = (x-self._x0)/(t*np.sqrt(self._g*self._h0))

        if xi < 2:
            return 12./(2 - xi)- 8/3 + 8*np.sqrt(3)/189 * ((2 - xi)**(3/2)) - 108/(7*(2 - xi)**2)
        else:
            return 0
        
    
    def compute_u(self,
                  x: float | np.ndarray, 
                  T: float | np.ndarray
                  ) -> None:
        """Call :meth:`compute_h`."""
        if self._u is None:
            self.compute_h(x, T)


    def compute_h(self, 
                  x: float | np.ndarray, 
                  T: float | np.ndarray
                  ) -> None:
        r"""Compute the flow height h(x, t) and velocity u(x, t) at given time and positions.
        
        .. math::
                h(x, t) = 
                \begin{cases}
                    h_0 & \text{if } x \leq x_A(t), \\\\
                    \frac{1}{g} \left( \frac{2}{3} \sqrt{g h_0} - \frac{x - x_0}{3t} + \frac{g^{2}}{C^2} \alpha_1 t \right)^2 & \text{if } x_A(t) < x \leq x_t(t), \\\\
                    \frac{-b-\sqrt{b^2 - 4 a (c-x(t))}}{2 a}    & \text{if } x_t(t) < x \leq x_B(t), \\\\
                    0 & \text{if } x_B(t) < x,
                \end{cases}
                
        with :math:`r = \left. \frac{dx}{dh} \right|_{h = h_t}`, :math:`c = x_B(t)`, :math:`a = \frac{r h_t + c - x_t}{h_t^2}`, :math:`b = r - 2 a h_t`. :math:`x_t` and :math:`h_t` being the position
        and the flow depth at the beginning of the tip area.
        
        .. math::
                u(x,t) = 
                \begin{cases}
                    0 & \text{if } x \leq x_A(t), \\\\
                    u_{co} = \frac{2\sqrt{g h_0}}{3} + \frac{2(x - x_0)}{3t} + \frac{g^2}{C^2} \alpha_2 t & \text{if } x_A(t) < x \leq x_t(t), \\\\
                    \max_{x \in [x_A(t), x_t(t)]} u_{co}(x, t) & \text{if } x_t(t) < x \leq x_B(t), \\\\
                    0 & \text{if } x_B(t) < x,
                \end{cases}
        
        Parameters
        ----------
        x : float | np.ndarray
            Spatial positions.
        T : float | np.ndarray
            Time instant.
            
        Notes
        -----        
        Updates the internal :attr:`tilupy.analytic_sol.Depth_result._h`, :attr:`tilupy.analytic_sol.Depth_result._u`, :attr:`tilupy.analytic_sol.Depth_result._x`, :attr:`tilupy.analytic_sol.Depth_result._t` attributes with the computed result.
        """
        if isinstance(x, float):
            x = [x]
        if isinstance(T, float):
            T = [T]

        self._x = x
        self._t = T
        
        h = []
        u = []
        
        xt = None
        ht = None
        ut = None
        
        for t in T:
            sub_h = []
            sub_u = []
            for i in x:
                if i <= self.xa(t):
                    sub_h.append(self._h0)
                    sub_u.append(np.nan)

                elif self.xa(t) < i <= self.xb(t):
                    if t == 0:
                        t = 1e-18
                    term = ((2/3)*np.sqrt(self._g*self._h0)) - ((i - self._x0) / (3*t)) + ((self._g**2)/(self._c**2)) * self.alpha1(i, t) * t
                    h_val = (1/self._g) * term**2
                    
                    if sub_u[-1] is np.nan:
                        sub_u[-1] = 0
                    u_val = (2/3)*np.sqrt(self._g*self._h0)*(1+(i-self._x0)/(np.sqrt(self._g*self._h0)*t)) + ((self._g**2)/(self._c**2))*self.alpha2(i, t)*t

                    if  u_val < sub_u[-1] and xt is None:                            
                        xt = i
                        ht = sub_h[-1]
                        ut = sub_u[-1]
                        
                        dx = x[1] - x[0]
                        dh = sub_h[-1] - sub_h[-2]
                        
                        r = dx / dh
                        c = self.xb(t)
                        a = (r * ht + c - xt) / (ht ** 2)
                        b = r - 2 * a * ht
                        
                    if xt is not None:
                        u_val = ut           
                        h_val = (-b-np.sqrt((b**2) - 4.*a*(c-i))) / (2*a)

                    sub_h.append(h_val)
                    sub_u.append(u_val)
                    
                else:
                    sub_h.append(0)
                    sub_u.append(np.nan)
            h.append(sub_h)
            u.append(sub_u)
            
            self._xt.append(xt)
            xt = None
            ht = None
            ut = None
            
        self._h = np.array(h)
        self._u = np.array(u)


class Chanson_dry(Depth_result):
    r"""Dam-break solution on a dry domain with friction using shallow water theory.

    This class implements the 1D analytical Chanson's solution of an ideal dam break on a dry domain with friction.
    The dam break is instantaneous, over an horizontal and flat surface with friction.
    It computes the flow height (took verticaly) and velocity over space and time, based on the equation implemanted
    in SWASHES, based on Chanson's equation.
    
    Chanson, H., 2005, Applications of the Saint-Venant Equations and Method of Characteristics to the Dam Break Wave Problem. https://espace.library.uq.edu.au/view/UQ:9438
   
    Parameters
    ----------
        x_0 : float 
            Initial dam location (position along x-axis).            
        h_0 : float
            Water depth to the left of the dam.
        f : float
            Darcy friction factor.
    
    Attributes
    ----------
        _x0 : float 
            Initial dam location (position along x-axis).        
        _h0 : float
            Water depth to the left of the dam.
        _f : float, optional
            Darcy friction factor.
    """
    def __init__(self, 
                 h_0: float,
                 x_0: float,
                 f: float
                 ):
        super().__init__()
        self._h0 = h_0
        self._x0 = x_0
        self._f = f
        

    def xa(self, t: float) -> float:
        r"""
        Position of the rarefaction wave front (left-most edge) :
        
        .. math::
            x_A(t) = x_0 - t \sqrt{g h_0}

        Parameters
        ----------
        t : float
            Time instant.

        Returns
        -------
        float
            Position of the front edge of the rarefaction wave.
        """
        return self._x0 - (t * np.sqrt(self._g*self._h0))


    def xb(self, t: float) -> float:
        r"""
        Position of the tip of the flow:
        
        .. math::
            x_B(t) = x_0 + \left( \frac{3}{2} \frac{U(t)}{\sqrt{g h_0}} - 1 \right) t \sqrt{g h_0}

        Parameters
        ----------
        t : float
            Time instant.

        Returns
        -------
        float
            Position of the flow tip.
        """
        cf = self.compute_cf(t)
        return self._x0 + ((3*cf)/(2*np.sqrt(self._g*self._h0))-1) * (t*np.sqrt(self._g*self._h0))
        # return ((3/2) * cf - np.sqrt(self._g * self._h0)) * t


    def xc(self, t: float) -> float:
        r"""
        Position of the contact discontinuity:
        
        .. math::
            x_C(t) = x_0 + \left( \frac{3}{2} \frac{U(t)}{\sqrt{g h_0}} - 1 \right) t \sqrt{\frac{g}{h_0}} + \frac{4}{f\frac{U(t)^2}{g h_0}} \left( 1 - \frac{U(t)}{2 \sqrt{g h_0}} \right)^4

        Parameters
        ----------
        t : float
            Time instant.

        Returns
        -------
        float
            Position of the contact wave (wave front).
        """
        cf = self.compute_cf(t)
        
        term1 = ((1.5 * (cf / np.sqrt(self._g * self._h0))) - 1) * np.sqrt(self._g / self._h0) * t
        term2 = (4 / (self._f * ((cf**2) / (self._g * self._h0)))) * (1 - 0.5 *  (cf / np.sqrt(self._g * self._h0)))**4

        x_s = self._x0 + self._h0 * (term1 + term2)
        return x_s


    def compute_cf(self, t: float) -> float:
        r"""Compute the celerity of the wave front by resolving:
        
        .. math::
            \left( \frac{U}{\sqrt{g h_0}}  \right)^3 - 8 \left( 0.75 - \frac{3 f t \sqrt{g}}{8 \sqrt{h_0}} \right) \left( \frac{U}{\sqrt{g h_0}}  \right)^2 + 12 \left( \frac{U}{\sqrt{g h_0}}  \right) - 8 = 0       

        Parameters
        ----------
        t : float
            Time instant

        Returns
        -------
        float
            Value of the front wave velocity.
        """
        coeffs = [1, (-8*(0.75 - ((3 * self._f * t * np.sqrt(self._g)) / (8 * np.sqrt(self._h0))))), 12, -8]
        roots = np.roots(coeffs)
                
        real_root = roots[-1].real
        return real_root * np.sqrt(self._g * self._h0)


    def compute_h(self, 
                  x: float | np.ndarray, 
                  T: float | np.ndarray
                  ) -> None:
        r"""Compute the flow height h(x, t) at given time and positions.

        .. math::
                h(x, t) = 
                \begin{cases}
                    h_0 & \text{if } x \leq x_A(t), \\\\
                    \frac{4}{9g} \left( \sqrt{g h_0} - \frac{x - x_0}{2t} \right)^2 & \text{if } x_A(t) < x \leq x_B(t), \\\\
                    \sqrt{\frac{f}{4} \frac{U(t)^2}{g h_0} \frac{x_C(t)-x}{h_0}} & \text{if } x_B(t) < x \leq x_C(t), \\\\
                    0 & \text{if } x_C(t) < x
                \end{cases}

        Parameters
        ----------
        x : float or np.ndarray
            Spatial positions.
        T : float or np.ndarray
            Time instant.

        Notes
        -----        
        Updates the internal :attr:`tilupy.analytic_sol.Depth_result._h`, :attr:`tilupy.analytic_sol.Depth_result._x`, :attr:`tilupy.analytic_sol.Depth_result._t` attributes with the computed result.
        """
        if isinstance(x, float):
            x = [x]
        if isinstance(T, float):
            T = [T]

        self._x = x
        self._t = T
        
        h = []
        for t in T:
            sub_h = []
            cf = self.compute_cf(t)
            
            for i in x:
                if i <= self.xa(t):
                    sub_h.append(self._h0)

                elif self.xa(t) < i <= self.xb(t):                    
                    sub_h.append((4/(9*self._g)) * (np.sqrt(self._g*self._h0)-((i-self._x0)/(2*t)))**2)
                    
                elif self.xb(t) <= i <= self.xc(t):
                    # h_left = (4/(9*self._g)) * (np.sqrt(self._g*self._h0) - (self.xb(t)/(2*t)))**2
                    # K = (h_left / self._h0)**2 / ((self.xc(t) - self.xb(t)) / self._h0)
                    # term = K * ((self.xc(t) - i) / self._h0)
                    # val = np.sqrt(term) * self._h0
                    
                    # h_left = (4/(9*self._g)) * (np.sqrt(self._g*self._h0) - (self.xb(t)/(2*t)))**2
                    # term_denominator = (self._f / 4) * ((cf**2) / (self._g * self._h0)) * ((self.xc(t)-self.xb(t)) / self._h0)    
                    # val_term_at_xb = np.sqrt(term_denominator) * self._h0
                    # C = h_left / val_term_at_xb
                    # term = (self._f / 4) * ((cf**2) / (self._g * self._h0)) * ((self.xc(t)-i) / self._h0)
                    # val = C * np.sqrt(term) * self._h0
                    
                    term = (self._f / 4) * ((cf**2) / (self._g * self._h0)) * ((self.xc(t)-(i)) / self._h0)                    
                    val = np.sqrt(term) * self._h0
                    sub_h.append(val)
                    
                else:
                    sub_h.append(0)  
            h.append(sub_h)
        self._h = np.array(h)
            
            
    def compute_u(self, 
                  x: float | np.ndarray, 
                  T: float | np.ndarray
                  ) -> None:
        r"""No solution"""
        self._u = None


class Shape_result(ABC):
    """Abstract base class representing shape results of a simulated flow.

    This class defines a common interface for flow simulation that compute 
    the geometry of the final shape of a flow simulation. 

    Parameters
    ----------
        theta : float, optional
            Angle of the surface, in radian, by default 0.
    
    Attributes
    ----------
        _g = 9.81 : float 
            Gravitational constant.
        _theta : float
            Angle of the surface, in radian.
        _x : float or np.ndarray
            Spatial coordinates.
        _h : np.ndarray
            Flow height depending on space.
    """
    def __init__(self,
                 theta: float=0):
        self._g = 9.81
        self._theta = theta
        
        self._x = None
        self._y = None
        self._h = None
    
    
    @property
    def h(self):
        """Accessor of the shape h of the flow.
        
        Returns
        -------
        numpy.ndarray
            Attribute :attr:`_h`. If None, no solution computed.
        """
        return self._h


    @property
    def x(self):
        """Accessor of the spatial distribution of the computed solution.
        
        Returns
        -------
        numpy.ndarray
            Attribute :attr:`_x`. If None, no solution computed.
        """
        return self._x
    
    
    @property
    def y(self):
        """Accessor of the lateral spatial distribution of the computed solution.
        
        Returns
        -------
        numpy.ndarray
            Attribute :attr:`_y`. If None, no solution computed.
        """
        return self._y


class Coussot_shape(Shape_result):
    r"""Shape solution on an inclined dry domain without friction.

    This class implements the final shape of a simulated flow.
    The flow is over an inclined and flat surface without friction with a finite volume of fluid.
    It computes the spatial coordinates from the flow lenght and height.
    
    Coussot, P., Proust, S., & Ancey, C., 1996, Rheological interpretation of deposits of yield stress fluids, 
    Journal of Non-Newtonian Fluid Mechanics, v. 66(1), p. 55-70, doi:10.1016/0377-0257(96)01474-7.

    Parameters
    ----------
        rho : float
            Fluid density.
        tau : float
            Threshold constraint.
        theta : float, optional
            Angle of the surface, in degree, by default 0.
        h_final : float, optional
            The final flow depth, by default 1.
        H_size : int, optional
            Number of value wanted in the H array, by default 100.
    
    Attributes
    ----------
        _rho : float
            Fluid density.
        _tau : float
            Threshold constraint.
        _D : float or numpy.ndarray
            Normalized distance of the front from the origin.
        _H : float or numpy.ndarray
            Normalized fluid depth.
        _d : float or numpy.ndarray
            Distance of the front from the origin.
        _h : float or numpy.ndarray
            Fluid depth.
        _H_size : int
            Number of point in H-axis.
    
    """   
    def __init__(self, 
                 rho: float,
                 tau: float,
                 theta: float=0,
                 h_final: float=1,
                 H_size: int=100
                 ):
        super().__init__(np.radians(theta))
        self._rho = rho
        self._tau = tau
        
        self._H_size = H_size
        self._D = None
        self._d = None
        if theta>0 and self.h_to_H(h_final) >=1:
            self._H = np.linspace(0, 0.99999999, H_size)
        else:
            self._H = np.linspace(0, self.h_to_H(h_final), H_size)
        self._h = np.array([self.H_to_h(H) for H in self._H])


    def h_to_H(self, 
               h: float
               ) -> float:
        r"""Normalize the fluid depth by following:
        
        .. math::
            H = \frac{\rho g h \sin(\theta)}{\tau_c}
            
        If :math:`\theta = 0`, the expression is:
        
        .. math::
            H = \frac{\rho g h}{\tau_c} 

        Parameters
        ----------
        h : float
            Initial fluid depth.         

        Returns
        -------
        float
            Normalized fluid depth.
        """
        if self._theta == 0:
            return (self._rho*self._g*h)/self._tau
        else:
            return (self._rho*self._g*h*np.sin(self._theta))/self._tau


    def H_to_h(self,
               H: float
               ) -> float:
        r"""Find the original value of the fluid depth from the normalized one
        by following:
        
        .. math::
            h = \frac{H \tau_c}{\rho g \sin(\theta)} 

        If :math:`\theta = 0`, the expression is:
        
        .. math::
            h = \frac{H \tau_c}{\rho g} 
    
        Parameters
        ----------
        H : float
            Normalized value of the fluid depth.

        Returns
        -------
        float
            True value of the fluid depth.
        """
        if self._theta == 0:
            return ((H*self._tau)/(self._rho*self._g))
        else:    
            return ((H*self._tau)/(self._rho*self._g*np.sin(self._theta)))


    def x_to_X(self, 
               x: float
               ) -> float:
        r"""Normalize the spatial coordinates by following:
        
        .. math::
            X = \frac{\rho g x (\sin(\theta))^2}{\tau_c \cos(\theta)}
            
        If :math:`\theta = 0`, the expression is:
        
        .. math::
            X = \frac{\rho g x}{\tau_c} 

        Parameters
        ----------
        x : float
            Initial spatial coordinate.         

        Returns
        -------
        float
            Normalized spatial coordinate.
        """
        if self._theta == 0:
            return (self._rho*self._g*x)/self._tau
        else:
            return (self._rho*self._g*x*np.sin(self._theta)*np.sin(self._theta)) / (self._tau*np.cos(self._theta))
    
    
    def X_to_x(self,
               X: float
               ) -> float:
        r"""Find the original value of the spatial coordinates from the normalized one
        by following:
        
        .. math::
            x = \frac{X \tau_c \cos(\theta)}{\rho g (\sin(\theta))^2} 

        If :math:`\theta = 0`, the expression is:
        
        .. math::
            x = \frac{X \tau_c}{\rho g} 
    
        Parameters
        ----------
        X : float
            Normalized values of the spatial coordinates.

        Returns
        -------
        float
            True value of the spatial coordinate.
        """
        if self._theta == 0:
            return (X*self._tau)/(self._rho*self._g)
        else:
            return (X*self._tau*np.cos(self._theta))/(self._rho*self._g*np.sin(self._theta)*np.sin(self._theta))


    def compute_rheological_test_front_morpho(self) -> None:
        r"""Compute the shape of the frontal lobe from the normalized fluid depth for a rheological test on an inclined 
        surface by following :
        
        .. math::
                D = - H - \ln(1 - H)
        
        If :math:`\theta = 0`, the expression is:
        
        .. math::
                D = \frac{H^2}{2}
        """
        if self._theta == 0:
            D = []
            d = []
            for H_val in self._H:
                D.append((H_val*H_val)/2)
                d.append(self.X_to_x(D[-1]))
                
        else:
            D = []
            d = [] 
            for H_val in self._H:
                D.append(- H_val - np.log(1 - H_val))
                d.append(self.X_to_x(D[-1]))
                
        self._D = np.array(D)
        self._d = np.array(d)


    def compute_rheological_test_lateral_morpho(self) -> None:
        r"""Compute the shape of the lateral lobe from the normalized fluid depth for a rheological test on an inclined 
        surface by following :
        
        .. math::
                D = 1 - \sqrt{1 - H^2}
        """
        D = []
        d = []            
        for H_val in self._H:
            D.append(1 - np.sqrt(1 - (H_val**2)))
            d.append(self.X_to_x(D[-1]))
            
        self._D = np.array(D)
        self._d = np.array(d)


    def compute_slump_test_hf(self, h_init: float) -> float:
        r"""Compute the final fluid depth for a cylindrical slump test following :
        
        .. math::
                \frac{h_f}{h_i} = 1 - \frac{2 \tau_c}{\rho g h_i} \left( 1 - \ln{\frac{2 \tau_c}{\rho g h_i}} \right)
        
        N. Pashias, D. V. Boger, J. Summers, D. J. Glenister; A fifty cent rheometer for yield stress measurement. J. Rheol. 
        1 November 1996; 40 (6): 1179-1189. https://doi.org/10.1122/1.550780
        
        Parameters
        ----------
        h_init : float
            Initial fluid depth.

        Returns
        -------
        float
            Final fluid depth
        """
        H_init = self.h_to_H(h_init)
        val = 1 - ((2/H_init)*(1-np.log(2/H_init)))
        return self.H_to_h(val*H_init)
        
    
    def translate_front(self, d_final: float) -> None:
        """Translate the shape of the frontal (or transversal) lobe to the wanted x (or y) coordinate.

        Parameters
        ----------
        d_final : float
            Final wanted coordinate.
        """
        self._d += d_final
    
    
    def change_orientation_flow(self) -> None:
        """Swap the direction of the result.
        
        Notes
        ------
        There must not have been any prior translation to use this method.
        """
        self._h = self._h[::-1]
        new_d = [-1*v for v in self._d]
        self._d = np.array(new_d[::-1])
        
    
    def interpolate_on_d(self) -> None:
        """Interpolate the profile on d-axis.
        """
        from scipy.interpolate import interp1d
        
        d_min, d_max = self._d.min(), self._d.max()
        d_curve = np.linspace(d_min, d_max, self._H_size)

        f = interp1d(self._d, self._h, kind='cubic')
        h_curve = f(d_curve)
        
        self._d = d_curve
        self._h = h_curve


    @property
    def d(self):
        """Accessor of the spatial distribution of the computed solution.
        
        Returns
        -------
        numpy.ndarray
            Attribute :attr:`_d`. If None, no solution computed.
        """
        return self._d
    

class Front_result:
    """Class computing front position of a simulated flow.

    This class defines multiple methods for flow simulation that compute 
    the position of the front flow at the specified moment. 

    Parameters
    ----------
        h0 : float
            Initial fluid depth.
    
    Attributes
    ----------
        _g = 9.81 : float 
            Gravitational constant.
        _h0 : float
            Initial fluid depth.
        _xf : dictionnary
            Dictionnary of spatial coordinates of the front flow for each time step (keys).
        _labels : dictionnary
            Dictionnary of spatial coordinates computation's method for each time step (keys).
    """
    def __init__(self,
                 h0: float,
                 ):
        self._g = 9.81
        
        self._h0 = h0
        
        self._xf = {}
        self._labels = {}
    

    def xf_mangeney(self, 
                    t: float,
                    delta: float,
                    theta: float=0
                    ) -> float:
        r"""
        Mangeney's equation for a dam-break solution over an infinite inclined dry domain with friction
        and an infinitely-long fluid mass:
        
        .. math::
            x_f(t) = \frac{1}{2}mt^2 + 2 c_0 t
            
        with :math:`c_0` the initial wave propagation speed defined by:
        
        .. math::
            c_0 = \sqrt{g h_0 \cos{\theta}}

        and :math:`m` the constant horizontal acceleration of the front defined by:
    
        .. math::
            m = g \sin{\theta} - g \cos{\theta} \tan{\delta}
    
        Mangeney, A., Heinrich, P., & Roche, R., 2000, Analytical solution for testing debris avalanche numerical models, 
        Pure and Applied Geophysics, vol. 157, p. 1081-1096.

        Parameters
        ----------
        t : float
            Time instant.     
        delta : float
            Dynamic friction angle, in degree.        
        theta : float
            Slope angle, in degree.        

        Returns
        -------
        float
            Position of the front edge of the fluid.
        """
        theta_rad = np.radians(theta)
        delta_rad = np.radians(delta)

        m = self._g * np.sin(theta_rad) - (self._g * np.cos(theta_rad) * np.tan(delta_rad))
        c0 = np.sqrt(self._g * self._h0 * np.cos(theta_rad))
        xf = 0.5*m*(t**2) + (2*c0*t)
        
        if t in self._labels:
            if f"Mangeney d{delta}" not in self._labels[t] :
                 self._labels[t].append(f"Mangeney d{delta}")
                 self._xf[t].append(xf)
        else:
            self._labels[t] = [f"Mangeney d{delta}"]
            self._xf[t] = [xf]
            
        return xf


    def xf_dressler(self, 
                    t: float,
                    ) -> float:
        r"""
        Dressler's equation for a dam-break solution over an infinite inclined dry domain with friction:
        
        .. math::
            x_f(t) = 2 t \sqrt{g h_0}
    
        Parameters
        ----------
        t : float
            Time instant.

        Returns
        -------
        float
            Position of the front edge of the fluid.
        """
        xf = 2 * t * np.sqrt(self._g*self._h0)
        
        if t in self._labels:
            if "Dressler" not in self._labels[t] :
                 self._labels[t].append("Dressler")
                 self._xf[t].append(xf)
        else:
            self._labels[t] = ["Dressler"]
            self._xf[t] = [xf]
            
        return xf


    def xf_ritter(self,
                  t: float
                  ) -> float:
        r"""
        Ritter's equation for a dam-break solution over an infinite inclined dry domain without friction:
        
        .. math::
            x_f(t) = 2 t \sqrt{g h_0}
        
        Ritter A. Die Fortpflanzung der Wasserwellen. Zeitschrift des Vereines Deuscher Ingenieure 
        August 1892; 36(33): 947-954.
    
        Parameters
        ----------
        t : float
            Time instant.

        Returns
        -------
        float
            Position of the front edge of the fluid.
        """
        xf = 2 * t * np.sqrt(self._g*self._h0)
        
        if t in self._labels:
            if "Ritter" not in self._labels[t] :
                 self._labels[t].append("Ritter")
                 self._xf[t].append(xf)
        else:
            self._labels[t] = ["Ritter"]
            self._xf[t] = [xf]
            
        return xf


    def xf_stoker(self, 
                    t: float,
                    hr: float
                    ) -> float:
        r"""
        Stoker's equation for a dam-break solution over an infinite inclined wet domain without friction:
        
        .. math::
            x_f(t) =t c_m
            
        with :math:`c_m` the front wave velocity solution of:

        .. math::
            c_m h_r - h_r \left( \sqrt{1 + \frac{8 c_m^2}{g h_r}} - 1 \right) \left( \frac{c_m}{2} - \sqrt{g h_0} + \sqrt{\frac{g h_r}{2} \left( \sqrt{1 + \frac{8 c_m^2}{g h_r}} - 1 \right)} \right) = 0
           
        Stoker JJ. Water Waves: The Mathematical Theory with Applications, Pure and Applied Mathematics, 
        Vol. 4. Interscience Publishers: New York, USA, 1957.
        
        Sarkhosh, P. (2021). Stoker solution package (1.0.0). Zenodo. https://doi.org/10.5281/zenodo.5598374
    
        Parameters
        ----------
        t : float
            Time instant.
        hr : float
            Fluid depth at the right of the dam.

        Returns
        -------
        float
            Position of the front edge of the fluid.
        """
        f_cm = 1
        df_cm = 1
        cm = 10 * self._h0
        
        while abs(f_cm / cm) > 1e-10:
            root_term = np.sqrt(8 * cm**2 / np.sqrt(self._g * hr)**2 + 1)
            inner_sqrt = np.sqrt(self._g * hr * (root_term - 1) / 2)
            
            f_cm = cm * hr - hr * (root_term - 1) * (cm / 2 - np.sqrt(self._g * self._h0) + inner_sqrt)
            df_cm = (hr 
                     - hr * ((2 * cm * self._g * hr) / (np.sqrt(self._g * hr)**2 * root_term * inner_sqrt) + 0.5) * (root_term - 1)
                     - (8 * cm * hr * (cm / 2 - np.sqrt(self._g * self._h0) + inner_sqrt)) / (np.sqrt(self._g * hr)**2 * root_term))
            
            cm -= f_cm / df_cm

        xf = cm * t
            
        if t in self._labels:
            if "Stoker" not in self._labels[t] :
                self._labels[t].append("Stoker")
                self._xf[t].append(xf)
        else:
            self._labels[t] = ["Stoker"]
            self._xf[t] = [xf]
            
        return xf
    
    
    def xf_chanson(self, 
                   t: float,
                   f: float
                   ) -> float:
        r"""
        Chanson's equation for a dam-break solution over an infinite inclined dry domain with friction:
        
        .. math::
            x_f(t) = \left( \frac{3}{2} \frac{U(t)}{\sqrt{g h_0}} - 1 \right) t \sqrt{\frac{g}{h_0}} + \frac{4}{f\frac{U(t)^2}{g h_0}} \left( 1 - \frac{U(t)}{2 \sqrt{g h_0}} \right)^4

        with :math:`U(t)` the front wave velocity solution of:

        .. math::
            \left( \frac{U}{\sqrt{g h_0}}  \right)^3 - 8 \left( 0.75 - \frac{3 f t \sqrt{g}}{8 \sqrt{h_0}} \right) \left( \frac{U}{\sqrt{g h_0}}  \right)^2 + 12 \left( \frac{U}{\sqrt{g h_0}}  \right) - 8 = 0       

        Chanson, Hubert. (2005). Analytical Solution of Dam Break Wave with Flow Resistance: Application to Tsunami Surges. 137. 
        
        Parameters
        ----------
        t : float
            Time instant.
        f : float
            Darcy friction coefficient.

        Returns
        -------
        float
            Position of the front edge of the fluid.
        """
        coeffs = [1, (-8*(0.75 - ((3 * f * t * np.sqrt(self._g)) / (8 * np.sqrt(self._h0))))), 12, -8]
        roots = np.roots(coeffs)
                
        real_root = roots[-1].real
        cf = real_root * np.sqrt(self._g * self._h0)
        
        term1 = ((1.5 * (cf / np.sqrt(self._g * self._h0))) - 1) * np.sqrt(self._g / self._h0) * t
        term2 = (4 / (f * ((cf**2) / (self._g * self._h0)))) * (1 - 0.5 *  (cf / np.sqrt(self._g * self._h0)))**4

        xf = self._h0 * (term1 + term2)
        
        if t in self._labels:
            if "Chanson" not in self._labels[t] :
                self._labels[t].append("Chanson")
                self._xf[t].append(xf)
        else:
            self._labels[t] = ["Chanson"]
            self._xf[t] = [xf]
            
        return xf


    def compute_cf(self, t: float) -> float:
        r"""Compute the celerity of the wave front by resolving:
        

        Parameters
        ----------
        t : float
            Time instant

        Returns
        -------
        float
            Value of the front wave velocity.
        """
  

    def show_fronts_over_methods(self, x_unit: str="m") -> None:
        """Plot the front distance from the initial position for each method.

        Parameters
        ----------
        x_unit : str, optional
            X-axis unit, by default "m"
        """
        fig, ax = plt.subplots(figsize=(10, 5))

        label_order = []
        for t in sorted(self._labels.keys()):
            for label in self._labels[t]:
                if label not in label_order:
                    label_order.append(label)

        y_levels = {label: i for i, label in enumerate(reversed(label_order))}
        yticks = list(y_levels.values())
        yticklabels = list(reversed(label_order))

        sorted_times = sorted(self._xf.keys())
        colors = cm.copper(np.linspace(0, 1, len(sorted_times)))

        for color, t in zip(colors, sorted_times):
            x_list = self._xf[t]
            label_list = self._labels[t]

            for x, label in zip(x_list, label_list):
                y = y_levels[label]
                ax.vlines(x, y - 0.3, y + 0.3, color=color, linewidth=2)
                ax.text(x + 1, y, f"{x:.2f}", rotation=90, va='center', fontsize=8, color=color)

        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.invert_yaxis()

        ax.set_xlim(left=0)
        ax.set_xlim(right=max(x for sublist in self._xf.values() for x in sublist) + 5)

        ax.set_xlabel(f"x [{x_unit}]")
        ax.set_title("Flow front positions over time")

        ax.grid(True, axis='x')

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=color, lw=2, label=f"t = {t}s")
            for color, t in zip(colors, sorted_times)
        ]
        ax.legend(handles=legend_elements, title="Time steps", loc="best")

        plt.tight_layout()
        plt.show()


    def show_fronts_over_time(self, x_unit: str="m") -> None:
        """Plot the front distance from the initial position over time for each method.

        Parameters
        ----------
        x_unit : str, optional
            X-axis unit, by default "m"
        """
        T = sorted(self._labels.keys())
        
        dico_xf = {}
        dico_time = {}
        for t in T:
            for i in range(len(self._labels[t])):
                if self._labels[t][i] not in dico_xf:
                    dico_xf[self._labels[t][i]] = [self._xf[t][i]]
                    dico_time[self._labels[t][i]] = [t]
                else:
                    dico_xf[self._labels[t][i]].append(self._xf[t][i])
                    dico_time[self._labels[t][i]].append(t)
        
        for label in dico_xf.keys():
            plt.scatter(dico_xf[label], dico_time[label], marker='x', label=label)
        
        plt.xlabel(f'Distance to the dam break [{x_unit}]')
        plt.ylabel('Time [s]')
        
        
        plt.grid(which="major")
        plt.legend(loc='best')
        plt.show()
