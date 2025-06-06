# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 12:09:41 2023

@author: peruzzetto
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


from scipy.optimize import fsolve

from abc import ABC, abstractmethod


class Depth_result(ABC):
    """Abstract base class representing simulation results for flow depth and velocity.

    This class defines a common interface for analytical solution that compute flow height 
    h(x,t) and flow velocity u(x,t).

    Attributes:
    -----------
        _g : float 
            Gravitational constant.
        _theta : float
            Angle of the surface, in radian.
        _x : int or np.ndarray
            Spatial coordinates.
        _t : int or np.ndarray
            Time instant.
        _h : np.ndarray
            Flow height depending on space at a moment.
        _u : np.ndarray
            Flow velocity depending on space at a moment.
    
    Parameters:
    -----------
        theta : float, optional
            Angle of the surface, in radian, by default 0.
    """
    def __init__(self, 
                 theta: float=None):
        self._g = 9.81
        self._theta = theta
        self._x = None
        self._t = None
        self._h = None
        self._u = None


    @abstractmethod
    def compute_h(self, 
          x: int | np.ndarray, 
          t: int | np.ndarray
          ) -> None:
        """Virtual function that compute the flow height 'h' at given space and time.

        Parameters
        ----------
        x : int or np.ndarray
            Spatial coordinates.
        t : int or np.ndarray
            Time instant.
        """
        pass


    @abstractmethod
    def compute_u(self, 
          x: int | np.ndarray, 
          t: int | np.ndarray
          ) -> None:
        """Virtual function that compute the flow velocity 'u' at given space and time.

        Parameters
        ----------
        x : int or np.ndarray
            Spatial coordinates.
        t : int or np.ndarray
            Time instant.
        """
        pass


    @property
    def h(self):
        """Accessor of h(x,t) solution.
        
        Returns
        -------
        self._h : np.ndarray
            Flow height in self._x at self._t. If None, no solution computed.
        """
        return self._h


    @property
    def u(self):
        """Accessor of u(x,t) solution.
        
        Returns
        -------
        self._u : np.ndarray
            Flow velocity in self._x at self._t. If None, no solution computed.
        """
        return self._u


    @property
    def x(self):
        """Accessor of the spatial distribution of the computed solution.
        
        Returns
        -------
        self._x : np.ndarray
            Spatial distribution of the computed solution. If None, no solution computed.
        """
        return self._x


    @property
    def t(self):
        """Accessor of the time instant of the computed solution.
        
        Returns
        -------
        self._t : int or float
            Time instant of the computed solution. If None, no solution computed.
        """
        return self._t


    def show_res(self, 
                 show_h=False, 
                 show_u=False,
                 show_slop=False,
                 x_unit:str = "m",
                 h_unit:str = "m",
                 u_unit:str = "m/s"
                ):
        """Plot the simulation results.

        Parameters
        ----------
        show_h : bool, optional
            If True, plot the flow height ('h') curve.
        show_u : bool, optional
            If True, plot the flow velocity ('u') curve.
        show_slop : bool, optional
            If True, plot the slop of the surface.
        x_unit: str
            Space unit.
        h_unit: str
            Height unit.
        u_unit: str
            Velocity unit.
        """
        inclined_surf = None
        z_surf = [0, 0]
        
        if self._theta is not None and show_slop:
            z_surf = [z_surf[0], -self._x[-1]*np.tan(self._theta)]
            inclined_surf = np.linspace(z_surf[0], z_surf[1], len(self._x))
        
        if show_h and self._h is not None:
            if self._h.ndim == 1:
                if self._theta is not None and show_slop:
                    h_inclined = [(self._h[i]/np.cos(self._theta)) + inclined_surf[i] for i in range(len(self._h))]
                    plt.plot(self._x, h_inclined, color='black', linewidth=1)
                else:
                    plt.plot(self._x, self._h, color='black', linewidth=1)
            else:
                for h in range(len(self._h)):
                    l_style = "-"
                    if self._t[h] == 0 :
                        l_style = ":"
                    if self._theta is not None and show_slop:
                        h_inclined = [(self._h[h][i]/np.cos(self._theta)) + inclined_surf[i] for i in range(len(self._h[h]))]
                        plt.plot(self._x, h_inclined, color='black', linewidth=1, linestyle=l_style)
                    else:
                        plt.plot(self._x, self._h[h], color='black', linewidth=1, linestyle=l_style)
            plt.plot([self._x[0], self._x[-1]], z_surf, color='black', linewidth=2)
            
            plt.title(f"Flow height for t={self._t}")
            plt.xlabel(f"x [{x_unit}]")
            plt.ylabel(f"h [{h_unit}]")
            plt.show()

        if show_u and self._u is not None:
            if self._u.ndim == 1:
                plt.plot(self._x, self._u, color='black', linewidth=1)
                
            else:
                for u in range(len(self._u)):
                    l_style = "-"
                    if self._t[u] == 0 :
                        l_style = ":"
                    plt.plot(self._x, self._u[u], color='black', linewidth=1, linestyle=l_style)

            plt.plot([self._x[0], self._x[-1]], z_surf, color='black', linewidth=2)
            plt.title(f"Flow velocity for t={self._t}")
            plt.xlabel(f"x [{x_unit}]")
            plt.ylabel(f"u [{u_unit}]")
            plt.show()


class Stocker_wet(Depth_result):
    r"""Dam-break solution on a wet domain using shallow water theory.

    This class implements the 1D analytical Stocker's solution of an ideal dam break on a wet domain.
    The dam break is instantaneous, over an horizontal and flat surface with no friction.
    It computes the flow height (took verticaly) and velocity over space and time, based on the equation implemanted
    in SWASHES, based on Stoker's equation.
    
    Stoker JJ. Water Waves: The Mathematical Theory with Applications, Pure and Applied Mathematics, 
    Vol. 4. Interscience Publishers: New York, USA, 1957.

    Attributes:
    -----------
        _x0 : int 
            Initial dam location (position along x-axis).
        _hl : float
            Water depth to the left of the dam.
        _hr : float
            Water depth to the right of the dam.
        _cm : float
            Critical velocity.
       
    Parameters:
    -----------
        x_0 : int
            Initial dam location (position along x-axis).
        h_l : float
            Water depth to the left of the dam.
        h_r : float
            Water depth to the right of the dam.
        h_m : float, optional
            Intermediate height used to compute the critical speed cm. If not provided,
            it will be computed numerically via the 'compute_cm()' method.
    """
    def __init__(self, 
                 x_0: int, 
                 h_l: int, 
                 h_r: int, 
                 h_m: int=None
                 ):
        super().__init__()
        self._x0 = x_0
        self._hl = h_l
        self._hr = h_r
        self._cm = None
        self.compute_cm()

        if h_m is not None:
            self._cm = np.sqrt(self._g * h_m)


    def xa(self, t: int) -> float:
        r"""
        Position of the rarefaction wave front (left-most edge) :
        
        .. math::
            x_A(t) = x_0 - t \sqrt{g h_l}

        Parameters
        ----------
        t : int
            Time instant.

        Returns
        -------
        float
            Position of the front edge of the rarefaction wave.
        """
        return self._x0 - (t * np.sqrt(self._g*self._hl))


    def xb(self, t: int) -> float:
        r"""
        Position of the contact discontinuity:
        
        .. math::
            x_B(t) = x_0 + t \left( 2 \sqrt{g h_l} - 3 c_m \right)

        Parameters
        ----------
        t : int
            Time instant.

        Returns
        -------
        float
            Position of the contact wave (end of rarefaction).
        """
        return self._x0 + (t * ((2 * np.sqrt(self._g*self._hl)) - (3*self._cm)))


    def xc(self, t: int) -> float:
        r"""
        Position of the shock wave front (right-most wave):
        
        .. math::
            x_C(t) = x_0 + t \cdot \frac{2 c_m^2 \left( \sqrt{g h_l} - c_m \right)}{c_m^2 - g h_r}

        Parameters
        ----------
        t : int
            Time instant.

        Returns
        -------
        float
            Position of the shock front.
        """
        return self._x0 + (t * (((2*self._cm**2)*(np.sqrt(self._g*self._hl)-self._cm)) / ((self._cm**2) - (self._g*self._hr))))


    def equation_cm(self, cm) -> float:
        r"""Equation of the critical velocity cm:
        
        .. math::
            -8.g.hr.cm^{2}.(g.hl - cm^{2})^{2} + (cm^{2} - g.hr)^{2} . (cm^{2} + g.hr) = 0

        Parameters
        ----------
        cm : float
            Trial value for cm.

        Returns
        -------
        float
            Residual of the equation. Zero when cm satisfies the system.
        """
        return -8 * self._g * self._hr * cm**2 * (self._g * self._hl - cm**2)**2 + (cm**2 - self._g * self._hr)**2 * (cm**2 + self._g * self._hr)


    def compute_cm(self) -> None:
        r"""Solves the non-linear equation to compute the critical velocity 'cm'.

        Uses numerical root-finding to find a valid value of cm that separates
        the flow regimes. Sets self._cm if a valid solution is found.
        """
        guesses = np.linspace(0.01, 1000, 1000)
        solutions = []

        for guess in guesses:
            sol = fsolve(self.equation_cm, guess)[0]

            if abs(self.equation_cm(sol)) < 1e-6 and not any(np.isclose(sol, s, atol=1e-6) for s in solutions):
                solutions.append(sol)

        for sol in solutions:
            hm = sol**2 / self._g
            if hm < self._hl and hm > self._hr:
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
                  x: int | np.ndarray, 
                  T: int | np.ndarray):
        r"""Compute the flow height h(x, t) at given time and positions.

        .. math::
                h(x, t) = 
                \begin{cases}
                    h_l & \text{if } x \leq x_A(t), \\\\
                    \frac{4}{9g} \left( \sqrt{g h_l} - \frac{x - x_0}{2t} \right)^2 & \text{if } x_A(t) < x \leq x_B(t), \\\\
                    \frac{c_m^2}{g} & \text{if } x_B(t) < x \leq x_C(t), \\\\
                    h_r & \text{if } x_C(t) < x,
                \end{cases}

        Parameters
        ----------
        x : int or np.ndarray
            Spatial positions.
        T : int or np.ndarray
            Time instant.

        Notes
        -----
        Updates the internal '_h', '_x', '_t' attributes with the computed result.
        """
        if self._cm is not None:
            if isinstance(x, int):
                x = [x]
            self._x = x
            self._t = T

            if isinstance(T, int):
                h = []
                for i in x:
                    if i <= self.xa(T):
                        h.append(self._hl)
                    # elif i > self.xa(t) and i <= self.xb(t):
                    elif self.xa(T) < i <= self.xb(T):
                        h.append((4/(9*self._g))*(np.sqrt(self._g *
                                self._hl)-((i-self._x0)/(2*T)))**2)   # i-x0 and not i to recenter the breach of the dam at x=0.
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
                            sub_h.append(self._hl)
                        elif self.xa(t) < i <= self.xb(t):
                            sub_h.append((4/(9*self._g))*(np.sqrt(self._g *
                                    self._hl)-((i-self._x0)/(2*t)))**2)
                        elif self.xb(t) < i <= self.xc(t):
                            sub_h.append((self._cm**2)/self._g)
                        else:
                            sub_h.append(self._hr)
                    h.append(sub_h)
                self._h = np.array(h)

        else:
            print("No critical velocity found")


    def compute_u(self, 
                  x: int | np.ndarray, 
                  T: int | np.ndarray):
        r"""Compute the flow velocity u(x, t) at given time and positions.

        .. math::
                u(x,t) = 
                \begin{cases}
                    0 & \text{if } x \leq x_A(t), \\\\
                    \frac{2}{3} \left( \frac{x - x_0}{t} + \sqrt{g h_l} \right) & \text{if } x_A(t) < x \leq x_B(t), \\\\
                    2 \left( \sqrt{g h_l} - c_m \right) & \text{if } x_B(t) < x \leq x_C(t), \\\\
                    0 & \text{if } x_C(t) < x,
                \end{cases}

        Parameters
        ----------
        x : int or np.ndarray
            Spatial positions.
        T : int or np.ndarray
            Time instant.

        Notes
        -----
        Updates the internal `_u`, `_x`, `_t` attributes with the computed result.
        """
        if self._cm is not None:
            if isinstance(x, int):
                x = [x]
            self._x = x
            self._t = T

            if isinstance(T, int):
                u = []
                for i in x:
                    if i <= self.xa(T):
                        u.append(0)
                    elif i > self.xa(T) and i <= self.xb(T):
                        u.append((2/3)*(((i-self._x0)/T) +
                                np.sqrt(self._g*self._hl)))
                    elif i > self.xb(T) and i <= self.xc(T):
                        u.append(2*(np.sqrt(self._g*self._hl) - self._cm))
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
                                    np.sqrt(self._g*self._hl)))
                        elif i > self.xb(t) and i <= self.xc(t):
                            sub_u.append(2*(np.sqrt(self._g*self._hl) - self._cm))
                        else:
                            sub_u.append(0)
                    u.append(sub_u)
                self._u = np.array(u)

        else:
            print("First define cm")


class Ritter_dry(Depth_result):
    r"""Dam-break solution on a dry domain using shallow water theory.

    This class implements the 1D analytical Ritter's solution of an ideal dam break on a dry domain.
    The dam break is instantaneous, over an horizontal and flat surface with no friction.
    It computes the flow height (took verticaly) and velocity over space and time, based on the equation implemanted
    in SWASHES, based on Ritter's equation.
    
    Ritter A. Die Fortpflanzung der Wasserwellen. Zeitschrift des Vereines Deuscher Ingenieure 
    August 1892; 36(33): 947-954.

    Attributes:
    -----------
        _x0 : int 
            Initial dam location (position along x-axis).
        _hl : int
            Water depth to the left of the dam.
        _hr : int
            Water depth to the right of the dam, by default 0.
        
    Parameters:
    -----------
        x_0 : int
            Initial dam location (position along x-axis).
        h_l : int
            Water depth to the left of the dam.
        h_r : int, optinal
            Water depth to the right of the dam, by default 0.
    """
    def __init__(self, 
                 x_0: int, 
                 h_l: int, 
                 h_r: int=0, 
                 ):
        super().__init__()
        self._x0 = x_0
        self._hl = h_l
        self._hr = h_r


    def xa(self, t: int) -> float:
        r"""
        Position of the rarefaction wave front (left-most edge) :
        
        .. math::
            x_A(t) = x_0 - t \sqrt{g h_l}

        Parameters
        ----------
        t : int
            Time instant.

        Returns
        -------
        float
            Position of the front edge of the rarefaction wave.
        """
        return self._x0 - (t * np.sqrt(self._g*self._hl))


    def xb(self, t: int) -> float:
        r"""
        Position of the contact discontinuity:
        
        .. math::
            x_B(t) = x_0 + 2 t \sqrt{g h_l}

        Parameters
        ----------
        t : int
            Time instant.

        Returns
        -------
        float
            Position of the contact wave (end of rarefaction).
        """
        return self._x0 + (2 * t * np.sqrt(self._g*self._hl))


    def compute_h(self, 
                  x: int | np.ndarray, 
                  T: int | np.ndarray):
        r"""Compute the flow height h(x, t) at given time and positions.

        .. math::
                h(x, t) = 
                \begin{cases}
                    h_l & \text{if } x \leq x_A(t), \\\\
                    \frac{4}{9g} \left( \sqrt{g h_l} - \frac{x - x_0}{2t} \right)^2 & \text{if } x_A(t) < x \leq x_B(t), \\\\
                    0 & \text{if } x_B(t) < x,
                \end{cases}

        Parameters
        ----------
        x : int or np.ndarray
            Spatial positions.
        T : int or nd.ndarray
            Time instant.

        Notes
        -----
        Updates the internal '_h', '_x', '_t' attributes with the computed result.
        """
        if isinstance(x, int):
            x = [x]
        self._x = x
        self._t = T

        if isinstance(T, int):
            h = []
            for i in x:
                if i <= self.xa(T):
                    h.append(self._hl)
                elif self.xa(T) < i <= self.xb(T):
                    h.append((4/(9*self._g)) *
                            (np.sqrt(self._g*self._hl)-((i-self._x0)/(2*T)))**2)
                else:
                    h.append(self._hr)
            self._h = np.array(h)
        
        else:  
            h = []
            for t in T:
                sub_h = []
                for i in x:
                    if i <= self.xa(t):
                        sub_h.append(self._hl)
                    elif self.xa(t) < i <= self.xb(t):
                        sub_h.append((4/(9*self._g)) *
                                (np.sqrt(self._g*self._hl)-((i-self._x0)/(2*t)))**2)
                    else:
                        sub_h.append(self._hr)
                h.append(sub_h)
            self._h = np.array(h)


    def compute_u(self, 
                  x: int | np.ndarray, 
                  T: int | np.ndarray):
        r"""Compute the flow velocity u(x, t) at given time and positions.

        .. math::
                u(x,t) = 
                \begin{cases}
                    0 & \text{if } x \leq x_A(t), \\\\
                    \frac{2}{3} \left( \frac{x - x_0}{t} + \sqrt{g h_l} \right) & \text{if } x_A(t) < x \leq x_B(t), \\\\
                    0 & \text{if } x_B(t) < x,
                \end{cases}

        Parameters
        ----------
        x : int or np.ndarray
            Spatial positions.
        T : int or np.ndarray
            Time instant.

        Notes
        -----
        Updates the internal `_u`, `_x`, `_t` attributes with the computed result.
        """
        if isinstance(x, int):
            x = [x]
        self._x = x
        self._t = T

        if isinstance(T, int):
            u = []
            for i in x:
                if i <= self.xa(T):
                    u.append(0)
                elif i > self.xa(T) and i <= self.xb(T):
                    u.append((2/3)*(((i-self._x0)/T) + np.sqrt(self._g*self._hl)))
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
                        sub_u.append((2/3)*(((i-self._x0)/t) + np.sqrt(self._g*self._hl)))
                    else:
                        sub_u.append(0)
                u.append(sub_u)
            self._u = np.array(u)


class Dressler_dry(Depth_result):
    r"""Dam-break solution on a dry domain with friction using shallow water theory.

    This class implements the 1D analytical Dressler's solution of an ideal dam break on a dry domain with friction.
    The dam break is instantaneous, over an horizontal and flat surface with friction.
    It computes the flow height (took verticaly) and velocity over space and time, based on the equation implemanted
    in SWASHES, based on Dressler's equation.
    
    Dressler RF. Hydraulic resistance effect upon the dam-break functions. Journal of Research of the National Bureau 
    of Standards September 1952; 49(3): 217-225.

    Attributes:
    -----------
        _x0 : int 
            Initial dam location (position along x-axis).
        _hl : int
            Water depth to the left of the dam.
        _hr : int
            Water depth to the right of the dam, by default 0.
        _c : int, optional
            Chézy coefficient, by default 40.
       
    Parameters:
    -----------
        x_0 : int
            Initial dam location (position along x-axis).
        h_l : int
            Water depth to the left of the dam.
        h_r : int, optinal
            Water depth to the right of the dam, by default 0.
        C : int, optional
            Chézy coefficient, by default 40.
    """
    def __init__(self, 
                 x_0: int, 
                 h_l: int, 
                 h_r: int=0,
                 C: int=40):
        super().__init__()
        self._x0 = x_0
        self._hl = h_l
        self._hr = h_r
        self._c = C


    def xa(self, t: int) -> float:
        r"""
        Position of the rarefaction wave front (left-most edge) :
        
        .. math::
            x_A(t) = x_0 - t \sqrt{g h_l}

        Parameters
        ----------
        t : int
            Time instant.

        Returns
        -------
        float
            Position of the front edge of the rarefaction wave.
        """
        return self._x0 - (t * np.sqrt(self._g*self._hl))


    def xb(self, t: int) -> float:
        r"""
        Position of the contact discontinuity:
        
        .. math::
            x_B(t) = x_0 + 2 t \sqrt{g h_l}

        Parameters
        ----------
        t : int
            Time instant.

        Returns
        -------
        float
            Position of the contact wave (end of rarefaction).
        """
        return self._x0 + (2 * t * np.sqrt(self._g*self._hl))


    def alpha1(self, x, t):
        r"""
        Correction coefficient for the height:
        
        .. math::
            \alpha_1(\xi) = \frac{6}{5(2-\xi)} - \frac{2}{3} + \frac{4 \sqrt{3}}{135} (2-\xi)^{3/2}), \\\\
            \xi = \frac{x-x_0}{t\sqrt{g h_l}}

        Parameters
        ----------
        t : int
            Time instant.

        Returns
        -------
        float
            Correction coefficient.
        """
        xi = (x-self._x0)/(t*np.sqrt(self._g*self._hl))
        # return (6 / (5*(2 - (x/(t*np.sqrt(self._g*self._hl)))))) - (2/3) + (4*np.sqrt(3)/135)*((2 - (x/(t*np.sqrt(self._g*self._hl))))**(3/2))
        return (6 / (5 * (2 - xi))) - (2 / 3) + (4 * np.sqrt(3) / 135) * (2 - xi) ** (3 / 2)


    def alpha2(self, x, t):
        r"""
        Correction coefficient for the velocity:
        
        .. math::
            \alpha_2(\xi) = \frac{12}{2-(2-\xi)} - \frac{8}{3} + \frac{8 \sqrt{3}}{189} (2-\xi)^{3/2}) - \frac{108}{7(2 - \xi)}, \\\\
            \xi = \frac{x-x_0}{t\sqrt{g h_l}}

        Parameters
        ----------
        t : int
            Time instant.

        Returns
        -------
        float
            Correction coefficient.
        """
        return (12 / (2 - (x/(t*np.sqrt(self._g*self._hl))))) - (8/3) + (8*np.sqrt(3)/189)*((2 - (x/(t*np.sqrt(self._g*self._hl))))**(3/2)) - (108/(7*(2 - (x/(t*np.sqrt(self._g*self._hl))))))


    def compute_h(self, x, T):
        r"""Compute the flow height h(x, t) at given time and positions.

        .. math::
                h(x, t) = 
                \begin{cases}
                    h_l & \text{if } x \leq x_A(t), \\\\
                    \frac{1}{g} \left( \frac{2}{3} \sqrt{g h_l} - \frac{x - x_0}{3t} + \frac{g^{2}}{C^2} \alpha_1 t \right)^2 & \text{if } x_A(t) < x \leq x_B(t), \\\\
                    0 & \text{if } x_B(t) < x,
                \end{cases}

        Parameters
        ----------
        x : int or np.ndarray
            Spatial positions.
        T : int or np.ndarray
            Time instant.

        Notes
        -----
        Updates the internal '_h', '_x', '_t' attributes with the computed result.
        """
        if isinstance(x, int):
            x = [x]
        self._x = x
        self._t = T
        
        if isinstance(T, int):
            h = []
            for i in x:
                if i <= self.xa(T):
                    h.append(self._hl)
                elif self.xa(T) < i <= self.xb(T):
                    # h.append( (1/self._g) * ( ((2/3)*np.sqrt(self._g*self._hl)) - ((i-self._x0)/(3*t)) + ((self._g**2)/(self._c**2))*self.alpha1(i-self._x0, t)*t ) )
                    term = ((2/3)*np.sqrt(self._g*self._hl)) - ((i - self._x0) /
                                                                (3*T)) + ((self._g**2)/(self._c**2)) * self.alpha1(i, T) * T
                    h_val = (1/self._g) * term**2
                    if h_val > h[-1]:
                        term = ((2/3)*np.sqrt(self._g*self._hl)) - ((i - self._x0) /
                                                                    (3*T)) + ((self._g**2)/(self._c**2)) * self.alpha1(i, T) * T
                        h_val = (1/self._g) * term**2
                    h.append(h_val)
                else:
                    h.append(self._hr)
            self._h = np.array(h)
        
        else:
            h = []
            for t in T:
                sub_h = []
                for i in x:
                    if i <= self.xa(t):
                        sub_h.append(self._hl)
                    elif self.xa(t) < i <= self.xb(t):
                        # h.append( (1/self._g) * ( ((2/3)*np.sqrt(self._g*self._hl)) - ((i-self._x0)/(3*t)) + ((self._g**2)/(self._c**2))*self.alpha1(i-self._x0, t)*t ) )
                        term = ((2/3)*np.sqrt(self._g*self._hl)) - ((i - self._x0) /
                                                                    (3*t)) + ((self._g**2)/(self._c**2)) * self.alpha1(i, t) * t
                        h_val = (1/self._g) * term**2
                        # if h_val > sub_h[-1]:
                        #     term = ((2/3)*np.sqrt(self._g*self._hl)) - ((i - self._x0) /
                        #                                                 (3*t)) + ((self._g**2)/(self._c**2)) * self.alpha1(i, t) * t
                        #     h_val = (1/self._g) * term**2
                        sub_h.append(h_val)
                    else:
                        sub_h.append(self._hr)
                h.append(sub_h)
            self._h = np.array(h)
            
    
    def compute_u(self, 
                  x: int | np.ndarray, 
                  T: int | np.ndarray):
        r"""Compute the flow velocity u(x, t) at given time and positions.

        .. math::
                u(x,t) = 
                \begin{cases}
                    0 & \text{if } x \leq x_A(t), \\\\
                    \frac{2\sqrt{g h_l}}{3} + \frac{2(x - x_0)}{3t} + \frac{g^2}{C^2} \alpha_2 t & \text{if } x_A(t) < x \leq x_B(t), \\\\
                    0 & \text{if } x_B(t) < x,
                \end{cases}

        Parameters
        ----------
        x : int or np.ndarray
            Spatial positions.
        T : int or np.ndarray
            Time instant.

        Notes
        -----
        Updates the internal `_u`, `_x`, `_t` attributes with the computed result.
        """
        if isinstance(x, int):
            x = [x]
        self._x = x
        self._t = T

        if isinstance(T, int):
            u = []
            for i in x:
                if i <= self.xa(T):
                    u.append(0)
                elif self.xa(T) < i <= self.xb(T):
                    u_val = ((2*np.sqrt(self._g*self._hl))/3) + ((2*(i-self._x0))/3*T) + ((self._g**2)/(self._c**2)) * self.alpha2(i, T) * T
                    u.append(u_val)
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
                    elif self.xa(t) < i <= self.xb(t):
                        u_val = ((2*np.sqrt(self._g*self._hl))/3) + ((2*(i-self._x0))/3*t) + ((self._g**2)/(self._c**2)) * self.alpha2(i, t) * t
                        sub_u.append(u_val)
                    else:
                        sub_u.append(0)
                u.append(sub_u)
            self._u = np.array(u)


class Mangeney_dry(Depth_result):
    r"""Dam-break solution on an inclined dry domain with friction using shallow water theory.

    This class implements the 1D analytical Stocker's solution of an ideal dam break on a dry domain.
    The dam break is instantaneous, over an inclined and flat surface with friction.
    It computes the flow height (took normal to the surface) and velocity over space and time with an 
    infinitely-long fluid mass on an infinite surface.
    
    MANGENEY, A., HEINRICH, P., et ROCHE, R. Analytical solution for testing debris avalanche 
    numerical models. Pure and Applied Geophysics, 2000, vol. 157, p. 1081-1096.

    Attributes:
    -----------
        _delta : float 
            Dynamic friction angle, in radian.
        _h0 : int
            Initial water depth.
        _c0 : float
            Initial wave propagation speed.
        _m : float
            Constant horizontal acceleration of the front.
    
    Parameters:
    -----------
        theta : float
            Angle of the surface, in degree.
        delta : float
            Dynamic friction angle (20°-40° for debris avalanche), in degree.
        h_0 : int
            Initial water depth.
    """    
    def __init__(self,
                 theta: float,
                 delta: float,
                 h_0: int,  
                 ):
        super().__init__(theta=np.radians(theta))
        self._delta = np.radians(delta)
        self._h0 = h_0
        self._c0 = self.compute_c0()
        self._m = self.compute_m()
        
        
        print(f"delta: {self._delta}, theta: {self._theta}, m: {self._m}, c0: {self._c0}")


    def xa(self, t: int) -> float:
        r"""
        Front of the flow:
        
        .. math::
            x_A(t) = \frac{1}{2}mt - 2 c_0 t

        Parameters
        ----------
        t : int
            Time instant.

        Returns
        -------
        float
            Position of the front edge of the fluid, in negative axis.
        """
        return 0.5*self._m*t**2 - (2*self._c0*t)
    
    
    def xb(self, t: int) -> float:
        r"""
        Edge of the quiet area:
        
        .. math::
            x_B(t) = \frac{1}{2}mt + c_0 t

        Parameters
        ----------
        t : int
            Time instant.

        Returns
        -------
        float
            Position of the edge of the quiet region, in negative axis.
        """
        return 0.5*self._m*t**2 + (self._c0*t)


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
            m = -g \sin{\theta} + g \cos{\theta} \tan{\delta}
            
        Returns
        -------
        float
            Value of the constant horizontal acceleration of the front.
        """
        return -1 * (self._g * np.cos(self._theta)) * (np.tan(self._theta)-np.tan(self._delta))


    def compute_h(self, 
                  x: int | np.ndarray, 
                  T: int | np.ndarray):
        r"""Compute the flow height h(x, t) at given time and positions.

        .. math::
                h(x, t) = 
                \begin{cases}
                    0 & \text{if } x \leq x_A(t), \\\\
                    \frac{1}{9g cos(\theta)} \left( \frac{x}{t} + 2 c_0 - \frac{1}{2} m t \right)^2 & \text{if } x_A(t) < x \leq x_B(t), \\\\
                    h_0 & \text{if } x_B(t) < x,
                \end{cases}

        Parameters
        ----------
        x : int or np.ndarray
            Spatial positions.
        T : int or np.ndarray
            Time instant.

        Notes
        -----
        Updates the internal '_h', '_x', '_t' attributes with the computed result.
        """
        if isinstance(x, int):
            x = [x]
        self._t = T
        self._x = x
        
        x = [-i for i in x[::-1]]

        if isinstance(T, int):
            h = []
            
            for i in x:
                if i <= self.xa(T):
                    h.append(0)
                elif self.xa(T) < i < self.xb(T):
                    h.append( (1/(9*self._g*np.cos(self._theta))) * ( (i/T) + (2 * self._c0) - (0.5*T*self._m))**2 )
                else:
                    h.append(self._h0)

            if all(v == 0 for v in h):
                h[-1] = self._h0
            
            self._h = np.array(h[::-1])
            
        else:
            h = []
            for t in T:
                sub_h = []
                for i in x:
                    if i <= self.xa(t):
                        sub_h.append(0)
                    elif self.xa(t) < i < self.xb(t):
                        sub_h.append( (1/(9*self._g*np.cos(self._theta))) * ( (i/t) + (2 * self._c0) - (0.5*t*self._m))**2 )
                    else:
                        sub_h.append(self._h0)
                
                if all(v == 0 for v in h):
                    sub_h[-1] = self._h0
                
                h.append(sub_h[::-1])
                
            self._h = np.array(h)


    def compute_u(self, 
                  x: int | np.ndarray, 
                  T: int | np.ndarray):
        r"""Compute the flow velocity u(x, t) at given time and positions.

        .. math::
                u(x,t) = 
                \begin{cases}
                    0 & \text{if } x \leq x_A(t), \\\\
                    \frac{2}{3} \left( \frac{x}{t} - c_0 + mt \right) & \text{if } x_A(t) < x \leq x_B(t), \\\\
                    0 & \text{if } x_B(t) < x,
                \end{cases}

        Parameters
        ----------
        x : int or np.ndarray
            Spatial positions.
        T : int or np.ndarray
            Time instant.

        Notes
        -----
        Updates the internal `_u`, `_x`, `_t` attributes with the computed result.
        """
        if isinstance(x, int):
            x = [x]
        self._t = T
        self._x = x
        
        x = [-i for i in x[::-1]]

        if isinstance(T, int):
            u = []
            for i in x:
                if i <= self.xa(T):
                    u.append(0)
                elif self.xa(T) < i <= self.xb(T):
                    u_val = (2/3) * ( (x/T) - self._c0 + self._m * T )
                    u.append(u_val)
                else:
                    u.append(0)
            self._u = np.array(u[::-1])
        
        else:
            u = []
            for t in T:
                sub_u = []
                for i in x:
                    if i <= self.xa(t):
                        sub_u.append(0)
                    elif self.xa(t) < i <= self.xb(t):
                        u_val = (2/3) * ( (i/t) - self._c0 + self._m * t )
                        sub_u.append(u_val)
                    else:
                        sub_u.append(0)
                u.append(sub_u[::-1])
            self._u = np.array(u)


class Shape_result(ABC):
    """Abstract base class representing shape results of a simulated flow.

    This class defines a common interface for flow simulation that compute 
    the geometry of the final shape of a flow simulation. 

    Attributes:
    -----------
        _g : float 
            Gravitational constant.
        _theta : float
            Angle of the surface, in radian.
        _x : int or np.ndarray
            Spatial coordinates.
        _h : np.ndarray
            Flow height depending on space.
    
    Parameters:
    -----------
        theta : float, optional
            Angle of the surface, in radian, by default 0.
    """
    def __init__(self,
                 theta: float=0):
        self._g = 9.81
        self._theta = theta
        
        self._x = None
        self._h = None
    
    
    @property
    def h(self):
        """Accessor of the shape h of the flow.
        
        Returns
        -------
        self._h : np.ndarray
            Flow height in self._x. If None, no solution computed.
        """
        return self._h


    @property
    def x(self):
        """Accessor of the spatial distribution of the computed solution.
        
        Returns
        -------
        self._x : np.ndarray
            Spatial distribution of the computed solution. If None, no solution computed.
        """
        return self._x


    def show_res(self, 
                 show_slop=False,
                 x_unit:str = "m",
                 h_unit:str = "m",
                ):
        """Plot the shape results.

        Parameters
        ----------
        show_slop : bool, optional
            If True, plot the slop of the surface.
        x_unit: str
            Space unit.
        h_unit: str
            Height unit.
        """
        inclined_surf = None
        z_surf = [0, 0]
        
        if self._theta is not None and show_slop:
            z_surf = [z_surf[0], -self._x[-1]*np.tan(self._theta)]
            inclined_surf = np.linspace(z_surf[0], z_surf[1], len(self._x))
        
        if self._x is not None and self._h is not None:
            if self._h.ndim == 1:
                if self._theta is not None and show_slop:
                    h_inclined = [(self._h[i]/np.cos(self._theta)) + inclined_surf[i] for i in range(len(self._h))]
                    plt.plot(self._x, h_inclined, color='black', linewidth=1)
                else:
                    plt.plot(self._x, self._h, color='black', linewidth=1)

            plt.plot([self._x[0], self._x[-1]], z_surf, color='black', linewidth=2)
            
            plt.title("Flow height")
            plt.xlabel(f"x [{x_unit}]")
            plt.ylabel(f"h [{h_unit}]")
            plt.show()


class Coussot_shape(Shape_result):
    r"""Shape solution on an inclined dry domain without friction.

    This class implements the final shape of a simulated flow.
    The flow is over an inclined and flat surface without friction with a finite volume of fluid.
    It computes the spatial coordinates from the flow lenght and height.
    
    Coussot P., Proust S., Ancey C. Rheological interpretation of deposits of yield stress fluids, 
    Journal of Non-Newtonian Fluid Mechanics, 1996, vol 66(1), p. 55-70, DOI:10.1016/0377-0257(96)01474-7.

    Attributes:
    -----------
        _l0 : int 
            Length of the deposit.
        _L0 : int 
            Normalized length of the deposit.
        _hmax : int 
            Max fluid depth.
        _Hmax : int 
            Normalized max fluid depth.
        _rho : float
            Fluid density.
        _tau : float
            Threshold constraint.
        _X0 : float
            Normalized spatial coordinates of the maximal height.
        _X : float or np.ndarray
            Normalized spatial coordinates.
        _H : float or np.ndarray
            Normalized fluid depth.
    
    Parameters:
    -----------
        l0 : int 
            Length of the deposit.
        rho : float
            Fluid density.
        tau : float
            Threshold constraint.
        theta : float, optional
            Angle of the surface, in degree, by default 0.
        H_size : int, optional
            Number of value wanted in the H array, by default 100.
        hmax : float; optional
            Max fluid depth.
    """   
    def __init__(self, 
                 l0: int, #Longueur du dépôt ==> distance x0-xf (rupture barrage - front de l'écoulement)
                 rho: float,
                 tau: float,
                 theta: float=0,
                 H_size: int=100,
                 hmax: float=None):
        super().__init__(np.radians(theta))
        self._rho = rho
        self._tau = tau
        
        self._l0 = l0
        self._L0 = self.x_to_X(l0)
        self._X0 = self.compute_X0()
        self._hmax = hmax
        self._Hmax = self.compute_Hmax(hmax)
        
        self._X = None
        self._H = self.compute_H(H_size)
        
        print("X0: ", self._X0)
        print("Hmax: ", self._Hmax)
        print("l0: ", self._l0)
        print("L0: ", self._L0)


    def h_to_H(self, 
               h: float
               ) -> float:
        r"""Normalize the fluid depth by following:
        
        .. math::
            H = \frac{\rho g h \sin(\theta)}{\tau_c}
            
        If :math:`\theta = 0`, the the expression is:
        
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
               H: np.ndarray
               ) -> np.ndarray:
        r"""Find the original value of the fluid depth from the normalized one
        by following:
        
        .. math::
            h = \frac{H \tau_c}{\rho g \sin(\theta)} 

        If :math:`\theta = 0`, the the expression is:
        
        .. math::
            h = \frac{H \tau_c}{\rho g} 
    
        Parameters
        ----------
        H : np.ndarray
            Normalized values of the fluid depth.

        Returns
        -------
        np.ndarray
            True values of the fluid depth.
        """
        h = []
        for v in H:
            if self._theta == 0:
                h.append((v*self._tau)/(self._rho*self._g))
            else:    
                h.append((v*self._tau)/(self._rho*self._g*np.sin(self._theta)))
        return np.array(h)


    def x_to_X(self, 
               x: float
               ) -> float:
        r"""Normalize the spatial coordinates by following:
        
        .. math::
            X = \frac{\rho g x (\sin(\theta))^2}{\tau_c \cos(\theta)}
            
        If :math:`\theta = 0`, the the expression is:
        
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
               X: np.ndarray
               ) -> np.ndarray:
        r"""Find the original value of the spatial coordinates from the normalized one
        by following:
        
        .. math::
            x = \frac{X \tau_c \cos(\theta)}{\rho g (\sin(\theta))^2} 

        If :math:`\theta = 0`, the the expression is:
        
        .. math::
            x = \frac{X \tau_c}{\rho g} 
    
        Parameters
        ----------
        X : np.ndarray
            Normalized values of the spatial coordinates.

        Returns
        -------
        np.ndarray
            True values of the spatial coordinate.
        """
        x = []
        for v in X:
            if self._theta == 0:
                x.append((v*self._tau)/(self._rho*self._g))
            else:
                x.append((v*self._tau*np.cos(self._theta))/(self._rho*self._g*np.sin(self._theta)*np.sin(self._theta)))
        return x


    def compute_X0(self) -> float:
        r"""Compute the normalized coordinate of the maximal fluid depth:
        
        .. math::
            X_0 = \sqrt{1-exp(-L_0)} + \ln(1+\sqrt{1-exp(-L_0)})
            
        Returns
        -------
        float
            X-coordinate of the maximal fluid depth.
        """
        return np.sqrt(1-np.exp(-self._l0)) + np.log(1+np.sqrt(1-np.exp(-self._l0)))
        
    
    def compute_Hmax(self, 
                     hmax: float=None) -> float:
        r"""Compute the normalized maximal fluid depth:
        
        .. math::
            H_{max} = \sqrt{1-exp(-L_0)}
            
        Or if hmax is define: 
        
        .. math::
            H_{max} = \frac{\rho g h_{max} \sin(\theta)}{\tau_c}
              
        Returns
        -------
        float
            Maximal fluid depth (dimensionless).
        """
        return np.sqrt(1-np.exp(-self._l0))


    def compute_H(self,
                 size: int
                 ) -> np.ndarray:
        r"""Create an array of fluid depth from the normalized maximal fluid depth:

        Parameters
        ----------
        size : int
            Number of value wanted.      

        Returns
        -------
        np.ndarray
            Array of fluid depth (dimensionless).
        """
        H = np.linspace(0, 0.99, size)        
        
        return H


    def compute_Xx(self) -> None:
        r"""Compute the normalize x-coordinates (:math:`0 < X < L_0`) from the fluid depth by following:
        
        .. math::
                X = 
                \begin{cases}
                    H - \ln(1 + H) & \text{if } 0 \leq X \leq X_0, \\\\
                    H + L_0 + \ln(1 - H) & \text{if } X_0 \leq X \leq L_0
                \end{cases}     
        """
        self._h = self.H_to_h(np.concatenate((self._H, np.flip(self._H)), axis=None)) 
        xr = []
        xl = []        
        
        for H_val in self._H:
            if self._theta == 0:
                x_left = (H_val*H_val)/2
                x_right = (H_val*H_val)/2
            else:
                x_left = H_val - np.log(1 + H_val)
                if H_val < 1:
                    x_right = H_val + self._l0 + np.log(1 - H_val)
                else:
                    x_right = self._X0

            xl.append(x_left)
            xr.append(x_right)
        
        if self._theta == 0:
            xr = [(-1*(i-max(xr)))*(self._L0/max(xr)) for i in xr]
            xl = [(i-max(xl))*(self._L0/max(xl)) for i in xl]

        self._X = np.concatenate((xl, np.flip(xr)), axis=None)
        self._x = self.X_to_x(self._X)


    def compute_Xx_back(self) -> None:
        r"""Compute the normalize x-coordinates (:math:`0 < X < X_0`) from the fluid depth by following the first equation of:
        
        .. math::
                X = 
                \begin{cases}
                    H - \ln(1 + H) & \text{if } 0 \leq X \leq X_0, \\\\
                    H + L_0 + \ln(1 - H) & \text{if } X_0 \leq X \leq L_0
                \end{cases}     
        """
        self._h = self.H_to_h(self._H)
        X = []
        
        for H_val in self._H:
            if self._theta == 0:
                X.append((H_val*H_val)/2)
            else:
                X.append(H_val - np.log(1 + H_val))

        self._X = X
        self._x = self.X_to_x(self._X)


    def compute_Xx_front(self) -> None:
        r"""Compute the normalize x-coordinates (:math:`X_0 < X < L0`) from the fluid depth by following the second equation of:
        
        .. math::
                X = 
                \begin{cases}
                    H - \ln(1 + H) & \text{if } 0 \leq X \leq X_0, \\\\
                    H + L_0 + \ln(1 - H) & \text{if } X_0 \leq X \leq L_0
                \end{cases}     
        """
        self._h = self.H_to_h(self._H)
        X = []
        
        for H_val in self._H:
            if self._theta == 0:
                X.append((H_val*H_val)/2)
            else:
                X.append(H_val + self._L0 + np.log(1 - H_val))
        
        X = np.array(X)
        X[X<0] = 0
        
        if self._theta == 0:
            X = [-1*(i-max(X)) for i in X]

        self._X = X
        self._x = self.X_to_x(self._X)
        
             
    def compute_Xx_front_remaitre(self):
        r"""Compute the normalize x-coordinates (:math:`X_0 < X < L0`) from the fluid depth by following:
        
        .. math::
                X = -H - \ln(1-H) 
        """
        if self._theta == 0:
            self._h = np.concatenate((self.H_to_h(self._H), np.flip(self.H_to_h(self._H))), axis=None) * (self._hmax/np.max(self.H_to_h(self._H)))
        else:
            self._h = self.H_to_h(self._H)

        X = []
        for H_val in self._H:
            X.append(((-1*H_val) - np.log(1-(H_val))))

        if self._theta == 0:
            X_l = [(i - max(X))*(self._L0/max(X)) for i in X]
            X_r = [(-1*(i - max(X)))*(self._L0/max(X)) for i in X]
            self._X = np.concatenate((X_l, np.flip(X_r)), axis=None)
        else:
            self._X = [(-1*(i - max(X))) for i in X]
        self._x = self.X_to_x(self._X)


    def compute_Hf(self) -> float:
        Hmax = self.compute_Hmax()
        val = 1 - ((2/Hmax)*(1-np.log(2/Hmax)))
        print(self.H_to_h([val*Hmax]))


class Front_result:
    """Class computing front position of a simulated flow.

    This class defines multiple methods for flow simulation that compute 
    the position of the front flow at the specified moment. 

    Attributes:
    -----------
        _g : float 
            Gravitational constant.
        _theta : float
            Angle of the surface, in radian.
        _h0 : int
            Initial fluid depth.
        _xf : int or np.ndarray
            Spatial coordinates of the front flow.
        _t : int #TODO
            Time instant.
    
    Parameters:
    -----------
        theta : float, optional
            Angle of the surface, in radian, by default 0.
        h0 : int
            Initial fluid depth, by default 1.
    """
    def __init__(self,
                 theta: float=0,
                 h0: int=1):
        self._g = 9.81
        self._theta = np.radians(theta)
        
        self._h0 = h0
        
        self._xf = {}
        self._labels = {}
    

    def xf_mangeney(self, 
                    t: int,
                    delta: float
                    ) -> float:
        r"""
        Mangeney's equation for a dam-break solution over an infinite inclined dry domain with friction
        and an infinitely-long fluid mass:
        
        .. math::
            x_f(t) = \frac{1}{2}mt - 2 c_0 t
            
        with :math:`c_0` the initial wave propagation speed defined by:
        
        .. math::
            c_0 = \sqrt{g h_0 \cos{\theta}}

        and :math:`m` the constant horizontal acceleration of the front defined by:
    
        .. math::
            m = -g \sin{\theta} + g \cos{\theta} \tan{\delta}
    
        Parameters
        ----------
        t : int
            Time instant.
            
        delta : float
            Dynamic friction angle, in degree.        

        Returns
        -------
        float
            Position of the front edge of the fluid.
        """
        m = -1 * (self._g * np.cos(self._theta)) * (np.tan(self._theta)-np.tan(np.radians(delta)))
        c0 = np.sqrt(self._g * self._h0 * np.cos(self._theta))
        xf = -1*(0.5*m*(t**2) - (2*c0*t))
        
        if t in self._labels:
            if f"Mangeney d{delta}" not in self._labels[t] :
                 self._labels[t].append(f"Mangeney d{delta}")
                 self._xf[t].append(xf)
        else:
            self._labels[t] = [f"Mangeney d{delta}"]
            self._xf[t] = [xf]
                
        # if f"Mangeney d{delta}" not in self._labels :
        #     if self._xf is None:
        #         self._xf = [xf]
        #     else:
        #         self._xf.append(xf)

        #     self._labels.append(f"Mangeney d{delta}")
        #     self._t = t
            
        return xf


    def xf_dressler(self, 
                    t: int,
                    ) -> float:
        r"""
        Dressler's equation for a dam-break solution over an infinite inclined dry domain with friction:
        
        .. math::
            x_f(t) = 2 t \sqrt{g h_0}
    
        Parameters
        ----------
        t : int
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
            
        # if "Dressler" not in self._labels:
        #     if self._xf is None:
        #         self._xf = [xf]
        #     else:
        #         self._xf.append(xf)

        #     self._labels.append("Dressler")
        #     self._t = t
            
        return xf


    def xf_ritter(self, 
                    t: int,
                    ) -> float:
        r"""
        Ritter's equation for a dam-break solution over an infinite inclined dry domain without friction:
        
        .. math::
            x_f(t) = 2 t \sqrt{g h_0}
    
        Parameters
        ----------
        t : int
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
        
        # if "Ritter" not in self._labels:
        #     if self._xf is None:
        #         self._xf = [xf]
        #     else:
        #         self._xf.append(xf)

        #     self._labels.append("Ritter")
        #     self._t = t
            
        return xf


    def xf_stocker(self, 
                    t: int,
                    hr: int
                    ) -> float:
        r"""
        Stocker's equation for a dam-break solution over an infinite inclined wet domain without friction:
        
        .. math::
            x_f(t) =t \cdot \frac{2 c_m^2 \left( \sqrt{g h_l} - c_m \right)}{c_m^2 - g h_r}
            
        with :math:`c_m` the critical velocity defined by:

        .. math::
            -8.g.hr.cm^{2}.(g.hl - cm^{2})^{2} + (cm^{2} - g.hr)^{2} . (cm^{2} + g.hr) = 0
    
        Parameters
        ----------
        t : int
            Time instant.
        hr : int
            Fluid depth at the right of the dam.

        Returns
        -------
        float
            Position of the front edge of the fluid.
        """
        def equation_cm(cm) -> float:
            return -8 * self._g * hr * cm**2 * (self._g * self._h0 - cm**2)**2 + (cm**2 - self._g * hr)**2 * (cm**2 + self._g * hr)

        guesses = np.linspace(0.01, 1000, 1000)
        solutions = []

        for guess in guesses:
            sol = fsolve(equation_cm, guess)[0]

            if abs(equation_cm(sol)) < 1e-6 and not any(np.isclose(sol, s, atol=1e-6) for s in solutions):
                solutions.append(sol)

        for sol in solutions:
            hm = sol**2 / self._g
            if hm < self._h0 and hm > hr:
                find = True
                self._cm = sol
                break
            else:
                find = False

        if find:
            xf = t * (((2*self._cm**2)*(np.sqrt(self._g*self._h0)-self._cm)) / ((self._cm**2) - (self._g*hr)))
            
            if t in self._labels:
                if "Stocker" not in self._labels[t] :
                    self._labels[t].append("Stocker")
                    self._xf[t].append(xf)
            else:
                self._labels[t] = ["Stocker"]
                self._xf[t] = [xf]
            
            # if "Stocker" not in self._labels:
            #     if self._xf is None:
            #         self._xf = [xf]
            #     else:
            #         self._xf.append(xf)

            #     self._labels.append("Stocker")
            #     self._t = t
                
            return xf
            
        else:
            print("Didn't find cm, try with greater range of value.")
            return None


    # def show_res(self, 
    #              x_unit: str='m'):
    #     """Plot the front position.

    #     Parameters
    #     ----------
    #     x_unit: str
    #         Space unit.
    #     """
    #     print(self._xf)
    #     print(self._labels)
    #     y_levels = np.arange(len(self._xf))[::-1]
     
    #     fig, ax = plt.subplots(figsize=(10, 5))
        
    #     for x, y, label in zip(self._xf, y_levels, self._labels):
    #         ax.vlines(x, y - 0.3, y + 0.3, color='black', linewidth=2)
    #         ax.text(x + 0.5, y, f"{x:.4f}", rotation=90, va='center')
        
    #     ax.set_yticks(y_levels)
    #     ax.set_yticklabels(self._labels)
    #     ax.invert_yaxis()

    #     ax.set_xlim(left=0)
    #     ax.set_xlim(right=max(self._xf)+5)


    #     ax.set_xlabel(f"x [{x_unit}]")
    #     ax.set_title(f"Flow front positions at t = {self._t}" if self._t else "Flow front positions")

    #     ax.grid(True, axis='x')

    #     plt.tight_layout()
    #     plt.show()


    def show_fronts_over_time(self, x_unit="m"):
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
        colors = cm.viridis(np.linspace(0, 1, len(sorted_times)))

        for color, t in zip(colors, sorted_times):
            x_list = self._xf[t]
            label_list = self._labels[t]

            for x, label in zip(x_list, label_list):
                y = y_levels[label]
                ax.vlines(x, y - 0.3, y + 0.3, color=color, linewidth=2)
                ax.text(x + 0.5, y, f"{x:.2f}", rotation=90, va='center', fontsize=8, color=color)

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
            Line2D([0], [0], color=color, lw=2, label=f"t = {t}")
            for color, t in zip(colors, sorted_times)
        ]
        ax.legend(handles=legend_elements, title="Time steps", loc="lower right")

        plt.tight_layout()
        plt.show()


# a = Front_result(30, 20)
# print(a.xf_mangeney(5, 0))
# print(a.xf_mangeney(5, 25))

# print(a.xf_mangeney(10, 0))
# print(a.xf_mangeney(10, 25))

# print(a.xf_dressler(5))
# print(a.xf_ritter(5))
# print(a.xf_stocker(5, 0.01))

# a.show_fronts_over_time()

# A = Coussot_shape(l0=16.11, rho=1000, tau=500, theta=10, hmax=0.5)
# A.compute_Xx_front()

# B = Coussot_shape(l0=10, rho=1000, tau=500, theta=10, hmax=0.5)
# B.compute_Xx_front_remaitre()

# fig, ax = plt.subplots()
# ax.plot(A.x, A.h, label="First solution")
# ax.plot(B.x, B.h, label="Second solution")
# ax.legend()
# ax.set_xlabel('x [m]')
# ax.set_ylabel('h [m]')

# ax.plot([A.x[0], A.x[-1]], [0, 0], color='black', linewidth=2)
# plt.show()