# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 12:09:41 2023

@author: peruzzetto
"""
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fsolve
from abc import ABC, abstractmethod


class Depth_result(ABC):
    """Abstract base class representing simulation results for flow depth and velocity.

    This class defines a common interface for analytical solution that compute flow height 
    h(x,t) and flow velocity u(x,t).

    Parameters
    ----------
    theta : float, optional
        Angle of the surface, by default 0.
    """
    def __init__(self, 
                 theta: float=0):
        self._g = 9.18
        self._theta = theta
        self._x = None
        self._t = None
        self._h = None
        self._u = None


    @abstractmethod
    def h(self, 
          x: int | np.ndarray, 
          t: int | float
          ) -> None:
        """Virtual function that compute the flow height 'h' at given space and time.

        Parameters
        ----------
        x : int or np.ndarray
            Spatial coordinates.
        t : int | float
            Time instant.
        """
        pass


    @abstractmethod
    def u(self, 
          x: int | np.ndarray, 
          t: int | float
          ) -> None:
        """Virtual function that compute the flow velocity 'u' at given space and time.

        Parameters
        ----------
        x : int or np.ndarray
            Spatial coordinates.
        t : int | float
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
                 z_surf=[0, 0], 
                 show_h=False, 
                 show_u=False
                ):
        """Plot the simulation results.

        Parameters
        ----------
        z_surf : list of float, optional
            Surface elevation to be displayed as a reference line, by default [0, 0].
        show_h : bool, optional
            If True, plot the flow height ('h') curve.
        show_u : bool, optional
            If True, plot the flow velocity ('u') curve.
        """
        if show_h:
            plt.plot(self._x, self._h)
        if show_u:
            plt.plot(self._x, self._u)

        plt.plot([self._x[0], self._x[-1]], z_surf, color='black', linewidth=2)
        plt.show()


class Dam_break_wet_domain(Depth_result):
    """Dam-break solution on a wet domain using shallow water theory.

    This class implements the 1D analytical Stoker's solution of an ideal dam break on a wet wet domain.
    The dam break is instantaneous, over an horizontal and flat surface with no friction.
    It computes the flow height (took verticaly) and velocity over space and time, based on the equation implemanted
    in SWASHES, based on Stoker's equation.
    
    Stoker JJ. Water Waves: The Mathematical Theory with Applications, Pure and Applied Mathematics, 
    Vol. 4. Interscience Publishers: New York, USA, 1957.

    Parameters
    ----------
    x_0 : int
        Initial dam location (position along x-axis).
    l : int
        Spatial domain length.
    h_l : int
        Water depth to the left of the dam.
    h_r : int
        Water depth to the right of the dam.
    h_m : int, optional
        Intermediate height used to compute the critical speed cm. If not provided,
        it will be computed numerically via the 'compute_cm()' method.
    """
    def __init__(self, 
                 x_0: int, 
                 l: int, 
                 h_l: int, 
                 h_r: int, 
                 h_m: int=None
                 ):
        super().__init__()
        self._x0 = x_0
        self._l = l
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
            x_A = x_0 - t \sqrt{g h_l}

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


    def equation_cm(self, cm):
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


    def compute_cm(self):
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
                  t: int):
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
        t : int
            Time instant.

        Notes
        -----
        Updates the internal '_h', '_x', '_t' attributes with the computed result.
        """
        if self._cm is not None:
            if isinstance(x, int):
                x = [x]
            self._x = x
            self._t = t

            h = []
            for i in x:
                if i <= self.xa(t):
                    h.append(self._hl)
                elif i > self.xa(t) and i <= self.xb(t):
                    h.append((4/(9*self._g))*(np.sqrt(self._g *
                             self._hl)-((i-self._x0)/(2*t)))**2)
                elif i > self.xb(t) and i <= self.xc(t):
                    h.append((self._cm**2)/self._g)
                else:
                    h.append(self._hr)
            self._h = h

        else:
            print("No critical velocity found")

    def compute_u(self, 
                  x: int | np.ndarray, 
                  t: int):
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
        t : int
            Time instant.

        Notes
        -----
        Updates the internal `_u`, `_x`, `_t` attributes with the computed result.
        """
        if self._cm is not None:
            if isinstance(x, int):
                x = [x]
            self._x = x
            self._t = t

            u = []
            for i in x:
                if i <= self.xa(t):
                    u.append(0)
                elif i > self.xa(t) and i <= self.xb(t):
                    u.append((2/3)*(((i-self._x0)/t) +
                             np.sqrt(self._g*self._hl)))
                elif i > self.xb(t) and i <= self.xc(t):
                    u.append(2*(np.sqrt(self._g*self._hl) - self._cm))
                else:
                    u.append(0)
            self._u = u

        else:
            print("First define cm")


class Dam_break_dry_domain(Depth_result):
    def __init__(self, x_0, l, h_l, h_r=0):
        super().__init__()
        self._x0 = x_0
        self._l = l
        self._hl = h_l
        self._hr = h_r

    def xa(self, t):
        return self._x0 - (t * np.sqrt(self._g*self._hl))

    def xb(self, t):
        return self._x0 + (2 * t * np.sqrt(self._g*self._hl))

    def compute_h(self, x, t):
        if isinstance(x, int):
            x = [x]
        self._x = x
        self._t = t

        h = []
        for i in x:
            if i <= self.xa(t):
                h.append(self._hl)
            elif i > self.xa(t) and i <= self.xb(t):
                h.append((4/(9*self._g)) *
                         (np.sqrt(self._g*self._hl)-((i-self._x0)/(2*t)))**2)
            else:
                h.append(self._hr)
        self._h = h

    def compute_u(self, x, t):
        if isinstance(x, int):
            x = [x]
        self._x = x
        self._t = t

        u = []
        for i in x:
            if i <= self.xa(t):
                u.append(0)
            elif i > self.xa(t) and i <= self.xb(t):
                u.append((2/3)*(((i-self._x0)/t) + np.sqrt(self._g*self._hl)))
            else:
                u.append(0)
        self._u = u


class Dam_break_friction(Depth_result):
    def __init__(self, x_0, l, h_l, h_r=0, C=40):
        super().__init__()
        self._x0 = x_0
        self._l = l
        self._hl = h_l
        self._hr = h_r
        self._c = C

    def xa(self, t):
        return self._x0 - (t * np.sqrt(self._g*self._hl))

    def xb(self, t):
        return self._x0 + (2 * t * np.sqrt(self._g*self._hl))

    def alpha1(self, x, t):
        xi = (x-self._x0)/(t*np.sqrt(self._g*self._hl))
        # return (6 / (5*(2 - (x/(t*np.sqrt(self._g*self._hl)))))) - (2/3) + (4*np.sqrt(3)/135)*((2 - (x/(t*np.sqrt(self._g*self._hl))))**(3/2))
        return (6 / (5 * (2 - xi))) - (2 / 3) + (4 * np.sqrt(3) / 135) * (2 - xi) ** (3 / 2)

    def alpha2(self, x, t):
        return (12 / (2 - (x/(t*np.sqrt(self._g*self._hl))))) - (8/3) + (8*np.sqrt(3)/189)*((2 - (x/(t*np.sqrt(self._g*self._hl))))**(3/2)) - (108/(7*(2 - (x/(t*np.sqrt(self._g*self._hl))))))

    def compute_h(self, x, t):
        if isinstance(x, int):
            x = [x]
        self._x = x
        self._t = t
        # print(self.xa(t), self.xb(t))
        h = []
        for i in x:
            if i <= self.xa(t):
                h.append(self._hl)
            elif i > self.xa(t) and i <= self.xb(t):
                # h.append( (1/self._g) * ( ((2/3)*np.sqrt(self._g*self._hl)) - ((i-self._x0)/(3*t)) + ((self._g**2)/(self._c**2))*self.alpha1(i-self._x0, t)*t ) )
                term = ((2/3)*np.sqrt(self._g*self._hl)) - ((i - self._x0) /
                                                            (3*t)) + ((self._g**2)/(self._c**2)) * self.alpha1(i, t) * t
                h_val = (1/self._g) * term**2
                if h_val > h[-1]:
                    term = ((2/3)*np.sqrt(self._g*self._hl)) - ((i - self._x0) /
                                                                (3*t)) + ((self._g**2)/(self._c**2)) * self.alpha1(i, t) * t
                    h_val = (1/self._g) * term**2
                h.append(h_val)
            else:
                h.append(self._hr)
        self._h = h

    def compute_u(self, x, t):
        if isinstance(x, int):
            x = [x]
        self._x = x
        self._t = t

        u = []
        for i in x:
            if i <= self.xa(t):
                u.append(0)
            elif i > self.xa(t) and i <= self.xb(t):
                u.append((2/3)*(((i-self._x0)/t) + np.sqrt(self._g*self._hl)))
            else:
                u.append(0)
        self._u = u


class Shape_result:
    def __init__(self):
        pass

    def shape(self):
        return None

    def show_res(self):
        return None


class Front_result:
    def __init__(self):
        pass

    def xf(self):
        return None

    def show_res(self):
        return None


if __name__ == "__main__":
    # A = Dam_break_wet_domain(5, 10, 0.005, 0.001)
    B = Dam_break_dry_domain(5, 10, 0.005)
    C = Dam_break_friction(1000, 2000, 6)
    T = [i*5 for i in range(0, 1)]
    x = np.linspace(0, 2000, 100)
    for t in T:
        C.compute_h(x, t)
        # B.u(x, t)
        C.show_res(show_h=True)
        print(C.h)
