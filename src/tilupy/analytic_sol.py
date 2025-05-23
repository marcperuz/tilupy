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
        inclined_surf = None
        z_surf = [0, 0]
        if self._theta is not None:
            z_surf = [z_surf[0], -self._x[-1]*np.tan(self._theta)]
            inclined_surf = np.linspace(z_surf[0], z_surf[1], len(self._x))
        if show_h:
            if self._theta is not None:
                h_inclined = [(self._h[i]/np.cos(self._theta)) + inclined_surf[i] for i in range(len(self._h))]
                plt.plot(self._x, h_inclined)
            else:
                plt.plot(self._x, self._h)
        if show_u:
            plt.plot(self._x, self._u)

        plt.plot([self._x[0], self._x[-1]], z_surf, color='black', linewidth=2)
        plt.show()


class Dam_break_wet_domain(Depth_result):
    r"""Dam-break solution on a wet domain using shallow water theory.

    This class implements the 1D analytical Stocker's solution of an ideal dam break on a wet domain.
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
                # elif i > self.xa(t) and i <= self.xb(t):
                elif self.xa(t) < i <= self.xb(t):
                    h.append((4/(9*self._g))*(np.sqrt(self._g *
                             self._hl)-((i-self._x0)/(2*t)))**2)
                elif self.xb(t) < i <= self.xc(t):
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
    r"""Dam-break solution on a dry domain using shallow water theory.

    This class implements the 1D analytical Ritter's solution of an ideal dam break on a dry domain.
    The dam break is instantaneous, over an horizontal and flat surface with no friction.
    It computes the flow height (took verticaly) and velocity over space and time, based on the equation implemanted
    in SWASHES, based on Ritter's equation.
    
    Ritter A. Die Fortpflanzung der Wasserwellen. Zeitschrift des Vereines Deuscher Ingenieure 
    August 1892; 36(33): 947-954.

    Parameters
    ----------
    x_0 : int
        Initial dam location (position along x-axis).
    l : int
        Spatial domain length.
    h_l : int
        Water depth to the left of the dam.
    h_r : int, optinal
        Water depth to the right of the dam, by default 0.
    """
    def __init__(self, 
                 x_0: int, 
                 l: int, 
                 h_l: int, 
                 h_r: int=0, 
                 ):
        super().__init__()
        self._x0 = x_0
        self._l = l
        self._hl = h_l
        self._hr = h_r


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
                  t: int):
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
        t : int
            Time instant.

        Notes
        -----
        Updates the internal '_h', '_x', '_t' attributes with the computed result.
        """
        if isinstance(x, int):
            x = [x]
        self._x = x
        self._t = t

        h = []
        for i in x:
            if i <= self.xa(t):
                h.append(self._hl)
            elif self.xa(t) < i <= self.xb(t):
                h.append((4/(9*self._g)) *
                         (np.sqrt(self._g*self._hl)-((i-self._x0)/(2*t)))**2)
            else:
                h.append(self._hr)
        self._h = h


    def compute_u(self, 
                  x: int | np.ndarray, 
                  t: int):
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
        t : int
            Time instant.

        Notes
        -----
        Updates the internal `_u`, `_x`, `_t` attributes with the computed result.
        """
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
    r"""Dam-break solution on a dry domain with friction using shallow water theory.

    This class implements the 1D analytical Dressler's solution of an ideal dam break on a dry domain with friction.
    The dam break is instantaneous, over an horizontal and flat surface with friction.
    It computes the flow height (took verticaly) and velocity over space and time, based on the equation implemanted
    in SWASHES, based on Dressler's equation.
    
    Dressler RF. Hydraulic resistance effect upon the dam-break functions. Journal of Research of the National Bureau 
    of Standards September 1952; 49(3): 217-225.

    Parameters
    ----------
    x_0 : int
        Initial dam location (position along x-axis).
    l : int
        Spatial domain length.
    h_l : int
        Water depth to the left of the dam.
    h_r : int, optinal
        Water depth to the right of the dam, by default 0.
    C : int, optional
        Chézy coefficient, by default 40.
    """
    def __init__(self, x_0, l, h_l, h_r=0, C=40):
        super().__init__()
        self._x0 = x_0
        self._l = l
        self._hl = h_l
        self._hr = h_r
        self._c = C


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


    def compute_h(self, x, t):
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
        t : int
            Time instant.

        Notes
        -----
        Updates the internal '_h', '_x', '_t' attributes with the computed result.
        """
        if isinstance(x, int):
            x = [x]
        self._x = x
        self._t = t
        # print(self.xa(t), self.xb(t))
        h = []
        for i in x:
            if i <= self.xa(t):
                h.append(self._hl)
            elif self.xa(t) < i <= self.xb(t):
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


class Dam_break_friction_inclined(Depth_result):
    r"""Dam-break solution on an inclined dry domain with friction using shallow water theory.

    This class implements the 1D analytical Stocker's solution of an ideal dam break on a dry domain.
    The dam break is instantaneous, over an inclined and flat surface with friction.
    It computes the flow height (took normal to the surface) and velocity over space and time, based on Stocker's equation.
    
    MANGENEY, A., HEINRICH, P., et ROCHE, R. Analytical solution for testing debris avalanche 
    numerical models. Pure and Applied Geophysics, 2000, vol. 157, p. 1081-1096.

    Parameters
    ----------
    theta : int
        Angle of the surface, in degree.
    delta : int
        Dynamic friction angle (20°-40° for debris avalanche), in degree.
    h_0 : int
        Initial water depth.
    """
    def __init__(self, 
                 theta: int,
                 delta: int,
                 h_0: int,  
                 ):
        super().__init__(theta=np.radians(theta))
        self._delta = np.radians(delta) #is the dynamic friction angle (20°Bd B40° for debris avalanche)
        self._h0 = h_0
        self._c0 = np.sqrt(self._g * self._h0 * np.cos(self._theta))
        
        self._m = - (self._g * np.sin(self._theta)) + (self._g * np.cos(self._theta)) * np.tan(self._delta)


    def xa(self, t: int) -> float:
        r"""
        Front of the fluid:
        
        .. math::
            x_A = \frac{1}{2}mt - 2 c_0 t

        Parameters
        ----------
        t : int
            Time instant.

        Returns
        -------
        float
            Position of the front edge of the fluid.
        """
        return 0.5*self._m*t - (2*self._c0*t)
    
    
    def xb(self, t: int) -> float:
        r"""
        Edge of the quiet region:
        
        .. math::
            x_B = \frac{1}{2}mt + c_0 t

        Parameters
        ----------
        t : int
            Time instant.

        Returns
        -------
        float
            Position of the edge of the quiet region.
        """
        return 0.5*self._m*t + (self._c0*t)


    def compute_h(self, 
                  x: int | np.ndarray, 
                  t: int):
        r"""Compute the flow height h(x, t) at given time and positions.
        The x-axis must be negative oriented.

        .. math::
                h(x, t) = 
                \begin{cases}
                    0 & \text{if } x \leq x_A(t), \\\\
                    \frac{1}{9g cos(\thêta)} \left( \frac{x}{t} + 2 c_0 - \frac{1}{2} m t \right)^2 & \text{if } x_A(t) < x \leq x_B(t), \\\\
                    h_0 & \text{if } x_B(t) < x,
                \end{cases}

        Parameters
        ----------
        x : int or np.ndarray
            Spatial positions.
        t : int
            Time instant.

        Notes
        -----
        Updates the internal '_h', '_x', '_t' attributes with the computed result, and reorients 
        '_h' and '_x' to positive axes 
        """
        if isinstance(x, int):
            x = [x]
        self._x = [-i for i in x[::-1]]
        self._t = t
        
        h = []
        for i in x:
            if i <= self.xa(t):
                h.append(0)
            elif self.xa(t) < i < self.xb(t):
                h.append( (1/(9*self._g*np.cos(self._theta))) * ( (i/t) + (2 * self._c0) - (0.5*t*self._m))**2 )
            else:
                h.append(self._h0)
        
        if all(v == 0 for v in h):
            h[-1] = self._h0
        
        self._h = h[::-1]


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
    A = Dam_break_friction_inclined(25, 20, 50)
    T = [i for i in range(0, 5)]
    x = np.linspace(-100, 0, 100)
    for t in T:
        A.compute_h(x, t)
        A.show_res(show_h=True)
    
    # B = Dam_break_wet_domain(5, 15, 1, 0.3)
    # T = [i for i in range(0, 5)]
    # x = np.linspace(0, 15, 100)
    # for t in T:
    #     B.compute_h(x, t)
    #     B.show_res(show_h=True)
    
