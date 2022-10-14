import numpy as np
from numpy import polynomial as poly


class Ressort():

    def __init__(self, k, x_e, extr_x, extr_v):
        # init with k the spring stiffness et x_e the spring lengths at the equilibrium positions
        # polynomial is a numpy class
        self.k = k
        self.x_e = np.asarray(x_e)
        self.extr_x = np.asarray(extr_x)
        self.extr_v = np.asarray(extr_v)
        self.coeff = poly.polynomial.polyfromroots(x_e)
        self.polyn_force = poly.polynomial.Polynomial(self.coeff)
        self.polyn_energy = self.polyn_force.integ()

    def force(self, x):
        # Returns the force associated with a spring of length x
        return -self.k*self.polyn_force(x)

    def find_ext_force(self):
        # Returns the two extremal forces of a spring
        deriv_poly = self.polyn_force.deriv()
        return deriv_poly.roots()


class EQD_mouvement():
#Can change the app point of the force
    def __init__(self, ressorts, masse, c_frot):
        # init with ressort a list of spring, masse a list of masses between the springs
        # c_frot a list of friction coefficients for every mass
        # X0 the initial conditions.
        self.ressorts = ressorts
        self.masse = np.asarray(masse)
        self.c_frot = np.asarray(c_frot)
        self.n = len(masse)
        self.X_sol = None
        self.t_sol = None

    def force_comp(self, x, force):
        # Compute the forces F of the springs for masses at position x
        dx = np.diff(np.append(0, x))
        F = np.asarray([self.ressorts[k].force(dx[k]) for k in range(len(dx))])
        DF = np.append(-np.diff(F), force + F[-1])
        return F, DF

    def EQM(self, t, X, force):
        x = X[:self.n]
        v = X[self.n:]
        _, DF = self.force_comp(x, force)
        dv = 1/self.masse*(DF-self.c_frot*v)
        return np.append(v, dv)