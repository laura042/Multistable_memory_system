import numpy as np
from EQD import EQD_mouvement

import theano.tensor as T
import sys
from ilqr.dynamics import constrain, BatchAutoDiffDynamics, tensor_constrain, FiniteDiffDynamics



class EnvDynamics():

    def __init__(self,
                 ressorts,
                 masse,
                 c_frot,
                 dt,
                 **kwargs):
        self.n = len(masse)
        self.dt = dt

        self.eqd = EQD_mouvement(ressorts=ressorts,
                            masse=masse,
                            c_frot=c_frot)

        self.state_size = len(masse)*2
        self.action_size = 1

        self.dynamics = FiniteDiffDynamics(self.f, self.state_size, self.action_size)

    def f(self, x, u, i):
        x_ = []
        for idx in range(self.n*2):
            x_.append(x[..., idx])
        F = u[..., 0]

        x_dot = self.eqd.EQM(i, x_, F)
        return np.array([x_[k] + x_dot[k] * self.dt for k in range(self.n*2)])
