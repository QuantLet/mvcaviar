import numpy as np
import mvcaviar
import matplotlib.pyplot as plt


class GARCH2DSimulator:
    def __init__(self, pars, breaks):
        assert(len(pars) == len(breaks) + 1 )
        self.pars = pars
        self.breaks = breaks
        self.sigma0 = 1.

    def what_par(self, t):
        if len(self.breaks) == 0:
            return self.pars[0]
        if t >= self.breaks[-1]:
            return self.pars[-1]
        for (i, br) in enumerate(self.breaks):
            if br > t:
                return self.pars[i]

    def get_realisation(self, T):
        Y = np.random.randn(2, T)

        sigmas = np.empty((2, T))
        sigmas[:, 0] = (self.sigma0, self.sigma0)
        Y[:, 0] *= sigmas[:, 0]

        for t in range(1, T):
            par = self.what_par(t)
            (B, A) = mvcaviar.parse_par(2, 1, par)
            c = A[:, [0]]
            A = A[:, 1:3]
            x = np.matmul(B, sigmas[:, [t-1]]) + np.matmul(A, np.abs(Y[:, [t-1]]))
            #y = np.matmul(A, np.abs(Y[:, [t-1]]))
            #z = np.matmul(B, sigmas[:, t-1])
            sigmas[:, t] = np.reshape(c + x, (2,))
            Y[:, t] *= sigmas[:, t]

        return Y, sigmas


#theta0 = np.array([ 0.8, -0.3,
#                    0.0,  0.0,
#                    #
#                    .5, 0.05, 0.0,
#                    1.0, 0.0, 0.05])

theta0 = np.array([
    .5,  0.,
    0., .5,
    #
    0.5, .0, .2,
    .5, .2, .0
])

theta1 = np.array([
    -0.5, .0,
    .0, .5,
    #
    .5, .0, 0.2,
    .5, 0.2, .0
])

theta2 = np.array(theta0, copy=True)

sim = GARCH2DSimulator([theta0, theta1], [250])
Y, sigmas = sim.get_realisation(500)

psums = np.array(Y, copy=True)
for t in range(1, 500):
    psums[:, t] += psums[:, t-1]

fig, ax = plt.subplots()

ax.plot(list(range(500)), sigmas[0, :500], color='g', linewidth=.5)
ax.plot(list(range(500)), sigmas[1, :500], color='r', linewidth=.5)

plt.show()

np.savetxt("sim_ts.csv", Y, delimiter=",")
np.savetxt("sim_sigmas.csv", sigmas, delimiter=",")

thetas = []




