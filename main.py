#! /usr/bin/env python3

import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys

sns.set_style("darkgrid")


def sigmoid(t):
    t_trans = 10.0
    k = 10.0 / t_trans
    return np.reciprocal(1 + np.exp(-k * t))


def sigmoid_grad(t, t_on):
    fx = sigmoid(t - t_on)
    return fx * (1 - fx)


def SLS(t, g0, g1, tau):
    return g0 + g1 * np.exp(-t / tau)


class AnimatedConvolution:
    def __init__(self, repeat=True):
        self.repeat = repeat

        self.t = np.linspace(0.0, 1000.0, 2000)
        self.length = len(self.t)

        self.sls_t = SLS(self.t, 0.15, 0.5, 100.0)

        sls_t_padded = np.pad(
            self.sls_t, (self.length, self.length), "constant", constant_values=(0, 0)
        )
        self.sls_t_padded = np.flipud(sls_t_padded)  # we want convolution NOT correlation!

        self.e = sigmoid(self.t - 150.0) - sigmoid(self.t - 500.0)
        self.de_dt = sigmoid_grad(self.t, 150.0) - sigmoid_grad(self.t, 500.0)

        self.convolved = np.multiply(
            np.convolve(self.sls_t, self.de_dt, mode="full")[0 : self.length], np.gradient(self.t)
        )

        self.fig, self.axarr = plt.subplots(2, 2, figsize=(14.0, 9.0))

        self.multiplied_scaling = 2.0

        self.setup00(self.axarr[0, 0])
        self.setup01(self.axarr[0, 1])
        self.setup10(self.axarr[1, 0])
        self.setup11(self.axarr[1, 1])

    def show(self):
        plt.show()

    def setup00(self, ax):
        (self.e_line,) = ax.plot(self.t, self.e, "-", label="strain load", color="#566573")
        (self.de_line00,) = ax.plot(
            self.t, self.de_dt, "--", label="d(strain)/dt", color="#3385ff"
        )
        (self.dotfollow_line,) = ax.plot(self.t[0], self.e[0], "o", color="#dc7633")

        ax.legend(loc="lower left")
        ax.set_xlim(0, 1000)
        ax.set_ylim(-0.4, 1.2)

        self.lines00 = (self.e_line, self.de_line00, self.dotfollow_line)

        self.ani00 = anim.FuncAnimation(
            self.fig,
            self.animate00,
            np.arange(1, self.length - 1),
            interval=1,
            init_func=self.init00,
            blit=True,
            repeat=self.repeat,
        )

    def setup10(self, ax):
        ax.plot(self.t, self.sls_t, label="G_SLS(t)", color="#00b300")

        ax.legend(loc="lower left")
        ax.set_xlim(0, 1000)
        ax.set_ylim(0.0, 0.75)

    def setup01(self, ax):
        ax.set_xlim(0, 1000)
        ax.set_ylim(-0.55, 0.75)

        (de_line,) = ax.plot(self.t, self.de_dt, label="d(strain)/dt", color="#3385ff")
        (sls_line,) = ax.plot(
            self.t,
            self.sls_t_padded[(2 * self.length - 1) : (3 * self.length - 1)],
            "-.",
            label="0 padded G_SLS(-t)",
            color="#00b300",
        )

        self.lines01 = (de_line, sls_line)

        ax.legend(loc="lower left")

        self.ani01 = anim.FuncAnimation(
            self.fig,
            self.animate01,
            np.arange(1, self.length - 1),
            interval=1,
            init_func=self.init01,
            blit=True,
            repeat=self.repeat,
        )

    def setup11(self, ax):
        (convolved_line,) = ax.plot(
            self.t[0], self.convolved[0], "-.", label="convolution", color="#dc7633"
        )

        (multiplied_line,) = ax.plot(
            self.t[0],
            self.multiplied_scaling
            * np.multiply(self.de_dt, self.sls_t_padded[(2 * self.length - 1) : (3 * self.length - 1)])[0],
            "-",
            label="multiplication",
            color="#a64dff",
            alpha=0.7,
        )

        ax.legend(loc="lower left")
        ax.set_xlim(0, 1000)
        ax.set_ylim(-0.55, 0.75)

        self.lines11 = (convolved_line, multiplied_line)

        self.ani11 = anim.FuncAnimation(
            self.fig,
            self.animate11,
            np.arange(1, self.length - 1),
            interval=1,
            init_func=self.init11,
            blit=True,
            repeat=self.repeat,
        )

    def init00(self):
        self.lines00[0].set_data(self.t, self.e)
        self.lines00[1].set_data(self.t, self.de_dt)
        return self.lines00

    def animate00(self, i):
        self.lines00[2].set_data(self.t[i], self.e[i])
        return self.lines00

    def init01(self):
        self.lines01[0].set_data(self.t, self.de_dt)
        return self.lines01

    def animate01(self, i):
        self.lines01[1].set_ydata(self.sls_t_padded[(2 * self.length - i) : (3 * self.length - i)])
        return self.lines01

    def init11(self):
        return self.lines11

    def animate11(self, i):
        self.lines11[0].set_data(self.t[0:i], self.convolved[0:i])
        self.lines11[1].set_data(
            self.t,
            self.multiplied_scaling
            * np.multiply(self.de_dt, self.sls_t_padded[(2 * self.length - i) : (3 * self.length - i)]),
        )
        return self.lines11


def main():
    plot = AnimatedConvolution()
    plot.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
