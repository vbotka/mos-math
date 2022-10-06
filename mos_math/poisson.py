# All rights reserved (c) 2022, Vladimir Botka <vbotka@gmail.com>
# Simplified BSD License, https://opensource.org/licenses/BSD-2-Clause

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Electrophysical properties of the MOS structure with implanted bulk
# https://github.com/vbotka/thesis
# Appendix A: Numerical solution of the Poisson equation
#
# Notes:
#
# mode: 1 low frequency
#       2 high frequency
#       3 deep depletion
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import math
import numpy as np


def poisn(u1, uf, x, alfa, h, ipn, ulimt, ulimu, mode):

    u = np.zeros(shape=(ipn))
    v = np.zeros(shape=(ipn))
    f = np.zeros(shape=(ipn))
    e = np.zeros(shape=(ipn))
    w = np.zeros(shape=(ipn))
    z = np.zeros(shape=(ipn))
    ff = np.zeros(shape=(ipn))

    u[0] = u1
    if mode == 3:
        v[0] = math.sqrt(2.0 * (math.exp(u1) - u1 - 1))
    else:
        v[0] = math.sqrt(2.0 * (math.exp(u1) + math.exp(2.0 * uf - u1) - u1 - 1))
    f[0] = dv(x[0], u[0], v[0], uf, alfa[0], mode)

    w[0] = 0.01
    if mode == 1:
        z[0] = w[0] * (math.exp(u1) - math.exp(2.0 * uf - u1) - 1.0) / v[0]
    else:
        z[0] = w[0] * (math.exp(u1) - 1.0) / v[0]
    ff[0] = dz(x[0], w[0], u[0], uf, alfa[0], mode)

    for i in range(4):
        e[i + 1] = 0.0
        u[i + 1], v[i + 1], w[i + 1], z[i + 1], f[i + 1], ff[i + 1] = \
            RungeKutta2(x[i], u[i], v[i], w[i], z[i], f[i], ff[i], x[i + 1], h, uf, alfa, mode)
        us, istat = umarg(u[i + 1], ulimt, ulimu)
        if istat:
            return

    for i in range(4, ipn):
        PredictorCorrector2(x, u, v, w, z, f, ff, e, h, i, uf, alfa[i], mode)
        us, istat = umarg(u[i + 1], ulimt, ulimu)
        if istat:
            return
    #      us          dus         beta        gama
    return u[ipn - 1], v[ipn - 1], w[ipn - 1], z[ipn - 1], istat


def umarg(u, ulimt, ulimu):
    if u > 0.0:
        if u > ulimu:
            return u, True
    else:
        if(abs(u) > ulimt):
            return u, True
    return 0.0, False


def PredictorCorrector2(x, u, v, w, z, f, ff, e, h, i, uf, alfa, mode):
    ''' Predictor-Corrector 2nd order '''

    a = np.array([+2.29166666666667, -2.45833333333333, +1.54166666666667, -0.37500000000000])
    b = np.array([+0.37500000000000, +0.79166666666667, -0.20833333333333, +0.04166666666667])
    c = 0.07037037037037

# predictor
    pred_v = v[i] + h * sum([a[j] * f[i - j] for j in range(4)])
    pred_u = u[i] + h * sum([a[j] * v[i - j] for j in range(4)])
    pred_z = z[i] + h * sum([a[j] * ff[i - j] for j in range(4)])
    pred_w = w[i] + h * sum([a[j] * z[i - j] for j in range(4)])

# corrector
    s = b[0] * dv(x[i + 1], pred_u, pred_v, uf, alfa, mode)
    v[i + 1] = v[i] + h * (s + sum([b[j] * f[i + 1 - j] for j in range(1, 4)]))

    s = b[0] * du(x[i + 1], pred_u, v[i + 1], mode)
    u[i + 1] = u[i] + h * (s + sum([b[j] * v[i + 1 - j] for j in range(1, 4)]))
    f[i + 1] = dv(x[i + 1], u[i + 1], v[i + 1], uf, alfa, mode)

    s = b[0] * dz(x[i + 1], pred_z, u[i + 1], uf, alfa, mode)
    z[i + 1] = z[i] + h * (s + sum([b[j] * ff[i + 1 - j] for j in range(1, 4)]))

    s = b[0] * dw(x[i + 1], pred_w, z[i + 1], mode)
    w[i + 1] = w[i] + h * (s + sum([b[j] * z[i + 1 - j] for j in range(1, 4)]))
    ff[i + 1] = dz(x[i + 1], w[i + 1], u[i + 1], uf, alfa, mode)

# error estimation
    e[i + 1] = e[i] + c * (pred_u - u[i + 1])

    return


def RungeKutta2(x, u, v, w, z, f, ff, x1, h, uf, alfa, mode):
    ''' Runge-Kutta 2nd order '''

    r1 = du(x, u, v, mode)
    q1 = dv(x, u, v, uf, alfa, mode)
    o1 = dw(x, w, z, mode)
    p1 = dz(x, w, u, uf, alfa, mode)
    h2 = h / 2.0

    r2 = du(x + h2, u + h2 * r1, v + h2 * q1, mode)
    q2 = dv(x + h2, u + h2 * r1, v + h2 * q1, uf, alfa, mode)
    o2 = dw(x + h2, w + h2 * o1, z + h2 * p1, mode)
    p2 = dz(x + h2, w + h2 * o1, u + h2 * r1, uf, alfa, mode)

    r3 = du(x + h2, u + h2 * r2, v + h2 * q2, mode)
    q3 = dv(x + h2, u + h2 * r2, v + h2 * q2, uf, alfa, mode)
    o3 = dw(x + h2, w + h2 * o2, z + h2 * p2, mode)
    p3 = dz(x + h2, w + h2 * o2, u + h2 * r2, uf, alfa, mode)

    r4 = du(x + h, u + h * r3, v + h * q3, mode)
    q4 = dv(x + h, u + h * r3, v + h * q3, uf, alfa, mode)
    o4 = dw(x + h, w + h * o3, z + h * p3, mode)
    p4 = dz(x + h, w + h * o3, u + h * r3, uf, alfa, mode)

    u1 = u + h * (r1 + 2.0 * r2 + 2.0 * r3 + r4) / 6.0
    v1 = v + h * (q1 + 2.0 * q2 + 2.0 * q3 + q4) / 6.0
    w1 = w + h * (o1 + 2.0 * o2 + 2.0 * o3 + o4) / 6.0
    z1 = z + h * (p1 + 2.0 * p2 + 2.0 * p3 + p4) / 6.0

    f1 = dv(x1, u1, v1, uf, alfa, mode)
    ff1 = dz(x1, w1, u1, uf, alfa, mode)

    return u1, v1, w1, z1, f1, ff1


def du(x, u, v, mode):
    return v


def dv(x, u, v, uf, alfa, mode):
    if mode == 1 | mode == 2:
        return math.exp(u) - math.exp(2.0 * uf - u) + alfa - 1
    if mode == 3:
        return math.exp(u) + alfa - 1


def dw(x, w, z, mode):
    return z


def dz(x, w, u, uf, alfa, mode):
    if mode == 1:
        return w * (math.exp(u) + math.exp(2.0 * uf - u))
    if mode == 2 | mode == 3:
        return w * math.exp(u)
