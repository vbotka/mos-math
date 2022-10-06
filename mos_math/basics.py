# All rights reserved (c) 2022, Vladimir Botka <vbotka@gmail.com>
# Simplified BSD License, https://opensource.org/licenses/BSD-2-Clause

from hamming_digital_filters import gradient as hdfg
import numpy

k = 1.38e-23
T = 300
q = 1.602e-19
ni = 1.45e16
beta = k * T / q


def clf_from_fi(fn, vg, fnorm, gtype, f_critical=0.1, no_filter_points=7):
    ''' Normalized LF C-V of MOS structure from the normalized surface potential '''
    f = [(1.0 - x) * fnorm for x in fn]
    dv = vg[1] - vg[0]
    if gtype == "hamming":
        dfdv = hdfg.gradient_nr(f, dv, f_critical, no_filter_points)
    else:
        dfdv = numpy.gradient(f, dv)
    return [(1.0 - x) for x in dfdv]


def _dit(clf, chf, cox):
    return ((clf * cox) / (1.0 - clf) - (chf * cox) / (1.0 - chf)) / q


def dit_from_clf_chf(clf, chf, cox):
    ''' Density of interface traps from LF and HF C-V '''
    return [_dit(clfi, chfi, cox) for clfi, chfi in zip(clf, chf)]


def eband_of_dit(fn, fnorm, bulk, stype):
    ''' Position of Dit in the forbiden band '''
    ff = beta * numpy.log(bulk / ni)
    if stype == 'P':
        ff = - ff
    return [((1.0 - f) * fnorm + 0.56 + ff) for f in fn]
