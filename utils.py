import numpy as np
import scipy.stats as ss
from numpy.linalg import norm

def JSD(P,Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (ss.entropy(_P, _M) + ss.entropy(_Q, _M))
