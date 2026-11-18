import numpy as np


def ccd(R, f, g, N, tol=1e-12):
    o = slice(0, N)
    v = slice(N, None)
    e = np.diag(f)
    eo = e[o]
    ev = e[v]
    oovv = g[o, o, v, v]
    D = eo[:, None, None, None] + eo[None, :, None, None] - ev[None, None, :, None] - ev[None, None, None, :]

    t_o = oovv / D
    E_o = 0.25 * np.einsum("ijab, ijab->", oovv, t_o)

    for n_iter in range(1, 201):
        t_n = t_o + R(f, g, t_o) / D
        E_n = 0.25 * np.einsum("ijab, ijab->", oovv, t_n)

        if np.abs(E_n - E_o) < tol:
            return E_n, n_iter
        t_o = t_n
        E_o = E_n
    raise RuntimeError("ccd did not converge")
