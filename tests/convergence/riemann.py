"""Exact Riemann solver for the 1D Euler equations (Toro, Ch. 4)."""
import numpy as np


def _fk(p, rho_k, p_k, c_k, g):
    """Pressure function and derivative for one side."""
    if p > p_k:                                   # shock
        A, B = 2.0 / ((g + 1.0) * rho_k), (g - 1.0) / (g + 1.0) * p_k
        f = (p - p_k) * np.sqrt(A / (B + p))
        df = np.sqrt(A / (B + p)) * (1.0 - 0.5 * (p - p_k) / (B + p))
    else:                                         # rarefaction
        f = 2.0 * c_k / (g - 1.0) * ((p / p_k) ** ((g - 1.0) / (2.0 * g)) - 1.0)
        df = 1.0 / (rho_k * c_k) * (p / p_k) ** (-(g + 1.0) / (2.0 * g))
    return f, df


def star_state(left, right, g):
    """(p_star, u_star) in the region between the two waves."""
    rho_l, u_l, p_l = left
    rho_r, u_r, p_r = right
    c_l, c_r = np.sqrt(g * p_l / rho_l), np.sqrt(g * p_r / rho_r)

    p = max(1e-12, 0.5 * (p_l + p_r) - 0.125 * (u_r - u_l) * (rho_l + rho_r) * (c_l + c_r))
    for _ in range(100):
        f_l, df_l = _fk(p, rho_l, p_l, c_l, g)
        f_r, df_r = _fk(p, rho_r, p_r, c_r, g)
        step = (f_l + f_r + u_r - u_l) / (df_l + df_r)
        p_new = max(1e-12, p - step)
        if abs(p_new - p) / (0.5 * (p_new + p)) < 1e-14:
            p = p_new
            break
        p = p_new
    f_l, _ = _fk(p, rho_l, p_l, c_l, g)
    f_r, _ = _fk(p, rho_r, p_r, c_r, g)
    return p, 0.5 * (u_l + u_r + f_r - f_l)


def sample(xi, left, right, g):
    """Density, velocity and pressure at self-similar coordinate xi = (x - x0) / t."""
    rho_l, u_l, p_l = left
    rho_r, u_r, p_r = right
    c_l, c_r = np.sqrt(g * p_l / rho_l), np.sqrt(g * p_r / rho_r)
    p_s, u_s = star_state(left, right, g)
    xi = np.asarray(xi, dtype=np.float64)
    rho = np.empty_like(xi); u = np.empty_like(xi); p = np.empty_like(xi)

    for i, s in enumerate(np.nditer(xi)):
        s = float(s)
        if s <= u_s:                                          # left of the contact
            if p_s > p_l:                                     # left shock
                sl = u_l - c_l * np.sqrt((g + 1.0) / (2.0 * g) * p_s / p_l + (g - 1.0) / (2.0 * g))
                if s <= sl:
                    rho[i], u[i], p[i] = rho_l, u_l, p_l
                else:
                    ratio = p_s / p_l
                    rho[i] = rho_l * (ratio + (g - 1.0) / (g + 1.0)) / ((g - 1.0) / (g + 1.0) * ratio + 1.0)
                    u[i], p[i] = u_s, p_s
            else:                                             # left rarefaction
                c_s = c_l * (p_s / p_l) ** ((g - 1.0) / (2.0 * g))
                if s <= u_l - c_l:
                    rho[i], u[i], p[i] = rho_l, u_l, p_l
                elif s >= u_s - c_s:
                    rho[i] = rho_l * (p_s / p_l) ** (1.0 / g)
                    u[i], p[i] = u_s, p_s
                else:                                         # inside the fan
                    u[i] = 2.0 / (g + 1.0) * (c_l + (g - 1.0) / 2.0 * u_l + s)
                    c = 2.0 / (g + 1.0) * (c_l + (g - 1.0) / 2.0 * (u_l - s))
                    rho[i] = rho_l * (c / c_l) ** (2.0 / (g - 1.0))
                    p[i] = p_l * (c / c_l) ** (2.0 * g / (g - 1.0))
        else:                                                 # right of the contact
            if p_s > p_r:                                     # right shock
                sr = u_r + c_r * np.sqrt((g + 1.0) / (2.0 * g) * p_s / p_r + (g - 1.0) / (2.0 * g))
                if s >= sr:
                    rho[i], u[i], p[i] = rho_r, u_r, p_r
                else:
                    ratio = p_s / p_r
                    rho[i] = rho_r * (ratio + (g - 1.0) / (g + 1.0)) / ((g - 1.0) / (g + 1.0) * ratio + 1.0)
                    u[i], p[i] = u_s, p_s
            else:                                             # right rarefaction
                c_s = c_r * (p_s / p_r) ** ((g - 1.0) / (2.0 * g))
                if s >= u_r + c_r:
                    rho[i], u[i], p[i] = rho_r, u_r, p_r
                elif s <= u_s + c_s:
                    rho[i] = rho_r * (p_s / p_r) ** (1.0 / g)
                    u[i], p[i] = u_s, p_s
                else:
                    u[i] = 2.0 / (g + 1.0) * (-c_r + (g - 1.0) / 2.0 * u_r + s)
                    c = 2.0 / (g + 1.0) * (c_r - (g - 1.0) / 2.0 * (u_r - s))
                    rho[i] = rho_r * (c / c_r) ** (2.0 / (g - 1.0))
                    p[i] = p_r * (c / c_r) ** (2.0 * g / (g - 1.0))
    return rho, u, p


def max_wave_speed(left, right, g):
    """Fastest signal in the solution, shocks included.

    A periodic box turns one discontinuity into two: the wrap-around jump launches its own
    Riemann fan inward from each edge. Cells within max_wave_speed * t of an edge are
    contaminated by it and have to be excluded before comparing against the single-jump
    exact solution.
    """
    rho_l, u_l, p_l = left
    rho_r, u_r, p_r = right
    c_l, c_r = np.sqrt(g * p_l / rho_l), np.sqrt(g * p_r / rho_r)
    p_s, _ = star_state(left, right, g)

    if p_s > p_l:   # left shock
        s_l = u_l - c_l * np.sqrt((g + 1.0) / (2.0 * g) * p_s / p_l + (g - 1.0) / (2.0 * g))
    else:           # head of the left rarefaction
        s_l = u_l - c_l
    if p_s > p_r:   # right shock
        s_r = u_r + c_r * np.sqrt((g + 1.0) / (2.0 * g) * p_s / p_r + (g - 1.0) / (2.0 * g))
    else:
        s_r = u_r + c_r
    return max(abs(s_l), abs(s_r))
