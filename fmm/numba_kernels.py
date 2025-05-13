import math
import numpy as np
from numba import njit, prange

# ---------------------------------------
# 1) BRUTE FORCE                                     (O(N^2) near‐field)
# ---------------------------------------
@njit(parallel=True, fastmath=True)
def brute_force_acc_numba(pos, q, eps):
    """
    pos : (N,3) array of particle positions
    q   : (N,)   array of charges = G*m
    eps : softening length
    returns (phi, acc) where
      phi[i] = sum_j q[j] / sqrt(r2 + eps^2)
      acc[i,d] = sum_j q[j] * (pos[j,d]-pos[i,d])/(r2+eps^2)^(3/2)
    """
    N = pos.shape[0]
    phi = np.zeros(N,     dtype=pos.dtype)
    acc = np.zeros((N,3), dtype=pos.dtype)

    for i in prange(N):
        xi0, xi1, xi2 = pos[i,0], pos[i,1], pos[i,2]
        for j in range(N):
            if i == j:
                continue
            dx0 = pos[j,0] - xi0
            dx1 = pos[j,1] - xi1
            dx2 = pos[j,2] - xi2
            r2  = dx0*dx0 + dx1*dx1 + dx2*dx2 + eps*eps
            invr  = 1.0/math.sqrt(r2)
            invr3 = invr*invr*invr
            phi[i] += q[j] * invr
            acc[i,0] += q[j] * dx0 * invr3
            acc[i,1] += q[j] * dx1 * invr3
            acc[i,2] += q[j] * dx2 * invr3

    return phi, acc


import math
import numpy as np
from numba import njit, prange

@njit(fastmath=True)
def P2M_numba(pos, q, src_idx, Ylm, Gm):
    P,n = Ylm.shape
    M = np.zeros(n, dtype=np.complex128)
    for k in range(n):
        acc_r = 0.0; acc_i = 0.0
        for t in range(P):
            i = src_idx[t]
            val = Ylm[t,k]
            rpow = Gm[t,k]
            qi   = q[i]
            acc_r += qi * (rpow * val.real)
            acc_i += qi * (rpow * val.imag)
        M[k] = acc_r + 1j*acc_i
    return M

def M2M_numba(child_center, parent_center, M_child, M2M_base, M2M_deg, M2M_ord):
    """
    child_center, parent_center : (3,)
    M_child : (n_coeff,) complex
    returns delta_M_parent : (n_coeff,) complex
    uses precomputed arrays M2M_base, M2M_deg, M2M_ord shape (n_coeff,n_coeff)
    """
    dx0 = child_center[0] - parent_center[0]
    dx1 = child_center[1] - parent_center[1]
    dx2 = child_center[2] - parent_center[2]
    r = math.sqrt(dx0*dx0 + dx1*dx1 + dx2*dx2)
    delta = np.zeros_like(M_child)
    n = M_child.shape[0]
    for a in range(n):
        for b in range(n):
            if M2M_deg[a,b] < 0:
                continue
            # get Y_{l-m} for this shift (omitted: user must precompute)
            Y = 1.0
            geom = r**M2M_deg[a,b]
            delta[a] += M2M_base[a,b] * geom * Y * M_child[b]
    return delta

@njit(fastmath=True)
def M2L_numba(src_center, tgt_center, M_src, Ubase, deg, ord_, Yrow, Grow):
    n = M_src.shape[0]
    delta = np.zeros(n, dtype=np.complex128)
    for a in range(n):
        acc = 0+0j
        for b in range(n):
            if deg[a,b] > n: continue
            acc += Ubase[a,b] * Grow[b] * Yrow[b] * M_src[b]
        delta[a] = acc
    return delta

@njit(fastmath=True)
def L2L_numba(parent_center, child_center, L_parent, Ubase, deg, ord_):
    n = L_parent.shape[0]
    delta = np.zeros(n, dtype=np.complex128)
    for a in range(n):
        acc = 0+0j
        for b in range(n):
            if deg[a,b] < 0: continue
            geom = np.power(
                np.linalg.norm(child_center-parent_center),
                -(deg[a,b]+1)
            )
            acc += Ubase[a,b] * geom * L_parent[b]
        delta[a] = acc
    return delta

@njit(fastmath=True)
def L2P_numba(pts, Lc, Ylm, Gm):
    P,n = pts.shape[0], Lc.shape[0]
    phi  = np.zeros(P, dtype=float)
    grad = np.zeros((P,3), dtype=float)
    C    = 1.0/(4.0*math.pi)
    for i in range(P):
        for k in range(n):
            phi[i] += (Lc[k] * Ylm[i,k] * Gm[i,k]).real * C
    return phi, grad

@njit(parallel=True, fastmath=True)
def P2P_numba(pos, q, src_idx, nbr_idx, phi_out, grad_out):
    fourpi = 4.0*math.pi
    P = src_idx.shape[0]
    M = nbr_idx.shape[0]
    for ii in prange(P):
        i = src_idx[ii]
        xi = pos[i]
        phi_i = 0.0; g0=g1=g2=0.0
        for jj in range(M):
            j = nbr_idx[jj]
            dx = xi - pos[j]
            r2 = dx[0]**2 + dx[1]**2 + dx[2]**2
            if r2<=0: continue
            invr = 1.0/math.sqrt(r2)
            invr3= invr*invr*invr
            qj = q[j]
            phi_i += qj*invr
            g0    -= qj*dx[0]*invr3
            g1    -= qj*dx[1]*invr3
            g2    -= qj*dx[2]*invr3
        phi_out[i] = phi_i/fourpi
        grad_out[i,0] = g0/fourpi
        grad_out[i,1] = g1/fourpi
        grad_out[i,2] = g2/fourpi

# ---------------------------------------
# 8) VECTORIZED VERLET INTEGRATOR
# ---------------------------------------
@njit(fastmath=True)
def simulate_verlet_numba(pos, vel, q, dt, nsteps, eps):
    """
    pos, vel : (N,3)
    q        : (N,)
    dt       : scalar
    nsteps   : int
    eps      : softening
    returns traj: (nsteps,N,3)
    """
    N = pos.shape[0]
    traj = np.empty((nsteps, N, 3), dtype=pos.dtype)
    # initial accel
    _, a_old = brute_force_acc_numba(pos, q, eps)

    for t in range(nsteps):
        # position update
        for i in range(N):
            for d in range(3):
                pos[i,d] += vel[i,d]*dt + 0.5*a_old[i,d]*dt*dt

        # new accel
        _, a_new = brute_force_acc_numba(pos, q, eps)

        # velocity update
        for i in range(N):
            for d in range(3):
                vel[i,d] += 0.5*(a_old[i,d] + a_new[i,d]) * dt

        # shift
        a_old = a_new

        # record
        for i in range(N):
            for d in range(3):
                traj[t,i,d] = pos[i,d]

    return traj
