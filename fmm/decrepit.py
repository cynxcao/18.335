import numpy as np
from collections import deque
from scipy.special import sph_harm_y, factorial
from itertools import product

class Body:
    def __init__(self, p, v, q, idx):
        self.p = np.array(p, dtype=float)
        self.v = np.array(v, dtype=float)
        self.q = q
        self.idx = idx

class Node:
    def __init__(self, center, radius, level):
        self.center = np.array(center)
        self.radius = radius
        self.level = level
        self.children = []
        
        # indices
        self.src_idx = []

        self.P2P_list  = []
        self.M2L_list  = []

        # expansions:
        self.M_coeffs = None 
        self.L_coeffs = None 

class UniformFMM:
    def __init__(self, bodies, p, max_level, theta, dim=3):
        self.bodies = bodies
        self.p = p
        self.max_level = max_level
        self.theta = theta
        self.dim = dim
        self.n = (p + 1)**2

        points = np.vstack([b.p for b in bodies])
        p_max, p_min = points.min(0), points.max(0)

        center = (p_min + p_max)/2
        radius = max((p_min - p_max))/2 * 1.00001

        # build tree
        self.root = Node(center, radius, level=0)
        self.root.src_idx = list(range(len(bodies)))
        self.build_tree(self.root)

        for node in self.traverse_nodes():
            node.M_coeffs = np.zeros(self.n, dtype=complex)
            node.L_coeffs = np.zeros(self.n, dtype=complex)

        # Build interaction lists
        self.build_interactions()

        ### REEEE ####
        self.build_lm_arrays()
        # vectorized stuff
        L = self.l_arr[:, None]
        M = self.m_arr[:, None]
        K = self.l_arr[None, :]
        N = self.m_arr[None, :]

        # --- for M2L
        self.M2L_deg = L + K          # (nc,nc)
        self.M2L_ord = M + N

        # precompute the geometry‑independent part of the factorial
        num = factorial(self.M2L_deg, exact=True)
        denom = ( factorial(L-M, exact=True)
                * factorial(L+M, exact=True)
                * factorial(K-N, exact=True)
                * factorial(K+N, exact=True) )
        sign = (-1)**K
        self.M2L_base = sign * 4*np.pi * num / denom  

        # --- for M2M
        self.M2M_deg  = L - K                # (nc,nc)
        self.M2M_ord  = M - N                # (nc,nc)
        self.M2M_mask = (self.M2M_deg >= 0)  # only non‑negative shifts
        self.M2M_base = np.zeros_like(self.M2M_deg, float)
        self.M2M_base[self.M2M_mask] = np.sqrt(
            4*np.pi/(2*self.M2M_deg[self.M2M_mask] + 1)
        )

        # --- for L2L: deg = k − ℓ, order = n − m
        self.L2L_deg  = K - L
        self.L2L_ord  = N - M
        self.L2L_mask = (self.L2L_deg >= 0)
        self.L2L_base = np.zeros_like(self.L2L_deg, float)
        self.L2L_base[self.L2L_mask] = np.sqrt(
            4*np.pi/(2*self.L2L_deg[self.L2L_mask] + 1)
        )

    def build_tree(self, node):
        if node.level == self.max_level:
            return
        h = node.radius / 2

        # bucket bodies into 2**dim
        buckets = [[] for _ in range(2**self.dim)]
        for idx in node.src_idx:
            dx = self.bodies[idx].p - node.center
            b0 = int(dx[0] >= 0)
            b1 = int(dx[1] >= 0) 
            b2 = int(dx[2] >= 0)   
            buckets[b0 + 2*b1 + 4*b2].append(idx)

        # create children
        for octant, idx_list in enumerate(buckets):
            offset = np.array([
                h if (octant // (2**i)) % 2 else -h for i in range(3)
            ])
            child = Node(node.center + offset, h, node.level+1)
            child.src_idx = idx_list
            node.children.append(child)
            self.build_tree(child)

    def traverse_nodes(self):
        q = deque([self.root])
        while q:
            node = q.popleft()
            yield node
            for c in node.children:
                q.append(c)

    def build_interactions(self):
        nodes = list(self.traverse_nodes())
        for target in nodes:
            for src in nodes:
                if src is target:
                    continue
                d = np.linalg.norm(src.center - target.center)
                if (2*src.radius / d) < self.theta:
                    target.M2L_list.append(src)
                elif src.level == target.level:
                    target.P2P_list.append(src)

    def build_lm_arrays(self):
        # Flattened list of (l,m), and then two 1D arrays
        lm = [(l, m)
              for l in range(self.p + 1)
              for m in range(-l, l+1)]
        self.l_arr = np.array([l for l,m in lm], int)   # shape (ncoeff,)
        self.m_arr = np.array([m for l,m in lm], int)

    def get_sphere_coords(self, dx):
        r = np.linalg.norm(dx, axis=-1)
        # guard small r for arccos
        with np.errstate(invalid='ignore'):
            th = np.arccos(dx[...,2]/r)
        ph = np.arctan2(dx[...,1], dx[...,0])
        return r, th, ph

    def P2M(self, node):
        idx = node.src_idx
        if len(idx)==0:
            node.M_coeffs[:] = 0
            return

        # gather
        pts = np.vstack([self.bodies[i].p for i in idx])            # (Nsrc,3)
        qs  = np.array([self.bodies[i].q for i in idx])             # (Nsrc,)
        dxs = pts - node.center                                     # (Nsrc,3)
        r, theta, phi = self.get_sphere_coords(dxs)                     # each (Nsrc,)
        mask = r>0                                                  # skip zero
        r = r[mask]; theta=theta[mask]; phi=phi[mask]; qs=qs[mask]

        M = np.zeros(self.n, dtype=complex)
        # vectorized over sources but still loop over (l,m):
        for k,(l,m) in enumerate(zip(self.l_arr, self.m_arr)):
            Y = sph_harm_y(l, m, theta, phi)                             # (Nsrc,)
            M[k] = np.sum(qs * (r**l) * np.conj(Y))
        node.M_coeffs = M

    def M2M(self, node):
        for child in node.children:
            dx = child.center - node.center
            r, theta, phi = self.get_sphere_coords(dx)
            if r == 0:
                continue

            mask = self.M2M_mask
            geom = r ** self.M2M_deg[mask]
            Y = sph_harm_y(self.M2M_deg[mask], self.M2M_ord[mask], theta, phi)

            T = np.zeros((self.n, self.n), dtype=complex)
            T[mask] = self.M2M_base[mask] * geom * Y
            node.M_coeffs += T.dot(child.M_coeffs)

    def M2L(self, node):
        for target in node.M2L_list:
            dx = target.center - node.center
            r, theta, phi = self.get_sphere_coords(dx)
            if r == 0: 
                continue
            mask = (self.M2L_deg <= self.p)
            # power = -(self.M2L_deg[mask] + 1)
            # geom_factor = r**power

            Y = sph_harm_y(self.M2L_deg[mask], self.M2L_ord[mask], theta, phi)
            U = np.zeros((self.n, self.n), dtype=complex)
            U[mask] = self.M2L_base[mask] * Y # * geom_factor
            target.L_coeffs += U.dot(node.M_coeffs)

    def L2L(self, node):
        for child in node.children:
            dx = child.center - node.center
            r, theta, phi = self.get_sphere_coords(dx)
            if r == 0:
                continue

            mask = self.L2L_mask
            geom = r ** (-(self.L2L_deg[mask] + 1))

            Y = sph_harm_y(self.L2L_deg[mask], self.L2L_ord[mask], theta, phi) * q
            V = np.zeros((self.n, self.n), dtype=complex)
            V[mask] = self.L2L_base[mask] * geom * Y
            child.L_coeffs += V.dot(node.L_coeffs)

    def L2P(self, node):
        idx = node.src_idx
        if not idx:
            return {}, {}

        pts = np.vstack([self.bodies[i].p for i in idx])
        dxs = pts - node.center
        r, th, ph = self.get_sphere_coords(dxs)
        mask = r>0
        r = r[mask]; th=th[mask]; ph=ph[mask]
        valid_idx = np.array(idx)[mask]

        phi = {i:0.0 for i in idx}
        grad = {i:np.zeros(3, float) for i in idx}
        C = 1.0/(4*np.pi)

        # loop over coeffs but vector over targets
        for k,(l,m) in enumerate(zip(self.l_arr, self.m_arr)):
            coeff = node.L_coeffs[k]

            print(f"wawa, {l, m, th, ph}")
            Y, dY = sph_harm_y(l, m, th, ph, diff_n=1)
            print("REEEE")
            print(Y, dY)
            dY_dth = dY[0]
            dY_dph = 1j*m*Y

            r_l = r**l
            phi_vals = coeff * r_l * Y

            # common basis vectors
            er = dxs[mask]/r[:,None]
            et = np.vstack([
                np.cos(th)*np.cos(ph),
                np.cos(th)*np.sin(ph),
                -np.sin(th)
            ]).T
            ep = np.vstack([
                -np.sin(ph),
                 np.cos(ph),
                 0.0
            ]).T

            # partials
            df_dr = coeff * l * (r**(l-1)) * Y
            df_dt = coeff * r_l * dY_dth
            df_dp = coeff * r_l * dY_dph

            # build gradient term‐wise, guard sinθ
            sin_t = np.sin(th)
            tmp_grad = (df_dr[:,None]*er
                      + df_dt[:,None]*et/r[:,None]
                      + np.where(
                           sin_t[:,None]>0,
                           (df_dp[:,None]*ep)/(r[:,None]*sin_t[:,None]),
                           0.0
                        ))
            # scatter back
            for i,val in zip(valid_idx, phi_vals):
                phi[i]  += val.real
            for i,vec in zip(valid_idx, tmp_grad):
                grad[i] -= vec.real

        return phi, grad

    def P2P(self, node):
        C = 1.0/(4*np.pi)
        phi  = {}
        grad = {}
        for i in node.src_idx:
            p_i    = self.bodies[i].p
            phi_i  = 0.0
            grad_i = np.zeros(3, float)

            # sum over *all* sources in your P2P list
            for nbr in node.P2P_list:
                for j in nbr.src_idx:
                    p_j = self.bodies[j].p
                    d   = p_i - p_j
                    r2  = d.dot(d)
                    if r2 <= 0.0:
                        continue
                    invr  = 1.0/np.sqrt(r2)
                    invr3 = invr**3

                    qj = self.bodies[j].q
                    phi_i  += qj * invr
                    grad_i -= qj * d * invr3

            phi[i] = phi_i  * C
            grad[i] = grad_i * C

        return phi, grad

    def upward_pass(self, node):
        if node.children:
            for child in node.children:
                self.upward_pass(child)
            self.M2M(node)
        else:
            self.P2M(node)

    def downward_pass(self, node):
        self.M2L(node)
        self.L2L(node)
        for child in node.children:
            self.downward_pass(child)

    def solve(self):
        phi  = {i: 0.0 for i in range(len(self.bodies))}
        grad = {i: np.zeros(3, float) for i in range(len(self.bodies))}

        self.upward_pass(self.root)
        self.downward_pass(self.root)
     
        for node in self.traverse_nodes():
            if not node.children:
                phi_far, grad_far = self.L2P(node)
                phi_near, grad_near = self.P2P(node)

                for i in node.src_idx:
                    phi[i]  += phi_far[i]  + phi_near[i]
                    grad[i] += grad_far[i] + grad_near[i]

        return phi, grad
