import numpy as np
from collections import deque
from scipy.special import sph_harm_y, factorial

class Body:
    def __init__(self, p, v, q, idx):
        self.p = np.array(p, dtype=float)
        self.v = np.array(v, dtype=float)
        self.q = q
        self.idx = idx

class Node:
    def __init__(self, center, radius, level):
        self.center = np.array(center, dtype=float)
        self.radius = radius
        self.level = level
        self.children = []
        self.src_idx = []
        self.P2P_list = []
        self.M2L_list = []
        self.M_coeffs = None
        self.L_coeffs = None

class UniformFMM:
    def __init__(self, bodies, p, max_level, theta, dim=3):
        self.bodies = bodies
        self.p = p
        self.max_level = max_level
        self.theta = theta
        self.dim = dim

        # Build tree
        points = np.vstack([b.p for b in bodies])
        p_min, p_max = points.min(0), points.max(0)
        center = (p_min + p_max) / 2
        radius = np.max(p_max - p_min) / 2 * 1.00001

        self.root = Node(center, radius, level=0)
        self.root.src_idx = list(range(len(bodies)))
        self.build_tree(self.root)

        # Build interactions
        self.build_interactions()

        # Build lm-arrays and set coefficient length
        self.build_lm_arrays()
        self.n = len(self.l_arr)

        # Allocate multipole/local storage
        for node in self.traverse_nodes():
            node.M_coeffs = np.zeros(self.n, dtype=complex)
            node.L_coeffs = np.zeros(self.n, dtype=complex)

        # Vectorized precomputations
        L = self.l_arr[:, None]
        M = self.m_arr[:, None]
        K = self.l_arr[None, :]
        N = self.m_arr[None, :]

        # M2L precompute
        self.M2L_deg = L + K
        self.M2L_ord = M + N
        num = factorial(self.M2L_deg, exact=True)
        denom = (factorial(L - M, exact=True)
                 * factorial(L + M, exact=True)
                 * factorial(K - N, exact=True)
                 * factorial(K + N, exact=True))
        sign = (-1)**K
        self.M2L_base = sign * 4 * np.pi * num / denom

        # M2M precompute
        self.M2M_deg = L - K
        self.M2M_ord = M - N
        self.M2M_mask = (self.M2M_deg >= 0)
        self.M2M_base = np.zeros_like(self.M2M_deg, float)
        self.M2M_base[self.M2M_mask] = np.sqrt(
            4 * np.pi / (2 * self.M2M_deg[self.M2M_mask] + 1)
        )

        # L2L precompute
        self.L2L_deg = K - L
        self.L2L_ord = N - M
        self.L2L_mask = (self.L2L_deg >= 0)
        self.L2L_base = np.zeros_like(self.L2L_deg, float)
        self.L2L_base[self.L2L_mask] = np.sqrt(
            4 * np.pi / (2 * self.L2L_deg[self.L2L_mask] + 1)
        )

    def build_tree(self, node):
        if node.level == self.max_level:
            return
        
        # divide into octants
        h = node.radius / 2
        buckets = [[] for _ in range(2**self.dim)]
        for i in node.src_idx:
            dx = self.bodies[i].p - node.center
            idx = (int(dx[0]>=0) + 2*int(dx[1]>=0) + 4*int(dx[2]>=0))
            buckets[idx].append(i)

        # populate children
        for octant, lst in enumerate(buckets):
            child_center = node.center + h * np.array([
                1 if octant & (1<<d) else -1
                for d in range(self.dim)
            ], dtype=float)
            child = Node(child_center, h, node.level+1)
            child.src_idx = lst
            node.children.append(child)
            self.build_tree(child)

    def traverse_nodes(self):
        q = deque([self.root])
        while q:
            n = q.popleft()
            yield n
            for c in n.children:
                q.append(c)

    def build_interactions(self):
        all_nodes = list(self.traverse_nodes())

        # iterate through all targe/tsource pairs
        for tgt in all_nodes:
            for src in all_nodes:
                if src is tgt:
                    continue
                # get distance
                d = np.linalg.norm(tgt.center - src.center)

                # divide in M2L, P2P
                if (2*src.radius/d) < self.theta:
                    tgt.M2L_list.append(src)
                elif src.level == tgt.level:
                    tgt.P2P_list.append(src)

    def build_lm_arrays(self):
        # build lm arrays for vectorized spherical harmonics
        lm = [(l,m) for l in range(self.p+1) for m in range(-l,l+1)]
        self.l_arr = np.array([l for l,m in lm], int)
        self.m_arr = np.array([m for l,m in lm], int)

    def get_sphere_coords(self, dx):
        r = np.linalg.norm(dx, axis=-1)
        small = (r==0)
        r = np.where(small, 1.0, r)

        cos_t = np.clip(dx[...,2]/r, -1.0, 1.0)
        th = np.arccos(cos_t)
        ph = np.arctan2(dx[...,1], dx[...,0])

        return r, th, ph

    def P2M(self, node):
        idx = node.src_idx
        if not idx:
            node.M_coeffs.fill(0)
            return
        
        # get all points + charges
        pts = np.vstack([self.bodies[i].p for i in idx])
        qs  = np.array([self.bodies[i].q for i in idx])

        # get distance form node center in w/ sphericalc oordinates
        dxs = pts - node.center
        r, th, ph = self.get_sphere_coords(dxs)
        mask = (r>0)
        r, th, ph, qs = r[mask], th[mask], ph[mask], qs[mask]

        # fill in M array
        M = np.zeros(self.n, dtype=complex)
        for k,(l,m) in enumerate(zip(self.l_arr, self.m_arr)):
            Y = sph_harm_y(l, m, th, ph) * np.sqrt(4 * np.pi)
            M[k] = np.sum(qs * (r**l) * np.conj(Y))
        node.M_coeffs = M

    def M2M(self, node):
        # iterate through cildren
        for child in node.children:
            dx = child.center - node.center
            r, th, ph = self.get_sphere_coords(dx)
            if r<=0: # check not same point
                continue

            mask = self.M2M_mask # make mask
            if not np.any(mask): 
                continue
            
            # see before
            geom = r**self.M2M_deg[mask]
            Y = sph_harm_y(self.M2M_deg[mask], self.M2M_ord[mask], th, ph) * np.sqrt(4 * np.pi)
            T = np.zeros((self.n,self.n),dtype=complex)
            T[mask] = self.M2M_base[mask]*geom*Y
            node.M_coeffs += T.dot(child.M_coeffs)

    def M2L(self, node):
        # iterate through m2l list to get itneractions
        for src in node.M2L_list:
            dx = node.center - src.center
            r, th, ph = self.get_sphere_coords(dx)
            if r<=0: # check not same point
                continue

            mask = (self.M2L_deg <= self.p)
            if not np.any(mask): # make mask
                continue
            
            # spherical harmonics bc 3d
            power = -(self.M2L_deg[mask] + 1) 
            geom = r**power
            Y = sph_harm_y(self.M2L_deg[mask], self.M2L_ord[mask], th, ph) * np.sqrt(4 * np.pi)
            U = np.zeros((self.n,self.n),dtype=complex)
            U[mask] = self.M2L_base[mask]*geom*Y
            node.L_coeffs += U.dot(src.M_coeffs)

    def L2L(self, node):
        # get local interactions, iterate through children
        for child in node.children:
            dx = child.center - node.center
            r, th, ph = self.get_sphere_coords(dx)
            if r<=0: # check not same point
                continue

            mask = self.L2L_mask
            if not np.any(mask): 
                continue
            
            # spherical harmonics bc 3d
            geom = r**(-(self.L2L_deg[mask] + 1))
            Y = sph_harm_y(self.L2L_deg[mask], self.L2L_ord[mask], th, ph) * np.sqrt(4 * np.pi)
            V = np.zeros((self.n,self.n),dtype=complex)
            V[mask] = self.L2L_base[mask]*geom*Y
            child.L_coeffs += V.dot(node.L_coeffs)

    def L2P(self, node):
        idx = node.src_idx
        phi = {i:0.0 for i in idx}
        grad = {i:np.zeros(3,float) for i in idx}
        if not idx: 
            return phi, grad
        
        pts = np.vstack([self.bodies[i].p for i in idx])
        dxs = pts - node.center
        r, th, ph = self.get_sphere_coords(dxs)
        valid = (r>0) # mask
        valid_idx = np.array(idx)[valid]
        r, th, ph, dxs = r[valid], th[valid], ph[valid], dxs[valid]

        for k,(l,m) in enumerate(zip(self.l_arr, self.m_arr)):
            coeff = node.L_coeffs[k]
            Y, dY = sph_harm_y(l, m, th, ph, diff_n=1)

            Y *= np.sqrt(4 * np.pi)

            # calculate gradients for time-stepping in simulations
            dY[:, 0] *= np.sqrt(4 * np.pi)

            # spherical harmonic gradients
            phi_vals = coeff * r**l * Y
            df_dr = coeff*l*(r**(l-1))*Y
            df_dt = coeff*r**l*dY[:, 0]

            df_dp = 1j*m*coeff*r**l*Y
            er = dxs/r[:,None]
            et = np.vstack([np.cos(th)*np.cos(ph), np.cos(th)*np.sin(ph), -np.sin(th)]).T
            ep = np.vstack([-np.sin(ph), np.cos(ph), np.zeros_like(ph)]).T
            sin_t = np.sin(th)
            grad_vecs = ( df_dr[:,None]*er
                        + df_dt[:,None]*et/r[:,None]
                        + np.where(sin_t[:,None]>0,
                                   df_dp[:,None]*ep/(r[:,None]*sin_t[:,None]),
                                   0.0))
            for ii,i in enumerate(valid_idx):
                phi[i]  += (phi_vals[ii].real)
                grad[i] -= (grad_vecs[ii].real)
        return phi, grad

    def P2P(self, node, eps=2.0):
        # equal to brute force for each node
        phi  = {}
        grad = {}
        for i in node.src_idx:

            p_i = self.bodies[i].p
            phi_i = 0.0
            grad_i = np.zeros(3,float)

            for nbr in node.P2P_list:
                for j in nbr.src_idx:
                    d = p_i - self.bodies[j].p
                    r2 = d.dot(d) + eps**2
                    if r2<=0:# check not same point
                        continue

                    invr = 1.0/np.sqrt(r2)
                    invr3= invr**3
                    qj = self.bodies[j].q
                    phi_i  += qj*invr

                    # more gradient computation
                    grad_i -= qj*d*invr3
            phi[i]  = phi_i #/(4*np.pi)
            grad[i] = grad_i #/(4*np.pi)
        return phi, grad

    def upward_pass(self, node):
        if node.children:
            for c in node.children:
                self.upward_pass(c)
            self.M2M(node)
        else:
            self.P2M(node)

    def downward_pass(self, node):
        self.M2L(node)
        self.L2L(node)
        for c in node.children:
            self.downward_pass(c)

    def solve(self):
        # init potentials and gradients
        phi  = {i:0.0 for i in range(len(self.bodies))}
        grad = {i:np.zeros(3,float) for i in range(len(self.bodies))}

        # perform FMM passes
        self.upward_pass(self.root)
        self.downward_pass(self.root)

        # sum up across each node 
        for node in self.traverse_nodes():
            if not node.children:
                phi_far,grad_far = self.L2P(node)
                phi_near,grad_near = self.P2P(node)
                for i in node.src_idx:
                    phi[i]  += phi_far[i]  + phi_near[i]
                    grad[i] += grad_far[i] + grad_near[i]
        return phi, grad
