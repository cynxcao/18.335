# fast_uniform_fmm.py
import numpy as np
from collections import deque
from scipy.special import sph_harm_y, factorial
from numba_kernels import (
    P2M_numba,
    M2M_numba,
    M2L_numba,
    L2L_numba,
    L2P_numba,
    P2P_numba
)
import math

class Body:
    def __init__(self, p, v, q, idx):
        self.p   = np.array(p, dtype=float)
        self.v   = np.array(v, dtype=float)
        self.q   = q
        self.idx = idx

class Node:
    def __init__(self, center, radius, level):
        self.center    = np.array(center, dtype=float)
        self.radius    = radius
        self.level     = level
        self.children  = []
        self.src_idx   = []
        self.P2P_list  = []
        self.M2L_list  = []
        self.M_coeffs  = None
        self.L_coeffs  = None
        # precomputed tables
        self.M2L_Y     = None
        self.M2L_geom  = None
        self.L2P_Y     = None
        self.L2P_geom  = None
        self.P2M_Y     = None
        self.P2M_geom  = None

class UniformFMM:
    def __init__(self, bodies, p, max_level, theta, dim=3):
        self.bodies    = bodies
        self.p         = p
        self.max_level = max_level
        self.theta     = theta
        self.dim       = dim

        # Build tree
        points    = np.vstack([b.p for b in bodies])
        p_min, p_max = points.min(0), points.max(0)
        center    = (p_min + p_max)/2
        radius    = np.max(p_max - p_min)/2*1.00001
        self.root = Node(center, radius, level=0)
        self.root.src_idx = list(range(len(bodies)))
        self.build_tree(self.root)

        # Build interactions
        self.build_interactions()

        # Build lm-arrays
        self.build_lm_arrays()
        self.n = len(self.l_arr)

        # Allocate M and L storage
        for node in self.traverse_nodes():
            node.M_coeffs = np.zeros(self.n, dtype=complex)
            node.L_coeffs = np.zeros(self.n, dtype=complex)

        # Precompute bases
        L = self.l_arr[:,None]; M = self.m_arr[:,None]
        K = self.l_arr[None,:]; N = self.m_arr[None,:]
        # M2L
        self.M2L_deg  = L + K
        self.M2L_ord  = M + N
        num   = factorial(self.M2L_deg, exact=True)

        denom1 = factorial(L - M, exact=True) * factorial(L + M, exact=True)  # (n,1)
        denom2 = factorial(K - N, exact=True) * factorial(K + N, exact=True)  # (1,n)
        denom = denom1 * denom2
        sign = (-1)**K    # shape (1,n) broadcasts to (n,n)
        self.M2L_base = sign * 4.0 * np.pi * num / denom
       
        # M2M
        self.M2M_deg  = L - K
        self.M2M_ord  = M - N
        self.M2M_mask = (self.M2M_deg>=0)
        self.M2M_base = np.zeros_like(self.M2M_deg, float)
        self.M2M_base[self.M2M_mask] = np.sqrt(
            4*np.pi/(2*self.M2M_deg[self.M2M_mask]+1)
        )
        # L2L
        self.L2L_deg  = K - L
        self.L2L_ord  = N - M
        self.L2L_mask = (self.L2L_deg>=0)
        self.L2L_base = np.zeros_like(self.L2L_deg, float)
        self.L2L_base[self.L2L_mask] = np.sqrt(
            4*np.pi/(2*self.L2L_deg[self.L2L_mask]+1)
        )

        # Precompute sph-harm tables
        self.precompute_M2L_Y()
        self.precompute_L2P_Y()
        self.precompute_P2M_Y()

    def build_tree(self, node):
        if node.level==self.max_level:
            return
        h = node.radius/2
        buckets = [[] for _ in range(2**self.dim)]
        for i in node.src_idx:
            dx  = self.bodies[i].p - node.center
            idx = int(dx[0]>=0) + 2*int(dx[1]>=0) + 4*int(dx[2]>=0)
            buckets[idx].append(i)
        for octant, lst in enumerate(buckets):
            child_center = node.center + h * np.array([
                1 if octant&(1<<d) else -1 for d in range(self.dim)
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
        for tgt in all_nodes:
            for src in all_nodes:
                if src is tgt: continue
                d= np.linalg.norm(tgt.center-src.center)
                if 2*src.radius/d < self.theta:
                    tgt.M2L_list.append(src)
                elif src.level==tgt.level:
                    tgt.P2P_list.append(src)

    def build_lm_arrays(self):
        lm = [(l,m) for l in range(self.p+1) for m in range(-l,l+1)]
        self.l_arr = np.array([l for l,m in lm], int)
        self.m_arr = np.array([m for l,m in lm], int)

    def get_sphere_coords(self, dx):
        # compute Euclidean norm along the last axis
        r = np.linalg.norm(dx, axis=-1)
        # replace any zero distances with 1.0 (avoids division by zero)
        r = np.where(r == 0.0, 1.0, r)

        # now safe to divide
        cos_t = np.clip(dx[..., 2] / r, -1.0, 1.0)
        th    = np.arccos(cos_t)
        ph    = np.arctan2(dx[..., 1], dx[..., 0])
        return r, th, ph


    def precompute_M2L_Y(self):
        """
        For each node we build
          node.M2L_Y    : shape (num_src, n_coeff) of Y_{l,m}(phi,theta)
          node.M2L_geom : shape (num_src, n_coeff) of r^{-(l+1)}
        """
        n = self.n
        for node in self.traverse_nodes():
            srcs = node.M2L_list
            M    = len(srcs)
            # init arrays
            node.M2L_Y    = np.zeros((M, n), dtype=complex)
            node.M2L_geom = np.zeros((M, n), dtype=float)

            for k, src in enumerate(srcs):
                dx = node.center - src.center
                r, th, ph = self.get_sphere_coords(dx)
                # now loop over each coefficient
                for ci, (l, m) in enumerate(zip(self.l_arr, self.m_arr)):
                    # sph_harm_y takes (m, l, phi, theta)
                    Ylm = sph_harm_y(m, l, ph, th) * math.sqrt(4.0*math.pi)
                    node.M2L_Y[k,   ci] = Ylm
                    node.M2L_geom[k,ci] = r**(-(l+1))


    def precompute_L2P_Y(self):
        """
        For each leaf (no children) we build
          leaf.L2P_Y    : shape (P, n_coeff) of Y_{l,m}(phi,theta)
          leaf.L2P_geom : shape (P, n_coeff) of r^l
        """
        n = self.n
        leaves = [node for node in self.traverse_nodes() if not node.children]
        for leaf in leaves:
            idx = leaf.src_idx
            P   = len(idx)
            if P == 0:
                # no targets → empty arrays
                leaf.L2P_Y    = np.zeros((0, n), dtype=complex)
                leaf.L2P_geom = np.zeros((0, n), dtype=float)
                continue

            # init
            leaf.L2P_Y    = np.zeros((P, n), dtype=complex)
            leaf.L2P_geom = np.zeros((P, n), dtype=float)

            # positions relative to leaf center
            pts = np.vstack([self.bodies[i].p for i in idx]) - leaf.center
            r, th, ph = self.get_sphere_coords(pts)

            # loop over each target p and each coeff ci
            for p_i in range(P):
                for ci, (l, m) in enumerate(zip(self.l_arr, self.m_arr)):
                    Ylm = sph_harm_y(m, l, ph[p_i], th[p_i]) * math.sqrt(4.0*math.pi)
                    leaf.L2P_Y   [p_i,ci] = Ylm
                    leaf.L2P_geom[p_i,ci] = r[p_i]**l
                    
    def precompute_P2M_Y(self):
        n = self.n
        # find the leaves
        leaves = [node for node in self.traverse_nodes() if not node.children]

        for leaf in leaves:
            idx = leaf.src_idx
            P   = len(idx)

            if P == 0:
                # empty leaf → zero‐shape tables
                leaf.P2M_Y    = np.zeros((0, n), dtype=complex)
                leaf.P2M_geom = np.zeros((0, n), dtype=float)
                continue

            # allocate
            leaf.P2M_Y    = np.zeros((P, n), dtype=complex)
            leaf.P2M_geom = np.zeros((P, n), dtype=float)

            # positions relative to leaf center
            pts = np.vstack([self.bodies[i].p for i in idx]) - leaf.center
            r, th, ph = self.get_sphere_coords(pts)

            # fill in per‐particle, per‐(l,m) tables
            for pi in range(P):
                for ci, (l, m) in enumerate(zip(self.l_arr, self.m_arr)):
                    # sph_harm_y(m,l,phi,theta) returns Y_l^m(phi,theta)
                    Ylm = sph_harm_y(m, l, ph[pi], th[pi]) * math.sqrt(4.0*math.pi)
                    # P2M uses conj(Y) * r^l
                    leaf.P2M_Y   [pi,ci] = np.conj(Ylm)
                    leaf.P2M_geom[pi,ci] = r[pi]**l

    def P2M(self,node):
        if not node.src_idx:
            node.M_coeffs.fill(0); return
        pos = np.vstack([b.p for b in self.bodies]); q=np.array([b.q for b in self.bodies])
        src_idx = np.array(node.src_idx, dtype=np.int64)
        M_arr = P2M_numba(pos, q, src_idx, node.P2M_Y, node.P2M_geom)
        node.M_coeffs = M_arr

    def M2M(self,node):
        for child in node.children:
            delta = M2M_numba(
                child.center, node.center,
                child.M_coeffs,
                self.M2M_base, self.M2M_deg, self.M2M_ord
            )
            node.M_coeffs += delta

    def M2L(self,node):
        for k,src in enumerate(node.M2L_list):
            delta = M2L_numba(
                src.center, node.center,
                src.M_coeffs,
                self.M2L_base, self.M2L_deg, self.M2L_ord,
                node.M2L_Y[k], node.M2L_geom[k]
            )
            node.L_coeffs += delta

    def L2L(self,node):
        for child in node.children:
            delta = L2L_numba(
                node.center, child.center,
                node.L_coeffs,
                self.L2L_base, self.L2L_deg, self.L2L_ord
            )
            child.L_coeffs += delta

    def L2P(self,node):
        idx = node.src_idx
        if not idx:
            return {}, {}
        # --- otherwise, do the vstack safely ---
        pts = np.vstack([self.bodies[i].p for i in idx]) - node.center
        phi, grad = L2P_numba(pts, node.L_coeffs, node.L2P_Y, node.L2P_geom)
        return dict(zip(idx, phi)), dict(zip(idx, grad))

    def P2P(self,node):
        pos = np.vstack([b.p for b in self.bodies]); q=np.array([b.q for b in self.bodies])
        src = np.array(node.src_idx, dtype=np.int64)
        nbr = np.concatenate([n.src_idx for n in node.P2P_list]).astype(np.int64)
        phi = np.zeros(pos.shape[0], dtype=float)
        grad= np.zeros((pos.shape[0],3), dtype=float)
        P2P_numba(pos,q,src,nbr,phi,grad)
        return {i:phi[i] for i in src}, {i:grad[i] for i in src}
    
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
        phi  = {i:0.0 for i in range(len(self.bodies))}
        grad = {i:np.zeros(3,float) for i in range(len(self.bodies))}
        self.upward_pass(self.root)
        self.downward_pass(self.root)
        for node in self.traverse_nodes():
            if not node.children:
                pf,gf = self.L2P(node)
                pn,gn = self.P2P(node)
                for i in node.src_idx:
                    phi[i]  += pf[i] + pn[i]
                    grad[i] += gf[i] + gn[i]
        return phi, grad
