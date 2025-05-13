import numpy as np
import math
from boxtree import TreeBuilder

class BarnesHut3D:
    def __init__(self, positions, masses, max_level=10, theta=0.5):
        self.positions = np.asarray(positions, float)  # (N,3)
        self.masses    = np.asarray(masses,   float)  # (N,)
        self.N         = self.positions.shape[0]
        self.theta     = float(theta)

        # build the octree
        tb = TreeBuilder(
            points    = self.positions,
            leaf_size = 1,
            max_level = max_level
        )
        self.tree = tb.get_tree()

        # arrays for node mass and center-of-mass
        n_nodes = self.tree.n_nodes
        self.node_mass = np.zeros(n_nodes, float)
        self.node_com  = np.zeros((n_nodes,3), float)

        self._compute_center_of_mass()


    def _compute_center_of_mass(self):
        # bottom-up traversal
        for node in range(self.tree.n_nodes-1, -1, -1):
            bodies = self.tree.idx_on_node[node]
            # start with bodies in this cell
            if bodies.size>0:
                mb   = self.masses[bodies]
                pb   = self.positions[bodies]
                msum = mb.sum()
                com  = (pb * mb[:,None]).sum(axis=0)
            else:
                msum = 0.0
                com  = np.zeros(3, float)

            # add children
            for child in self.tree.child_ids[node]:
                if child>=0:
                    msum += self.node_mass[child]
                    com  += self.node_com[child] * self.node_mass[child]

            if msum>0.0:
                self.node_mass[node] = msum
                self.node_com [node] = com / msum
            else:
                # empty cell → use geometric center
                self.node_mass[node] = 0.0
                self.node_com [node] = self.tree.cell_centers[node]


    def solve(self):
        """
        Same interface as brute_force and UniformFMM:
          returns (phi_dict, grad_dict)
        where grad = +nabla(phi).
        """
        phi_arr, force = self.compute_potential_and_force()
        # note: force = sum mass_j*(com_j - pos_i)/dist^3 = +F = -grad(phi)
        phi  = {i: phi_arr[i]     for i in range(self.N)}
        grad = {i: -force[i].copy() for i in range(self.N)}
        return phi, grad


    def compute_potential_and_force(self):
        """
        Returns:
          phi   : (N,)   array of potentials
          force : (N,3) array of force vectors = -grad(phi)
        """
        N = self.N
        phi   = np.zeros(N,   float)
        force = np.zeros((N,3), float)
        for i in range(N):
            p_i, f_i = self._pot_and_force_on_particle(i, 0)
            phi[i]   = p_i
            force[i] = f_i
        return phi, force


    def _pot_and_force_on_particle(self, idx, node):
        """
        Recursively compute (phi, F) on body idx from this node.
        F = sum mass_node * (com_node - pos_i)/dist^3
        phi = sum mass_node / dist
        """
        pos_i = self.positions[idx]
        com   = self.node_com[node]
        d     = com - pos_i
        dist  = np.linalg.norm(d)
        if dist <= 0.0:
            return 0.0, np.zeros(3, float)

        size = self.tree.radii[node]*2.0
        # opening criterion or leaf
        if size/dist < self.theta or self.tree.child_ids[node,0] < 0:
            m    = self.node_mass[node]
            invr  = 1.0/dist
            invr3 = invr*invr*invr
            phi_c = m * invr
            f_c   = m * d   * invr3
            return phi_c, f_c
        else:
            phi_sum = 0.0
            f_sum   = np.zeros(3, float)
            for child in self.tree.child_ids[node]:
                if child >= 0:
                    p_c, f_c = self._pot_and_force_on_particle(idx, child)
                    phi_sum += p_c
                    f_sum   += f_c
            return phi_sum, f_sum
