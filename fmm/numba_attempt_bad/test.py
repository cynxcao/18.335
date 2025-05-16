import numpy as np

# --- simple Body container ---
class Body:
    def __init__(self, p, v, q, idx):
        # position, velocity are np arrays
        self.p = np.array(p, float)
        self.v = np.array(v, float)
        # q = G*m; keep it positive
        self.q = float(q)
        self.idx = idx

# --- sample N points uniformly inside a sphere of radius R ---
def sample_uniform_sphere(N, R, center=(0,0,0)):
    u    = np.random.rand(N)**(1/3)
    dirs = np.random.normal(size=(N,3))
    dirs /= np.linalg.norm(dirs, axis=1)[:,None]
    pts  = center + dirs * (u*R)[:,None]
    return pts

# --- build a “cold” spherical galaxy with mild velocity dispersion ---
def make_spherical_galaxy(center, idx_offset, N=25, R=10.0, mass_range=(1,5), G=1.0):
    ps = sample_uniform_sphere(N, R, center)
    ms = np.random.uniform(mass_range[0], mass_range[1], size=N)
    qs = G * ms
    # rough virial sigma^2 ≈ G*M/(5*R)
    # sigma = np.sqrt(G*ms.sum()/(5.0*R))
    # vs    = np.random.normal(scale=sigma, size=(N,3))
    vs = np.zeros((N, 3))
    bodies = []
    for i,(p,v,q) in enumerate(zip(ps,vs,qs)):
        bodies.append(Body(p, v, q, idx_offset+i))
    return bodies

# --- brute-force acceleration with softening eps ---
def brute_force_acc(bodies, eps=0.1):
    n   = len(bodies)
    acc = [np.zeros(3) for _ in bodies]
    for i, bi in enumerate(bodies):
        pi = bi.p
        for bj in bodies:
            if bi is bj: 
                continue
            rvec = bj.p - pi
            r2   = rvec.dot(rvec) + eps*eps
            acc[i] += bj.q * rvec / (r2**1.5)
    return np.vstack(acc)

# --- velocity-Verlet integrator ---
def simulate_brute(bodies, dt=0.001, nsteps=500, eps=0.1, debug=False):
    traj = []
    for step in range(nsteps):
        a_old = brute_force_acc(bodies, eps)
        # debug prints on first two bodies:
        if debug and len(bodies)>=2 and step%50==0:
            print(f"step {step}: p0={bodies[0].p}, a0={a_old[0]}")
            print(f"           p1={bodies[1].p}, a1={a_old[1]}")
        # update positions
        for i,b in enumerate(bodies):
            b.p += b.v*dt + 0.5*a_old[i]*dt*dt
        # new accel
        a_new = brute_force_acc(bodies, eps)
        # update velocities
        for i,b in enumerate(bodies):
            b.v += 0.5*(a_old[i] + a_new[i])*dt
        traj.append(np.vstack([b.p for b in bodies]))
    return np.stack(traj)  # shape (nsteps, n, 3)

if __name__ == "__main__":
    # make two clusters far apart
    gal1 = make_spherical_galaxy(center=[-50,0,0], idx_offset=0,  N=50, R=10, mass_range=(1,5), G=1.0)
    gal2 = make_spherical_galaxy(center=[+50,0,0], idx_offset=50, N=50, R=10, mass_range=(1,5), G=1.0)
    bodies = gal1 + gal2

    # simulate
    dt     = 0.001
    steps  = 500
    eps    = 0.5      # soften close encounters
    traj   = simulate_brute(bodies, dt=dt, nsteps=steps, eps=eps, debug=True)

    # now traj[t] is an array (100,3) of positions at time t
    # you can plug into your animation code directly.

