import numpy as np
from uniform_fmm import Body, UniformFMM
# from fast_uniform_fmm import Body, UniformFMM
from barnes_hut_3d import BarnesHut3D
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from matplotlib.animation import PillowWriter
import matplotlib
matplotlib.use('MacOSX') 

import cProfile
np.random.seed(42)

def brute_force_soln(bodies, eps=2.0):
    phi  = {i: 0.0 for i in range(len(bodies))}
    grad = {i: np.zeros(3, float) for i in range(len(bodies))}

    for i, body in enumerate(bodies):
        p_i = body.p
        for j, other in enumerate(bodies):
            if j == i:
                continue
        
            p_j = other.p
            dx = p_i - p_j
            # print(f"dx of body {j} to body {i}: {dx}")
            
            r2 = dx.dot(dx) + eps**2
            r  = np.sqrt(r2)

            #Plummer softening reee
            phi[i] += other.q / (r)
            grad[i] -= other.q * dx / (r**3)

    return phi, grad

def get_error(brute_force_out, uniform_fmm_out):
    return

def sample_uniform_sphere(n, radius, center):
    dirs = np.random.normal(size=(n,3))
    dirs /= np.linalg.norm(dirs,axis=1)[:,None]
    r = np.random.rand(n)**(1/3) * radius
    return center + dirs * r[:,None]

def sample_uniform_sphere(N, R, center):
    # N points uniformly in a sphere of radius R
    u = np.random.rand(N)
    r = u**(1/3) * R
    dirs = np.random.normal(size=(N,3))
    dirs /= np.linalg.norm(dirs,axis=1)[:,None]
    return center + dirs * r[:,None]

def make_spherical_galaxy(center, idx_offset=0, N=25, R=5.0,
                          mass_range=(5.0,10.0), G=3000.0):
    ps = sample_uniform_sphere(N, R, np.array(center))
    ms = np.random.uniform(mass_range[0], mass_range[1], size=N)
    qs = G * ms
    vs = np.zeros((N,3))
    bodies = []
    for i,(p,v,q) in enumerate(zip(ps, vs, qs)):
        bodies.append(Body(p.tolist(), v.tolist(), q, idx_offset + i))
    return bodies

def simulate(bodies, P, max_level, theta, dt, nsteps, algo="uniform-fmm"):
    """
    bodies: list of Body objects with .p, .v, .q
    dt: time step
    nsteps: number of steps
    """
    all_bodies = []

    # if algo == "uniform-fmm":
    #     fmm = UniformFMM(bodies, P, max_level, theta)
    # elif algo=="barnes":
    #     pos = np.array([b.p for b in bodies])
    #     m = np.array([b.q for b in bodies])
    #     solver = BarnesHut3D(pos, m, max_level=max_level, theta=theta)

    for step in range(nsteps):
        # print("hi", step)
        # merge_close_pairs(bodies, merge_dist=0.5)
        # a_old = brute_force_acc(bodies, eps=0.1)
        if algo == "uniform-fmm":
            fmm = UniformFMM(bodies, P, max_level, theta)
            _, a_old = fmm.solve()
        elif algo == "brute":
            _, a_old = brute_force_soln(bodies)
        # elif algo == "barnes":
        #     _, a_old = solver.solve()
        # else:
        #     raise NotImplementedError(algo)

        # for i,b in enumerate(bodies):
        #     b.v += a_old[i] * dt

        # # 3) “drift” positions
        # for b in bodies:
        #     b.p += b.v * dt
        # a_old = np.vstack([ grad_o[i] for i in range(len(bodies)) ])
        # debug print
        print(f"\nstep {step:2d}")
        # for i,b in enumerate(bodies):
        #     print(f"  body {i}: p={b.p}, v={b.v}, a_old={a_old[i]}")
        # a_old -= np.mean(a_old, axis=0)

        for i, b in enumerate(bodies):
            b.p += b.v*dt + 0.5*a_old[i]*(dt**2)

        # p_com = np.mean([b.p for b in bodies], axis=0)
        # for b in bodies:
        #     b.p += (prev_p_com - p_com)
        # prev_p_com = p_com

        if algo == "uniform-fmm":
            # fmm = UniformFMM(bodies, P, max_level, theta)
            _, a_new = fmm.solve()
        elif algo == "brute":
            # _, a_new = brute_force_soln(bodies)
            a_new = a_old
        # elif algo == "barnes":
        #     _, a_new = solver.solve()

        # a_new = np.vstack([ grad_n[i] for i in range(len(bodies)) ])
        # a_new -= np.mean(a_new, axis=0)
        for i, b in enumerate(bodies):
            b.v += 0.5*(a_old[i] + a_new[i])*dt

        # Vcom = np.mean([b.v for b in bodies], axis=0)
        # for b in bodies:
        #     b.v -= Vcom

        all_bodies.append(np.stack([b.p.copy() for b in bodies], axis=0))

    return np.stack(all_bodies, axis=0)  # shape (nsteps, nbodies, 3)

def make_disk_galaxy(center, idx_offset, N=100,
                     radius=50.0, thickness=0.5,
                     mass_range=(5.0,5.0), G=300.0):
    """
    Generate a rotating disk galaxy:
      center:      [x,y,z]
      idx_offset:  integer start index
      N:           number of stars
      radius:      disk radius
      thickness:   Gaussian z-dispersion
      mass_range:  (min_mass, max_mass)
      G:           gravitational constant
    Returns list of Body objects with .p,.v,.q,.idx
    """
    # sample disk in xy
    u = np.random.rand(N)
    r = np.sqrt(u)*radius
    th = np.random.rand(N)*2*np.pi
    xs = r * np.cos(th)
    ys = r * np.sin(th)
    zs = np.random.normal(scale=thickness, size=N)
    pts = np.vstack([xs, ys, zs]).T + np.array(center)

    # assign masses and charges
    masses = np.random.uniform(mass_range[0],
                               mass_range[1],
                               size=N)
    qs = G * masses

    # compute enclosed mass for each star
    Menc = np.zeros(N)
    for i in range(N):
        ri = r[i]
        Menc[i] = masses[r <= ri].sum()

    # tangential velocities for circular orbits
    vs = np.zeros((N,3))
    for i in range(N):
        if r[i] > 0:
            speed = np.sqrt(G*Menc[i]/r[i])
            # velocity perpendicular to radius vector
            vs[i,0] = -speed * np.sin(th[i])
            vs[i,1] =  speed * np.cos(th[i])
            # vs[i,2] stays zero

    bodies = []
    for i in range(N):
        idx = idx_offset + i
        p   = pts[i].tolist()
        v   = vs[i].tolist()
        qv  = qs[i]
        bodies.append(Body(p, v, qv, idx))
    return bodies


def generate_two_disk_collision(separation=100.0,
                                relative_speed=0.0,
                                **disk_kwargs):
    """
    Build two disk galaxies on a collision course.
    separation:     center-to-center distance along x
    relative_speed: initial speed of approach (each gets half)
    disk_kwargs:    passed to make_disk_galaxy
    Returns list of Body objects for both galaxies.
    """
    # first galaxy at x = -sep/2, moving +x
    gal1 = make_disk_galaxy(
        center=[-separation/2, 0.0, 0.0],
        idx_offset=0,
        **disk_kwargs
    )
    # second at x = +sep/2, moving -x
    gal2 = make_disk_galaxy(
        center=[+separation/2, 0.0, 0.0],
        idx_offset=len(gal1),
        **disk_kwargs
    )

    # add approach velocities
    v_half = relative_speed/2.0
    for b in gal1:
        b.v[0] +=  v_half
    for b in gal2:
        b.v[0] += -v_half

    return gal1 + gal2

def update(frame, positions, scat):
    # grab the 25 points at this timestep
    xs = positions[frame, :, 0]
    ys = positions[frame, :, 1]
    zs = positions[frame, :, 2]

    scat._offsets3d = (xs, ys, zs)
    return scat
   
def visualize_sim(positions):
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
  
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(np.min(positions[:,:,0]), np.max(positions[:,:,0]))
    ax.set_ylim(np.min(positions[:,:,1]), np.max(positions[:,:,1]))
    ax.set_zlim(np.min(positions[:,:,2]), np.max(positions[:,:,2]))

    xs = positions[0, :, 0]
    ys = positions[0, :, 1]
    zs = positions[0, :, 2]
    scat = ax.scatter(xs, ys, zs, c='blue', marker='o')
   
    ani = animation.FuncAnimation(fig, update, positions.shape[0], fargs=(positions, scat), interval=30, blit=False)

    ani.save('fmm-ree.gif', writer=PillowWriter(fps=10))
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_bodies", type=int)
    parser.add_argument("--exp_order", type=int)
    parser.add_argument("--max_level", type=int)
    parser.add_argument("--theta_thresh", type=float)
    parser.add_argument("--num_steps", type=int)
    parser.add_argument("--dt", type=float)
    args = parser.parse_args()

    # Example: two galaxies side by side
    all_bodies = []
    n = 10
    for i in range(n):
        p = np.array([i, i, i], float)
        v = np.array([i, i, i],       float)  # or whatever you want per‐body
        q = 1.0
        idx = i
        all_bodies.append(Body(p, v, q, idx))

    print([b.v for b in all_bodies])
    # gal1 = make_galaxy(center=[0.0,0.0,0.0], idx_offset=0,  N=args.num_bodies, R=10.0, thickness=10.0)
    # gal1 = make_spherical_galaxy(center=[-50,0,0], idx_offset=0,  N=100, R=50.0)
    # gal2 = make_spherical_galaxy(center=[50, 0, 0], idx_offset=100,  N=100, R=50.0)
    # all_bodies = gal1 + gal2
    # all_bodies = generate_two_disk_collision()

    # reee
    # gal1 = make_spherical_galaxy(center=[-100,0,0], idx_offset=0,
    #                              N=500, R=100.0)
    # gal2 = make_spherical_galaxy(center=[0,100,-100], idx_offset=500,
    #                              N=500, R=100.0)
    # all_bodies = gal1 + gal2

    # gal1 = make_spherical_galaxy(center=[-50,0,0], idx_offset=0,
    #                              N=100, R=50.0)
    # gal2 = make_spherical_galaxy(center=[50,0,0], idx_offset=100,
    #                              N=100, R=50.0)
    # all_bodies = gal1 + gal2

    # gal1 = make_spherical_galaxy(center=[-100,0,0], idx_offset=0,
    #                              N=500, R=100.0)
    # gal2 = make_spherical_galaxy(center=[100,0,0], idx_offset=500,
    #                              N=500, R=100.0)
    # all_bodies = gal1 + gal2

    # all_bodies = [Body(p = [1, 0, 0], v = [0, 0, 0], q = 1, idx = 0), 
    #               Body(p = [-1, 0, 0], v = [0, 0, 0], q = 1, idx = 1)]


    # simulate with bigger timesteps and more steps
    positions_over_time = simulate(
        bodies     = all_bodies,
        P          = 6,
        max_level  = 2,
        theta      = 0.5,
        dt         = 1.0,    # 10× larger
        nsteps     = 10,    # enough steps to see both effects
        algo       = "uniform-fmm"
    )

    # print(positions_over_time)

    np.save("fmm-reee.npy", positions_over_time)

    visualize_sim(positions_over_time)
    

if __name__ == "__main__":
    main()