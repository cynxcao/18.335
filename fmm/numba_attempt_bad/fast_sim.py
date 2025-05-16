import numpy as np
from uniform_fmm import Body
from numba_kernels import brute_force_acc_numba, simulate_verlet_numba
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from matplotlib.animation import PillowWriter

np.random.seed(42)

# sample points uniformly inside a sphere of radius R

def sample_uniform_sphere(n, radius, center):
    dirs = np.random.normal(size=(n,3))
    dirs /= np.linalg.norm(dirs,axis=1)[:,None]
    r = np.random.rand(n)**(1/3) * radius
    return center + dirs * r[:,None]

# build a cold spherical galaxy at rest

def make_spherical_galaxy(center, idx_offset=0, N=25, R=5.0,
                          mass_range=(5.0,10.0), G=30.0):
    ps = sample_uniform_sphere(N, R, np.array(center))
    ms = np.random.uniform(mass_range[0], mass_range[1], size=N)
    qs = G * ms
    vs = np.zeros((N,3))
    bodies = []
    for i,(p,v,q) in enumerate(zip(ps, vs, qs)):
        bodies.append(Body(p.tolist(), v.tolist(), q, idx_offset + i))
    return bodies

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

# update function for animation

def update(frame, traj, scat):
    xs = traj[frame,:,0]
    ys = traj[frame,:,1]
    zs = traj[frame,:,2]
    scat._offsets3d = (xs, ys, zs)
    return scat

# visualize in 3D and save gif

def visualize_sim(traj):
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(np.min(traj[:,:,0]), np.max(traj[:,:,0]))
    ax.set_ylim(np.min(traj[:,:,1]), np.max(traj[:,:,1]))
    ax.set_zlim(np.min(traj[:,:,2]), np.max(traj[:,:,2]))

    xs = traj[0,:,0]
    ys = traj[0,:,1]
    zs = traj[0,:,2]
    scat = ax.scatter(xs, ys, zs, c='blue', marker='o')

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=traj.shape[0],
        fargs=(traj, scat),
        interval=100,
        blit=False
    )
    ani.save('brute_disk_galaxy.gif', writer=PillowWriter(fps=40))
    plt.show()

# main entry

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_bodies', type=int, default=100)
    parser.add_argument('--dt', type=float, default=0.005)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--eps', type=float, default=2.0)
    args = parser.parse_args()

    # create two cold galaxies far apart
    # gal1 = make_spherical_galaxy(center=[-50,0,0], idx_offset=0,
    #                              N=args.num_bodies, R=50.0)
    # gal2 = make_spherical_galaxy(center=[0,50,-50], idx_offset=args.num_bodies,
    #                              N=args.num_bodies, R=50.0)
    # bodies = gal1 + gal2

    bodies = generate_two_disk_collision()

    # flatten for numba integrator
    pos = np.vstack([b.p for b in bodies])
    vel = np.vstack([b.v for b in bodies])
    q   = np.array([b.q for b in bodies])

    # run Verlet with brute-force numba kernel
    traj = simulate_verlet_numba(pos, vel, q,
                                 dt=args.dt,
                                 nsteps=args.num_steps,
                                 eps=args.eps)

    np.save('brute_disk_galaxy.npy', traj)
    visualize_sim(traj)

if __name__ == '__main__':
    main()
