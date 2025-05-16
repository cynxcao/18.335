import numpy as np
from uniform_fmm import Body, UniformFMM
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from matplotlib.animation import PillowWriter
import matplotlib
matplotlib.use('MacOSX') 

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
            
            r2 = dx.dot(dx) + eps**2
            r  = np.sqrt(r2)

            # plummer softening reee
            phi[i] += other.q / (r)
            grad[i] -= other.q * dx / (r**3)

    return phi, grad

def sample_uniform_sphere(n, radius, center):
    dirs = np.random.normal(size=(n,3))
    dirs /= np.linalg.norm(dirs,axis=1)[:,None]
    r = np.random.rand(n)**(1/3) * radius
    return center + dirs * r[:,None]

def make_spherical_galaxy(center, idx_offset=0, N=25, R=5.0, mass_range=(5.0,10.0), G=3000.0):
    ps = sample_uniform_sphere(N, R, np.array(center))
    ms = np.random.uniform(mass_range[0], mass_range[1], size=N)
    qs = G * ms
    vs = np.zeros((N,3))
    bodies = []
    for i,(p,v,q) in enumerate(zip(ps, vs, qs)):
        bodies.append(Body(p.tolist(), v.tolist(), q, idx_offset + i))
    return bodies

def simulate(bodies, P, max_level, theta, dt, nsteps, algo="uniform-fmm"):
    # step through simulation (can choose brute or fmm)
    all_bodies = []

    for step in range(nsteps):
        if algo == "uniform-fmm":
            fmm = UniformFMM(bodies, P, max_level, theta)
            _, a_old = fmm.solve()
        elif algo == "brute":
            _, a_old = brute_force_soln(bodies)

        for i, b in enumerate(bodies):
            b.p += b.v*dt + 0.5*a_old[i]*(dt**2)

        if algo == "uniform-fmm":
            # fmm = UniformFMM(bodies, P, max_level, theta)
            _, a_new = fmm.solve()
        elif algo == "brute":
            a_new = a_old

        for i, b in enumerate(bodies):
            b.v += 0.5*(a_old[i] + a_new[i])*dt

        all_bodies.append(np.stack([b.p.copy() for b in bodies], axis=0))

    return np.stack(all_bodies, axis=0) 

def make_disk_galaxy(center, idx_offset, N=100, radius=50.0, thickness=0.5, mass_range=(5.0,5.0), G=300.0):
    
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
            vs[i,0] = -speed * np.sin(th[i])
            vs[i,1] =  speed * np.cos(th[i])

    bodies = []
    for i in range(N):
        idx = idx_offset + i
        p   = pts[i].tolist()
        v   = vs[i].tolist()
        qv  = qs[i]
        bodies.append(Body(p, v, qv, idx))
    return bodies

def generate_two_disk_collision(separation=100.0, relative_speed=0.0, **disk_kwargs):
    # first galaxy at x = -sep/2, moving +x
    gal1 = make_disk_galaxy(center=[-separation/2, 0.0, 0.0], idx_offset=0,  **disk_kwargs)

    # second at x = +sep/2, moving -x
    gal2 = make_disk_galaxy(center=[+separation/2, 0.0, 0.0], idx_offset=len(gal1), **disk_kwargs)

    #approach velocities
    v_half = relative_speed/2.0
    for b in gal1:
        b.v[0] +=  v_half
    for b in gal2:
        b.v[0] += -v_half

    return gal1 + gal2

def update(frame, positions, scat):
    # updates each frame by grabbing 25 points
    xs = positions[frame, :, 0]
    ys = positions[frame, :, 1]
    zs = positions[frame, :, 2]

    scat._offsets3d = (xs, ys, zs)
    return scat
   
def visualize_sim(positions):
    # make animated gif
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
        v = np.array([i, i, i], float)
        q = 1.0
        idx = i
        all_bodies.append(Body(p, v, q, idx))

    print([b.v for b in all_bodies])

    positions_over_time = simulate(
        bodies     = all_bodies,
        P          = 6,
        max_level  = 2,
        theta      = 0.5,
        dt         = 1.0, # needs to be small or gradients blow up
        nsteps     = 10, # longer sim time
        algo       = "uniform-fmm"
    )

    np.save("fmm-reee.npy", positions_over_time)

    visualize_sim(positions_over_time)
    

if __name__ == "__main__":
    main()