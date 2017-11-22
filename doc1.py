"""Thomson Problem solver"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

# Plot creation and parameters
fig = plt.figure()
ax = Axes3D(fig)
ax.set_aspect("equal")
ax.axis("off")
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

def random_uniform_sphere(N):
    """Create N random points on a unit sphere using a uniform distribution"""
    points = []
    for _ in itertools.repeat(None, N):
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2*np.pi)
        points.append([np.cos(theta), np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi)])
    return points

def random_gaussian_sphere(N, theta, phi, variance):
    """Create N random points on a unit sphere centered using a gaussian distribution"""
    points = []
    for _ in itertools.repeat(None, N):
        bm_rand1 = np.random.uniform(0, 1)
        bm_rand2 = np.random.uniform(0, 1)
        theta_gaus = np.sqrt(-2*np.log(bm_rand1))*np.cos(2*np.pi*bm_rand2)*np.sqrt(variance)+theta
        phi_gaus = np.sqrt(-2*np.log(bm_rand1))*np.sin(2*np.pi*bm_rand2)*np.sqrt(2*variance)+phi
        points.append([np.cos(theta_gaus), np.sin(theta_gaus)*np.cos(phi_gaus), np.sin(theta_gaus)*np.sin(phi_gaus)])
    return points

def distance(point1, point2):
    """Distance between 2 points"""
    return np.sqrt((point2[0]-point1[0])**2+(point2[1]-point1[1])**2+(point2[2]-point1[2])**2)

def metropolis(points, iterations, temperature, method, variance):
    """Apply the Metropolis algorithm to a set of points"""
    system_energy = 0
    for i in range(0, len(points)):
        for j in range(0, len(points)):
            if j <= i:
                continue
            else:
                system_energy += 1/(distance(points[i], points[j]))
    print("starting energy = %f" % system_energy)

    for _ in itertools.repeat(None, iterations):
        i = np.random.randint(0, len(points)-1) # Pick a random point from the pointlist
        if method == "uniform": # Generates the compared point by a uniform random distribution
            random_point = random_uniform_sphere(1)[0]
        elif method == "gaussian": # Generates the compared point by a local gaussian distribution centered on the chosen existing point
            theta = np.arccos(points[i][0])
            phi = np.arctan2(points[i][2], points[i][1])
            random_point = random_gaussian_sphere(1, theta, phi, variance)[0]
        else:
            raise ValueError("Invalid method")
        old_point_energy = 0
        new_point_energy = 0
        for j in range(0, len(points)): # Compare the energies of the old and new point
            if i == j:
                continue
            else:
                old_point_energy += 1/distance(points[i], points[j])
                new_point_energy += 1/distance(random_point, points[j])
        if old_point_energy > new_point_energy: # The new point is improved so replaces the old point
            points[i] = random_point
            system_energy += (new_point_energy - old_point_energy)
            print("energy down -> current energy = %f, energy change = %f" % (system_energy, 2*(new_point_energy - old_point_energy)))
        else: # If the new point is not an improvement it still may be chosen according to its boltzmann probability
            j = np.random.uniform(0, 1)
            if j <= np.exp((old_point_energy-new_point_energy)/(1.3806503*(10**-23)*temperature)):
                # print "exp(delta(e)/kt = %f)" % np.exp((new_point_energy-old_point_energy)/(1.3806503*(10**-23)*temperature))
                points[i] = random_point
                system_energy -= (old_point_energy-new_point_energy)
                print("energy up -> current energy = %f, energy change = %f" % (system_energy, 2*(new_point_energy - old_point_energy)))
    print("final energy = %f" % system_energy)

    return points

def pointplot(points):
    """Display a set of points in 3D"""
    # Draws a sphere
    phi = np.linspace(0, 2*np.pi, 200)
    theta = np.linspace(0, np.pi, 200)
    xm = np.outer(np.cos(phi), np.sin(theta))
    ym = np.outer(np.sin(phi), np.sin(theta))
    zm = np.outer(np.ones(np.size(phi)), np.cos(theta))
    ax.plot_surface(xm, ym, zm, alpha=0.05, linewidth=0, color="k")
    # Draws the set of points
    ax.scatter([i[0] for i in points], [i[1] for i in points], [i[2] for i in points])

def hull(points):
    """Create a convex hull for a set of points"""
    pa = np.asarray(points)
    hullpoints = ConvexHull(pa)
    ax.scatter(pa[:, 0], pa[:, 1], pa[:, 2], s=10, color='k', alpha=1)
    for i in hullpoints.simplices:
        verts = [list(zip(pa[i, 0], pa[i, 1], pa[i, 2]))]
        poly = Poly3DCollection(verts)
        poly.set_facecolors((1, 0.9, 0.9, 0.7))
        poly.set_edgecolors((1, 0, 0, 0.5))
        ax.add_collection3d(poly)

def main():
    """Call the desired functions and display them"""
    iterations = 10000
    temperature = 5*10**16
    method = "gaussian"
    variance = 0.00001
    rus = random_uniform_sphere(6)
    lmcm = metropolis(rus, iterations, temperature, method, variance)
    #pointplot(lmcm)
    hull(lmcm)
    plt.show()

if __name__ == "__main__":
    main()
