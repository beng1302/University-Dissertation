from __future__ import division
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

f = np.exp(1)
N = 80
xmax = 10
Emax = xmax**2
Emin = Emax/N
E = np.linspace(Emin,Emax,N)
gE = [1]*(N)

print E,'\n\n','f values:'

xold = np.random.uniform(-xmax,xmax)
while f >= np.exp(10**(-6)):
	print f
	flat_test = 0
	hE = [0]*(N)
	while flat_test <= 0.95:
		xnew = np.random.uniform(-xmax,xmax)
		Eold = (xold)**2
		Enew = (xnew)**2
		for index in range(0,N):
			if Eold <= E[index]:
				old_N = index
				break
		for index in range(0,N):
			if Enew <= E[index]:
				new_N = index
				break
		p = np.random.uniform(0,1)
		if p < min(gE[old_N]/gE[new_N],1):
			xold = xnew
			gE[new_N] *= f
			hE[new_N] += 1
		else:
			gE[old_N] *= f
			hE[old_N] += 1
		if sum(hE) != 0:
			flat_test = min(hE)/(sum(hE)/len(hE))
	f = f**(1/2)

print '\n',gE/sum(gE)

fig, ax = plt.subplots(figsize=(10,5))
plt.subplots_adjust(left=0.1,bottom=0.15)
ax.bar(E-E[0],gE/sum(gE),width=Emax/N,align='edge',color='green')
ax.grid()
plt.xticks(rotation=45,ha='right')
ax.tick_params(which='both',length=5,direction='out')
ax.set_xticks(np.linspace(0,Emax,21),minor=False)
ax.set_yticks(np.linspace(0,0.25,11),minor=False)
ax.set_xlabel('Energy, E')
ax.set_ylabel('DOS, g(E)')
ax.set_title('Density of states against energy for E=x^2',y=1.05)
plt.show()
