import numpy as np
#Para o plot:
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull

def vertices(n):

    #n é o ciclo de criação de vértice
    #Primeiro ciclo: 6 vértices, percorre os ângulos a cada 90graus (pi/2)
    #Próximos ciclos, número de vértices percorre os ângulos a cada (pi/2)*(1/2)^n
    #assim, temos (8n^2+4n) vértices por ciclo, mas vários vão ser repetidos!
    #Usar np.unique() para tirar redundância do array

    delta = (np.pi)*(1/2)**n

    len_theta = int((2*np.pi)/delta)
    len_phi = int(np.pi/delta)+1
    theta = np.zeros(len_theta) 
    phi = np.zeros(len_phi)

    for i in range(len_theta):
        theta[i] = i*delta
    for i in range(len_phi):
        phi[i] = i*delta

    vert = np.zeros((len_theta*len_phi,3))

    for k in range(len_theta):
        for j in range(len_phi):
            i = k*len_phi + j
            vert[i][0] = np.around(np.sin(phi[j])*np.cos(theta[k]),decimals=15)
            vert[i][1] = np.around(np.sin(phi[j])*np.sin(theta[k]),decimals=15)
            vert[i][2] = np.around(np.cos(phi[j]),decimals=15)

    vert = np.unique(vert,axis=0)

    return theta,phi,vert

def polyhedron(vert):
    #Poliedro
    hull = ConvexHull(vert)
    #Insphere radius
    r = np.min(np.abs(hull.equations[:, -1]))
    return hull, r
    
def plot_polyhedron(vert):

    #Plota o poliedro recebendo apenas os vertices da forma: [[x,y,z],[x,y,z],...]

    hull,r = polyhedron(vert)

    polys = Poly3DCollection([hull.points[simplex] for simplex in hull.simplices])

    polys.set_edgecolor('deeppink')
    polys.set_linewidth(.8)
    polys.set_facecolor('hotpink')
    polys.set_alpha(.25)
    
    #Construindo a esfera interna
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
    x = r*np.cos(u)*np.sin(v)
    y = r*np.sin(u)*np.sin(v)
    z = r*np.cos(v)
    
    #Construindo a esfera unitária
    x_uni = np.cos(u)*np.sin(v)
    y_uni = np.sin(u)*np.sin(v)
    z_uni = np.cos(v)
    
    ax = Axes3D(plt.figure())
    
    #Plot insphere
    ax.plot_surface(x,y,z,color='lime',alpha=.35)
    #Plot unitary sphere
    ax.plot_surface(x_uni,y_uni,z_uni,color='lightgray',alpha=.15)
    #Plot polyhedron
    ax.set_xlim3d(-1,1)
    ax.set_ylim3d(-1,1)
    ax.set_zlim3d(-1,1)
    ax.set_box_aspect([1,1,1])
    #plt.axis('off')

    ax.add_collection3d(polys)
    plt.show()
    
    return 
    
#theta, phi, vert= vertices(2)
#print(np.degrees(theta))
#print(np.degrees(phi))
#print(vert.shape)
#hull, r = polyhedron(vert)
#print(r)
#plot_polyhedron(vert)
