#Montando os poliedros

import numpy as np
import math
#Para o plot:
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull

def tetraedro():

    #Retorna os vértices do tetraedro

    vert = np.zeros([4,3])

    vert[0] = np.array([1.0,1.0,1.0])
    vert[1] = np.array([1.0,-1.0,-1.0])
    vert[2] = np.array([-1.0,1.0,-1.0])
    vert[3] = np.array([-1.0,-1.0,1.0])

    #Normalizando os vetores dos vértices
    for i in range(0,4):
        vert[i] = vert[i]/math.sqrt(np.dot(vert[i],vert[i]))
        
    return vert
    
def octaedro():

    #Retorna os vértices do tetraedro

    vert = np.zeros([6,3])

    vert[0] = np.array([1.0,0.0,0.0])
    vert[1] = np.array([-1.0,0.0,0.0])
    vert[2] = np.array([0.0,1.0,0.0])
    vert[3] = np.array([0.0,-1.0,0.0])
    vert[4] = np.array([0.0,0.0,1.0])
    vert[5] = np.array([0.0,0.0,-1.0])

    #Normalizando os vetores dos vértices
    for i in range(0,6):
        vert[i] = vert[i]/math.sqrt(np.dot(vert[i],vert[i]))
        
    return vert

def cube():

    #Retorna os vértices do tetraedro

    vert = np.zeros([8,3])

    vert[0] = np.array([1.0,1.0,1.0])
    vert[1] = np.array([1.0,1.0,-1.0])
    vert[2] = np.array([1.0,-1.0,1.0])
    vert[3] = np.array([-1.0,1.0,1.0])
    vert[4] = np.array([-1.0,1.0,-1.0])
    vert[5] = np.array([-1.0,-1.0,1.0])
    vert[6] = np.array([1.0,-1.0,-1.0])
    vert[7] = np.array([-1.0,-1.0,-1.0])

    #Normalizando os vetores dos vértices
    for i in range(0,8):
        vert[i] = vert[i]/math.sqrt(np.dot(vert[i],vert[i]))
        
    return vert

def icosaedro():

    #Retorna os vértices do icosaedro

    vert = np.zeros([12,3])

    gr = ((1+math.sqrt(5))/2)#Golden ratio

    vert[0] = np.array([0.0,1.0,gr])
    vert[1] = np.array([0.0,-1.0,-gr])
    vert[2] = np.array([0.0,1.0,-gr])
    vert[3] = np.array([0.0,-1.0,gr])
    vert[4] = np.array([1.0,gr,0.0])
    vert[5] = np.array([1.0,-gr,0.0])
    vert[6] = np.array([-1.0,gr,0.0])
    vert[7] = np.array([-1.0,-gr,0.0])
    vert[8] = np.array([gr,0.0,1.0])
    vert[9] = np.array([gr,0.0,-1.0])
    vert[10] = np.array([-gr,0.0,1.0])
    vert[11] = np.array([-gr,0.0,-1.0])

    #Normalizando os vetores dos vértices
    for i in range(0,12):
        vert[i] = vert[i]/math.sqrt(np.dot(vert[i],vert[i]))
        
    return vert

def dodecaedro():

    #Retorna os vértices do tetraedro

    vert = np.zeros([20,3])
    
    gr = ((1+math.sqrt(5))/2)#Golden ratio

    vert[0] = np.array([1.0,1.0,1.0])
    vert[1] = np.array([1.0,1.0,-1.0])
    vert[2] = np.array([1.0,-1.0,1.0])
    vert[3] = np.array([-1.0,1.0,1.0])
    vert[4] = np.array([-1.0,1.0,-1.0])
    vert[5] = np.array([-1.0,-1.0,1.0])
    vert[6] = np.array([1.0,-1.0,-1.0])
    vert[7] = np.array([-1.0,-1.0,-1.0])
    vert[8] = np.array([0.0,gr,1/gr])
    vert[9] = np.array([0.0,-gr,-1/gr])
    vert[10] = np.array([0.0,gr,-1/gr])
    vert[11] = np.array([0.0,-gr,1/gr])
    vert[12] = np.array([gr,1/gr,0.0])
    vert[13] = np.array([gr,-1/gr,0.0])
    vert[14] = np.array([-gr,1/gr,0.0])
    vert[15] = np.array([-gr,-1/gr,0.0])
    vert[16] = np.array([1/gr,0.0,gr])
    vert[17] = np.array([1/gr,0.0,-gr])
    vert[18] = np.array([-1/gr,0.0,gr])
    vert[19] = np.array([-1/gr,0.0,-gr])

    #Normalizando os vetores dos vértices
    for i in range(0,20):
        vert[i] = vert[i]/math.sqrt(np.dot(vert[i],vert[i]))
        
    return vert
    
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

    ax.add_collection3d(polys)
    plt.show()
    
    return 

vert = dodecaedro()
print(math.sqrt(np.dot(vert[1],vert[1])))
plot_polyhedron(vert)
