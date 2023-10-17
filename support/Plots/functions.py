import numpy as np
from numpy import linalg as LA
from random import random
from scipy.stats import unitary_group
import picos as pic
from scipy.spatial import ConvexHull #Medições
#Para o plot:
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize


# First block: Functions to create and certify the states
# Generation of the random matrix from the Ginibre ensemble
'''A complex matrix with elements having real and complex part distributed with the normal distribution'''
def G_matrix(n,m):
    # Matrix G of size n x m
    G = (np.random.randn(n, m) + 1j * np.random.randn(n, m)) / np.sqrt(2)
    return G

# Generation a random mixed density matrix (Bures metric)
def rho_mixed(n):
    # Create random unitary matrix
    U = unitary_group.rvs(n)
    # Create random Ginibre matrix
    G = G_matrix(n,n)
    # Create identity matrix
    I = np.eye(4)
    # Construct density matrix
    rho = (I+U)@G@(G.conjugate().T)@(I+U.conjugate().T)
    # Normalize density matrix
    rho = rho/(rho.trace())
    return rho

# Generation a random mixed density matrix (Hilbert-Schmidt metric)
def rho_mixed_HS(n):
    # Create random Ginibre matrix
    G = G_matrix(n,n)
    # Construct density matrix
    rho = G@(G.conjugate().T)
    # Normalize density matrix
    rho = rho/(rho.trace())
    return rho

# Entanglement certification using PPT criterion
def Ent_cert(rho):
    # Calculate partial transpose
    n = rho.shape
    rho_TA = np.zeros((n[0],n[1]),dtype=np.complex_)
    a = int(n[0]/2)
    b = int(n[1]/2)
    rho_TA[:a,:b] = rho[:a,:b]
    rho_TA[a:,b:] = rho[a:,b:]
    rho_TA[a:,:b] = rho[a:,:b].T
    rho_TA[:a,b:] = rho[:a,b:].T
    # v - eigenvectors, w - eigenvalues
    w, v = LA.eig(rho_TA)
    # PPT Criterion: Are all eigenvalues >=0?
    if all(i >= 0 for i in w):
        # print('Yes: separable state.')
        ppt = 0
    else:
        # print('No: entangled state.')
        ppt = 1
    return w,v,ppt

# Second block: Fucntions to create the measurements
# Creating the measurements(n) function
def measurements(n,PLOT=False):
    #INPUT: n is the cicle we are in, may have the values: [1,2,3,4]; PLOT:False/True to plot or not the polyhedron

    #Create the vertices of the polytope
    vert_p = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])

    vert_s = np.array([[0,1/(np.sqrt(2)),1/(np.sqrt(2))],[0,-1/(np.sqrt(2)),-1/(np.sqrt(2))],
    [0,-1/(np.sqrt(2)),1/(np.sqrt(2))],[0,1/(np.sqrt(2)),-1/(np.sqrt(2))],
    [1/(np.sqrt(2)),0,1/(np.sqrt(2))],[-1/(np.sqrt(2)),0,-1/(np.sqrt(2))],
    [-1/(np.sqrt(2)),0,1/(np.sqrt(2))],[1/(np.sqrt(2)),0,-1/(np.sqrt(2))],
    [1/(np.sqrt(2)),1/(np.sqrt(2)),0],[-1/(np.sqrt(2)),-1/(np.sqrt(2)),0],
    [-1/(np.sqrt(2)),1/(np.sqrt(2)),0],[1/(np.sqrt(2)),-1/(np.sqrt(2)),0]
    ])

    vert_s = np.concatenate((vert_p,vert_s))

    vert_t = np.array([[1/2,1/2,1/(np.sqrt(2))],[-1/2,-1/2,-1/(np.sqrt(2))],
    [-1/2,1/2,1/(np.sqrt(2))],[1/2,-1/2,-1/(np.sqrt(2))],
    [1/2,1/2,-1/(np.sqrt(2))],[-1/2,-1/2,1/(np.sqrt(2))],
    [1/2,-1/2,1/(np.sqrt(2))],[-1/2,1/2,-1/(np.sqrt(2))]
    ])

    vert_t = np.concatenate((vert_s,vert_t))

    vert_q = np.array([[1/2,(np.sqrt(3))/2,0],[-1/2,-(np.sqrt(3))/2,0],
    [(np.sqrt(3))/2,1/2,0],[-(np.sqrt(3))/2,-1/2,0],
    [0,(np.sqrt(3))/2,1/2],[0,-(np.sqrt(3))/2,-1/2],
    [(np.sqrt(3))/2,0,1/2],[-(np.sqrt(3))/2,0,-1/2],
    [-1/2,(np.sqrt(3))/2,0],[1/2,-(np.sqrt(3))/2,0],
    [-(np.sqrt(3))/2,1/2,0],[(np.sqrt(3))/2,-1/2,0],
    [0,-(np.sqrt(3))/2,1/2],[0,(np.sqrt(3))/2,-1/2],
    [(np.sqrt(3))/2,0,-1/2],[-(np.sqrt(3))/2,0,1/2]
    ])

    vert_q = np.concatenate((vert_t,vert_q))

    #Choose which set of vertices we want to use depending on the cycle we are in
    if n == 1:
        vert = vert_p
    elif n == 2:
        vert = vert_s
    elif n == 3:
        vert = vert_t
    elif n == 4:
        vert = vert_q

    #Create the measurements of each vertex
    m_k = vert.shape[0]
    medicoes = np.zeros([m_k,2,2], dtype=complex)

    for i in range(m_k):
        med_00 = (1+vert[i][2])/2
        med_01 = (vert[i][0]-vert[i][1]*1j)/2
        med_10 = (vert[i][0]+vert[i][1]*1j)/2
        med_11 = (1-vert[i][2])/2
    
        medicoes[i] = [[med_00,med_01],[med_10,med_11]]

    #Verify that the sum of each dichotomous measurement is the identity
    #for i in range(int(m_k/2)):
        #print("Sum")
        #print(medicoes[2*i]+medicoes[2*i+1])
    
    #Construct the polyhedron
    hull = ConvexHull(vert)
    #Find the Insphere radius
    r = np.min(np.abs(hull.equations[:, -1]))

    #Plot
    if PLOT == True:
        polys = Poly3DCollection([hull.points[simplex] for simplex in hull.simplices])

        # Good choice of colors: 'deeppink' and 'hotpink'
        polys.set_edgecolor('#35193e')
        polys.set_linewidth(.8)
        polys.set_facecolor('#e13342')
        polys.set_alpha(.25)
        
        #Build the insphere
        u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
        x = r*np.cos(u)*np.sin(v)
        y = r*np.sin(u)*np.sin(v)
        z = r*np.cos(v)
        
        #Build the Bloch sphere
        x_uni = np.cos(u)*np.sin(v)
        y_uni = np.sin(u)*np.sin(v)
        z_uni = np.cos(v)
        
        fig = plt.figure()
        ax = fig.add_subplot(111,projection="3d")
        #ax = Axes3D(plt.figure())
        #Plot Bloch sphere
        ax.plot_surface(x_uni,y_uni,z_uni,color='#f6b48f',alpha=.15)
        #Plot insphere
        # # Good choice of color: 'lime', alpha=.35
        # ax.plot_surface(x,y,z,color='#e13342',alpha=.5)
        #Plot polyhedron
        ax.set_xlim3d(-1,1)
        ax.set_ylim3d(-1,1)
        ax.set_zlim3d(-1,1)
        ax.set_box_aspect([1,1,1])
        ax.set_xticks([-1,0,1])
        ax.set_yticks([-1,0,1])
        ax.set_zticks([-1,0,1])
        ax.view_init(15,75)
        #plt.axis('off')
        
        ax.add_collection3d(polys)
        #ax.set_title(str(int(m_k/2))+' measurements',fontsize=20)
        #plt.show()
        plt.savefig('poliedro_'+str(m_k)+'.png',dpi=300, bbox_inches='tight', transparent=True)
        plt.savefig('poliedro_'+str(m_k)+'.pdf', bbox_inches='tight')
    
    #Return the measurements and the insphere radius
    return medicoes,r

def pentakis(PLOT=False):
    #INPUT: n is the cicle we are in, may have the values: [1,2,3,4]; PLOT:False/True to plot or not the polyhedron

    #Create the vertices of the polytope
    g_r = (1+np.sqrt(5))/2
    vert = np.array([[0,1,g_r],[0,-1,-g_r],
                     [g_r,0,1],[-g_r,0,-1],
                     [1,g_r,0],[-1,-g_r,0],
                     [0,-1,g_r],[0,1,-g_r],
                     [g_r,0,-1],[-g_r,0,1],
                     [-1,g_r,0],[1,-g_r,0],
                     [1,1,1],[-1,-1,-1],
                     [1,1,-1],[-1,-1,1],
                     [1,-1,1],[-1,1,-1],
                     [-1,1,1],[1,-1,-1],
                     [g_r,1/g_r,0],[-g_r,-1/g_r,0],
                     [0,g_r,1/g_r],[0,-g_r,-1/g_r],
                     [1/g_r,0,g_r],[-1/g_r,0,-g_r],
                     [-g_r,1/g_r,0],[g_r,-1/g_r,0],
                     [0,-g_r,1/g_r],[0,g_r,-1/g_r],
                     [1/g_r,0,-g_r],[-1/g_r,0,g_r]])

    vert  = normalize(vert)
    #Create the measurements of each vertex
    m_k = vert.shape[0]
    medicoes = np.zeros([m_k,2,2], dtype=complex)

    for i in range(m_k):
        med_00 = (1+vert[i][2])/2
        med_01 = (vert[i][0]-vert[i][1]*1j)/2
        med_10 = (vert[i][0]+vert[i][1]*1j)/2
        med_11 = (1-vert[i][2])/2
    
        medicoes[i] = [[med_00,med_01],[med_10,med_11]]
    
    #Construct the polyhedron
    hull = ConvexHull(vert)
    #Find the Insphere radius
    r = np.min(np.abs(hull.equations[:, -1]))

    #Plot
    if PLOT == True:
        polys = Poly3DCollection([hull.points[simplex] for simplex in hull.simplices])

        # Good choice of colors: 'deeppink' and 'hotpink'
        polys.set_edgecolor('blue')
        polys.set_linewidth(.8)
        polys.set_facecolor('azure')
        polys.set_alpha(.25)
        
        #Build the insphere
        u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
        x = r*np.cos(u)*np.sin(v)
        y = r*np.sin(u)*np.sin(v)
        z = r*np.cos(v)
        
        #Build the Bloch sphere
        x_uni = np.cos(u)*np.sin(v)
        y_uni = np.sin(u)*np.sin(v)
        z_uni = np.cos(v)
        
        fig = plt.figure()
        ax = fig.add_subplot(111,projection="3d")
        #ax = Axes3D(plt.figure())
        #Plot Bloch sphere
        ax.plot_surface(x_uni,y_uni,z_uni,color='lightgray',alpha=.15)
        #Plot insphere
        # Good choice of color: 'lime', alpha=.35
        ax.plot_surface(x,y,z,color='yellow',alpha=.95)
        #Plot polyhedron
        ax.set_xlim3d(-1,1)
        ax.set_ylim3d(-1,1)
        ax.set_zlim3d(-1,1)
        ax.set_box_aspect([1,1,1])
        ax.set_xticks([-1,0,1])
        ax.set_yticks([-1,0,1])
        ax.set_zticks([-1,0,1])
        ax.view_init(15,75)
        #plt.axis('off')

        ax.add_collection3d(polys)
        #plt.show()
        plt.savefig('poliedro_'+str(m_k)+'.png', transparent=True,dpi=300)
    
    #Return the measurements and the insphere radius
    return medicoes,r

# Third block: Functions to create the deterministic strategies
# Creating the strategies_LHS(m,k) function
def strategies_LHS(m,k):
    #INPUT: m = number of measurements; k = number of results
    #k**m = number of strategies = n_lambdas

    n_lambdas = k**m
    
    #Creating the strategies
    all_est = [np.base_repr(el+n_lambdas,base=k)[-m:] for el in range(n_lambdas)]
    
    all_est = np.array([[int(digit) for digit in el] for el in all_est])

    detp = np.zeros((n_lambdas,k*m))

    for i in range(n_lambdas):
        for j in range(m):
            aux = np.zeros(k)
            aux[all_est[i][j]] = 1
            detp[i][j*k:j*k+k] = np.array(aux)    
            
    #Return the deterministic strategies
    return detp

# Forth block: Functions to create the SDP
# Creating the SDP_LHS(m,k,rho,rho_sep,eta,detp,medicoes) function
def SDP_LHS(m,k,rho,rho_sep,eta,detp,medicoes):

    #Creating the problem
    P = pic.Problem()

    #Creating the optimization variables
    q = pic.RealVariable('q')

    chi = pic.HermitianVariable('chi',(4,4))

    sigma = [pic.HermitianVariable('Sigma_lambda[{}]'.format(i),(2,2)) for i in range(k**m)]

    rho_q = rho*q+(1-q)*rho_sep

    rho_eta = eta*chi+(1-eta)*(pic.partial_trace(rho_sep,subsystems=1,dimensions=(2,2)))@(pic.partial_trace(chi,subsystems=0,dimensions=(2,2)))

    est_det = [pic.sum([sigma[j]*detp[j,i] for j in range(k**m)]) for i in range(k*m)]

    est = [(np.kron(medicoes[i],np.eye(2)))*chi for i in range(k*m)]

    #Creating the constraints
    P.add_constraint(q<=1)

    P.add_constraint(q>=0)

    P.add_list_of_constraints([sigma[i]>>0 for i in range(k**m)]) 

    P.add_constraint(rho_q == rho_eta)

    P.add_list_of_constraints([pic.partial_trace(est[i],subsystems=0,dimensions=(2,2))==est_det[i] for i in range(k*m)])

    #Setting the objective
    P.set_objective('max',q)

    #Finding the solution
    solution = P.solve()

    #Return the problem created, the solution found, the value of q
    return P, solution, q
