from cmath import sqrt
import picos as pic
import numpy as np
import math #Usar sqrt
from scipy.spatial import ConvexHull #Medições
import time
#Para o plot:
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt


def states():
    #Definir kets |00>, |01>, |10> e |11>
    ket_00 = np.array([[1],[0],[0],[0]])
    ket_01 = np.array([[0],[1],[0],[0]])
    ket_10 = np.array([[0],[0],[1],[0]])
    ket_11 = np.array([[0],[0],[0],[1]])
    #Definir ket psi = (|01>-|10>)/sqrt(2) e rho = |psi><psi|
    psi = (ket_01-ket_10)/math.sqrt(2)
    rho = psi@np.transpose(psi)
    #Definir matriz densidade I/4
    rho_sep = (np.eye(4))/4
    rho_00  = ket_00@np.transpose(ket_00)
    return rho, rho_sep , rho_00

#rho,rho_sep = states()
#print(rho)
#print(rho_sep)

def measurements(n):
    #n é o ciclo de criação de vértice

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
    [0,1/2,(np.sqrt(3))/2],[0,-1/2,-(np.sqrt(3))/2],
    [0,(np.sqrt(3))/2,1/2],[0,-(np.sqrt(3))/2,-1/2],
    [1/2,0,(np.sqrt(3))/2],[-1/2,0,-(np.sqrt(3))/2],
    [(np.sqrt(3))/2,0,1/2],[-(np.sqrt(3))/2,0,-1/2],
    [(np.sqrt(3))/4,1/4,(np.sqrt(3))/2],[-(np.sqrt(3))/4,-1/4,-(np.sqrt(3))/2],
    [1/4,(np.sqrt(3))/4,(np.sqrt(3))/2],[-1/4,-(np.sqrt(3))/4,-(np.sqrt(3))/2],
    [3/4,(np.sqrt(3))/4,1/2],[-3/4,-(np.sqrt(3))/4,-1/2],
    [(np.sqrt(3))/4,3/4,1/2],[-(np.sqrt(3))/4,-3/4,1/2],
    [-(np.sqrt(3))/4,1/4,(np.sqrt(3))/2],[(np.sqrt(3))/4,-1/4,-(np.sqrt(3))/2],
    [-1/4,(np.sqrt(3))/4,(np.sqrt(3))/2],[1/4,-(np.sqrt(3))/4,-(np.sqrt(3))/2],
    [-3/4,(np.sqrt(3))/4,1/2],[3/4,-(np.sqrt(3))/4,-1/2],
    [-(np.sqrt(3))/4,3/4,1/2],[(np.sqrt(3))/4,-3/4,-1/2],
    [-1/2,(np.sqrt(3))/2,0],[1/2,-(np.sqrt(3))/2,0],
    [-(np.sqrt(3))/2,1/2,0],[(np.sqrt(3))/2,-1/2,0],
    [0,-1/2,(np.sqrt(3))/2],[0,1/2,-(np.sqrt(3))/2],
    [0,-(np.sqrt(3))/2,1/2],[0,(np.sqrt(3))/2,-1/2],
    [1/2,0,-(np.sqrt(3))/2],[-1/2,0,(np.sqrt(3))/2],
    [(np.sqrt(3))/2,0,-1/2],[-(np.sqrt(3))/2,0,1/2]
    ])

    vert_q = np.concatenate((vert_t,vert_q))

    if n == 1:
        vert = vert_p
    elif n == 2:
        vert = vert_s
    elif n == 3:
        vert = vert_t
    elif n == 4:
        vert = vert_q

    m_k = vert.shape[0]
    medicoes = np.zeros([m_k,2,2], dtype=complex)

    sum_med = np.zeros([2,2])

    for i in range(m_k):
        med_00 = (1+vert[i][2])/2
        med_01 = (vert[i][0]-vert[i][1]*1j)/2
        med_10 = (vert[i][0]+vert[i][1]*1j)/2
        med_11 = (1-vert[i][2])/2
    
        medicoes[i] = [[med_00,med_01],[med_10,med_11]]

    # for i in range(int(m_k/2)):
    #     print("Soma")
    #     print(medicoes[2*i]+medicoes[2*i+1])
    
    #Poliedro
    hull = ConvexHull(vert)
    #Insphere radius
    r = np.min(np.abs(hull.equations[:, -1]))

    #Plota o poliedro recebendo apenas os vertices da forma: [[x,y,z],[x,y,z],...]

    #polys = Poly3DCollection([hull.points[simplex] for simplex in hull.simplices])

    #polys.set_edgecolor('deeppink')
    #polys.set_linewidth(.8)
    #polys.set_facecolor('hotpink')
    #polys.set_alpha(.25)
    
    #Construindo a esfera interna
    #u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
    #x = r*np.cos(u)*np.sin(v)
    #y = r*np.sin(u)*np.sin(v)
    #z = r*np.cos(v)
    
    #Construindo a esfera unitária
    #x_uni = np.cos(u)*np.sin(v)
    #y_uni = np.sin(u)*np.sin(v)
    #z_uni = np.cos(v)
    
    #ax = Axes3D(plt.figure())
    
    #Plot insphere
    #ax.plot_surface(x,y,z,color='lime',alpha=.35)
    #Plot unitary sphere
    #ax.plot_surface(x_uni,y_uni,z_uni,color='lightgray',alpha=.15)
    #Plot polyhedron
    #ax.set_xlim3d(-1,1)
    #ax.set_ylim3d(-1,1)
    #ax.set_zlim3d(-1,1)
    #ax.set_box_aspect([1,1,1])
    #plt.axis('off')

    #ax.add_collection3d(polys)
    #plt.show()
    #plt.savefig('poliedro_'+str(i+1)+'.png')
    
    return medicoes,r
    
#medicoes,r = measurements(4)
#print(medicoes.shape)
#print(r)

def strategies_LHS(m,k):

    #m = nº de medições
    #k = nº de resultados
    #k**m = nº de estratégias = n_lambdas

    n_lambdas = k**m
    
    all_est = [np.base_repr(el+n_lambdas,base=k)[-m:] for el in range(n_lambdas)]
    
    all_est = np.array([[int(digit) for digit in el] for el in all_est])

    detp = np.zeros((n_lambdas,k*m))

    for i in range(n_lambdas):
        for j in range(m):
            aux = np.zeros(k)
            aux[all_est[i][j]] = 1
            detp[i][j*k:j*k+k] = np.array(aux)    

    return detp

#detp = strategies_LHS(3,2)
#print(detp)
#print(detp[0,0])

def SDP_LHS(m,k,rho,rho_sep,eta,detp,medicoes):

    rho_lhs = [(i*0.044)*rho+(1-i*0.044)*rho_sep for i in range(10)]
    #print(rho_lhs)
    t = time.process_time()
    print('Entrando no SDP')

    P = pic.Problem()

    q = pic.RealVariable('q')
    
    p_k = [pic.RealVariable('p[{}]'.format(i),(1,1)) for i in range(10)]
    
    chi = pic.HermitianVariable('chi',(4,4))

    sigma = [pic.HermitianVariable('Sigma_lambda[{}]'.format(i),(2,2)) for i in range(k**m)]

    rho_q = rho*q+(1-q)*rho_sep

    rho_eta = eta*chi+(1-eta)*(pic.partial_trace(rho_sep,subsystems=1,dimensions=(2,2)))@(pic.partial_trace(chi,subsystems=0,dimensions=(2,2)))
    
    sum_lhs = pic.sum([p_k[i]*rho_lhs[i] for i in range(10)])

    est_det = [pic.sum([sigma[j]*detp[j,i] for j in range(k**m)]) for i in range(k*m)]

    est = [(np.kron(medicoes[i],np.eye(2)))*chi for i in range(k*m)]

    t_1 = time.process_time() - t
    print('Definidas as variaveis',t_1)
    P.add_constraint(q<=1)

    P.add_constraint(q>=0)

    P.add_list_of_constraints([sigma[i]>>0 for i in range(k**m)]) 
    
    P.add_list_of_constraints([p_k[i]>=0 for i in range(10)])

    P.add_constraint(rho_q - rho_eta - sum_lhs>>0)
    
    P.add_constraint(pic.partial_transpose(rho_q - rho_eta - sum_lhs,subsystems=1,dimensions=(2,2))>>0)
    
    P.add_constraint(pic.trace(chi)+pic.sum([p_k[i]*np.trace(rho_lhs) for i in range(10)])>=0)

    P.add_list_of_constraints([pic.partial_trace(est[i],subsystems=0,dimensions=(2,2))==est_det[i] for i in range(k*m)])

    t_2 = time.process_time() - t_1
    print('Definidas as restrições',t_2)
    P.set_objective('max',q)

    solution = P.solve()
    t_3 = time.process_time() - t_2
    print('Solucionado!',t_3)

    return P, solution, q, chi, sigma, rho_q, rho_eta, est_det, est


#Aqui chamamos as funções!
rho,rho_sep, rho_00 = states()

for i in range(3):
    print('Entrando no ciclo ',i+1)
    medicoes,r = measurements(i+1)
    m_k = medicoes.shape
    print('Número de vértices:')
    print(m_k[0])
    print('Raio inscrito:')
    print(r)
    k = 2
    m = int(m_k[0]/k)
    detp = strategies_LHS(m,k)

    P,solution,q,chi,sigma_lambda,rho_q,rho_eta,est_det,est = SDP_LHS(m,k,rho,rho_sep,r,detp,medicoes)
    print(P)
    print('q:',q)
    print('Ciclo ',i+1,' finalizado.')
