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
    [0,(np.sqrt(3))/2,1/2],[0,-(np.sqrt(3))/2,-1/2],
    [(np.sqrt(3))/2,0,1/2],[-(np.sqrt(3))/2,0,-1/2],
    [-1/2,(np.sqrt(3))/2,0],[1/2,-(np.sqrt(3))/2,0],
    [-(np.sqrt(3))/2,1/2,0],[(np.sqrt(3))/2,-1/2,0],
    [0,-(np.sqrt(3))/2,1/2],[0,(np.sqrt(3))/2,-1/2],
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

    # polys = Poly3DCollection([hull.points[simplex] for simplex in hull.simplices])

    # polys.set_edgecolor('deeppink')
    # polys.set_linewidth(.8)
    # polys.set_facecolor('hotpink')
    # polys.set_alpha(.25)
    
    # #Construindo a esfera interna
    # u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
    # x = r*np.cos(u)*np.sin(v)
    # y = r*np.sin(u)*np.sin(v)
    # z = r*np.cos(v)
    
    # #Construindo a esfera unitária
    # x_uni = np.cos(u)*np.sin(v)
    # y_uni = np.sin(u)*np.sin(v)
    # z_uni = np.cos(v)
    
    # ax = Axes3D(plt.figure())
    
    # #Plot insphere
    # ax.plot_surface(x,y,z,color='lime',alpha=.35)
    # #Plot unitary sphere
    # ax.plot_surface(x_uni,y_uni,z_uni,color='lightgray',alpha=.15)
    # #Plot polyhedron
    # ax.set_xlim3d(-1,1)
    # ax.set_ylim3d(-1,1)
    # ax.set_zlim3d(-1,1)
    # ax.set_box_aspect([1,1,1])
    # #plt.axis('off')

    # ax.add_collection3d(polys)
    # #plt.show()
    # plt.savefig('poliedro_'+str(m_k)+'.png')
    
    return medicoes,r
    
# medicoes,r = measurements(4)
# print(medicoes.shape)
# print(r)

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

# detp = strategies_LHS(3,2)
# print(detp.shape)
# print(detp)

def SDP_LHS(m_anterior,detp_excluidos,m,k,rho,rho_sep,eta,medicoes):

    t = time.process_time()
    print('Entrando no SDP')

    if len(detp_excluidos) == 0:
        detp = strategies_LHS(m,k)
    else:
        detp_anterior = strategies_LHS(m_anterior,k)
        detp_otimizado = np.delete(detp_anterior,detp_excluidos,axis=0)

        detp_novo = strategies_LHS(m-m_anterior,k)

        detp = np.zeros((detp_novo.shape[0]*detp_otimizado.shape[0],detp_novo.shape[1]+detp_otimizado.shape[1]))

        for j in range(detp_otimizado.shape[0]):
            for i in range(detp_novo.shape[0]):
                detp[j*(detp_novo.shape[0])+i] = np.append(detp_otimizado[j],detp_novo[i])

    P = pic.Problem()

    q = pic.RealVariable('q')

    chi = pic.HermitianVariable('chi',(4,4))

    sigma = [pic.HermitianVariable('Sigma_lambda[{}]'.format(i),(2,2)) for i in range(len(detp))]

    rho_q = rho*q+(1-q)*rho_sep

    rho_eta = eta*chi+(1-eta)*(pic.partial_trace(rho_sep,subsystems=1,dimensions=(2,2)))@(pic.partial_trace(chi,subsystems=0,dimensions=(2,2)))

    est_det = [pic.sum([sigma[j]*detp[j,i] for j in range(len(detp))]) for i in range(k*m)]

    est = [(np.kron(medicoes[i],np.eye(2)))*chi for i in range(k*m)]

    t_1 = time.process_time() - t
    print('Definidas as variaveis',t_1)
    P.add_constraint(q<=1)

    P.add_constraint(q>=0)

    P.add_list_of_constraints([sigma[i]>>0 for i in range(len(detp))]) 

    P.add_constraint(rho_q == rho_eta)

    P.add_list_of_constraints([pic.partial_trace(est[i],subsystems=0,dimensions=(2,2))==est_det[i] for i in range(k*m)])

    t_2 = time.process_time() - t_1
    print('Definidas as restrições',t_2)
    P.set_objective('max',q)

    solution = P.solve()
    t_3 = time.process_time() - t_2
    print('Solucionado!',t_3)

    p_lambda = np.zeros(len(sigma))

    detp_exc = np.array([],dtype=int)

    for i in range(len(sigma)):
        p_lambda[i] = np.trace(sigma[i])
        if p_lambda[i] <= 10**(-9):
            detp_exc = np.append(detp_exc,i)

    # print(p_lambda)
    # print(detp_exc)

    if len(detp_exc) == 0:
        detp_exc = np.array([])
    else:
        print('detp_exc:',detp_exc.shape)

    print('detp:',detp.shape)

    return P, solution, q, chi, sigma, rho_q, rho_eta, est_det, est, detp_exc, m


#Aqui chamamos as funções!
rho,rho_sep, rho_00 = states()

m_anterior = 0
detp_excluidos = np.array([])

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

    P,solution,q,chi,sigma_lambda,rho_q,rho_eta,est_det,est,detp_exc,m_ant = SDP_LHS(m_anterior,detp_excluidos,m,k,rho,rho_sep,r,medicoes)
    detp_excluidos = detp_exc
    m_anterior = m
    #print(P)
#     print(solution)
#     #print(solution.primals)
    print('q:',q)
#     print('chi:')
#     print(chi)
#     print('sigma_lambda:')
#     print(sigma_lambda)
#     print('rho_q:')
#     print(rho_q)
#     print('rho_eta:')
#     print(rho_eta)
#     print('est_det:')
#     print(est_det)
#     print('est:')
#     print(est)
#     print('Ciclo ',i+1,' finalizado.')
