print("Primeiro código LHV para estados de 2 qubits")
#Bibliotecas necessárias:
import numpy as np #Usar arrays
import math #Usar sqrt
import picos as pic #Resolver SDPs
from scipy.spatial import ConvexHull #Medições
#Para o plot:
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

print('Solvers supported on this installation of PICOS:',pic.solvers.all_solvers().keys())

print('Solvers available to PICOS on this machine:',pic.solvers.available_solvers())

def WernerClass():

	#Primeira etapa: Estado alvo -> classe de estados de Werner de dois qubits

	#Definir kets |00>, |01>, |10> e |11>
	ket_00 = np.array([[1],[0],[0],[0]])
	ket_01 = np.array([[0],[1],[0],[0]])
	ket_10 = np.array([[0],[0],[1],[0]])
	ket_11 = np.array([[0],[0],[0],[1]])
	#Definir ket psi = (|01>-|10>)/sqrt(2) e rho = |psi><psi|
	psi = (ket_01-ket_10)/math.sqrt(2)
	rho = psi*np.transpose(psi)
	#Definir matriz densidade I = |00><00|+|01><01|+|10><10|+|11><11|
	iden = ket_00*np.transpose(ket_00)+ket_01*np.transpose(ket_01)+ket_10*np.transpose(ket_10)+ket_11*np.transpose(ket_11)
	#Definir rho_sep = I/4
	rho_sep = iden/4
	
	return rho, rho_sep

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

    if n == 1:
        vert = vert_p
    elif n == 2:
        vert = vert_s
    elif n == 3:
        vert = vert_t

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
    #plt.show()
    # plt.savefig('poliedro_'+str(i+1)+'.png')
    
    return medicoes,r

def strategies_LHS(m,k):

    #m = nº de medições
    #k = nº de resultados
    #k**m = nº de estratégias = n_lambdas

    n_lambdas = k**m
    
    all_est = [np.base_repr(el+n_lambdas,base=k)[-m:] for el in range(n_lambdas)]
    
    all_est = np.array([[int(digit) for digit in el] for el in all_est])
    
    #print(all_est)

    detp = np.zeros((n_lambdas,k*m))

    for i in range(n_lambdas):
        for j in range(m):
            aux = np.zeros(k)
            aux[all_est[i][j]] = 1
            detp[i][j*k:j*k+k] = np.array(aux)                
    return detp
    
def strategies_LHV(m_A,k_A,m_B,k_B):
    
    #m_A = nº de medições da Alice
    #k_A = nº de resultados da Alice
    #m_B = nº de medições do Bob
    #k_B = nº de resultados do Bob

    detp_A = strategies_LHS(m_A,k_A)
    detp_B = strategies_LHS(m_B,k_B)

    detp = np.kron(detp_A,detp_B)

    return detp

def SDP_LHV(k_A,m_A,k_B,m_B,rho,rho_sep,eta_A,eta_B,detp,mea_A,mea_B):

    P = pic.Problem()

    q = pic.RealVariable('q')

    chi = pic.HermitianVariable('chi',(4,4))

    sigma = [pic.HermitianVariable('Sigma_lambda[{}]'.format(i),(1,1)) for i in range((k_A**m_A)*(k_B**m_B))]

    rho_q = rho*q+(1-q)*rho_sep

    rho_eta = eta_A*eta_B*chi+eta_A*(1-eta_B)*pic.partial_trace(chi,subsystems=1,dimensions=(2,2))@(pic.partial_trace(rho_sep,subsystems=0, dimensions=(2,2))) + eta_B*(1-eta_A)*pic.partial_trace(rho_sep,subsystems=1,dimensions=(2,2))@(pic.partial_trace(chi,subsystems=0,dimensions=(2,2))) + (1-eta_A)*(1-eta_B)*pic.partial_trace(rho_sep,subsystems=1,dimensions=(2,2))@(pic.partial_trace(rho_sep,subsystems=0,dimensions=(2,2)))

    print(detp)
    print(detp[:,0])

    est_det = [pic.sum(detp[j,i*k]*sigma[j]for j in range((k_A**m_A)*(k_B**m_B))) for i in range(k_A*m_A) for k in range(k_B*m_B)]

    est = [(np.kron(mea_A[i],mea_B[j]))*chi for i in range(k_A*m_A) for j in range(k_B*m_B)]

    P.add_constraint(q<=1)

    P.add_constraint(q>=0)

    P.add_list_of_constraints([sigma[i]>=0 for i in range((k_A**m_A)*(k_B**m_B))]) 

    P.add_constraint(rho_q == rho_eta)

    P.add_list_of_constraints([pic.trace(est[i])==est_det[i] for i in range((k_A*m_A)*(k_B*m_B))])

    P.set_objective('max',q)

    solution = P.solve()

    return sigma, q, chi, solution, P

#Chamando as funções

rho, rho_sep = WernerClass()

medicoes_A, eta_A =  measurements(1)
medicoes_B, eta_B =  measurements(1)
k_A = 2
k_B = 2
m_A = int(medicoes_A.shape[0]/2)
m_B = int(medicoes_B.shape[0]/2)
print(m_A,m_B)
det_est = strategies_LHV(m_A,k_A,m_B,k_B)

#Aplicar SDP

p_lambda, q, chi, solution, prob = SDP_LHV(k_A,m_A,k_B,m_B,rho,rho_sep,eta_A,eta_B,det_est,medicoes_A,medicoes_B)

print('Problema')
print(prob)
print('Solução do problema')
#print('p_lambda',p_lambda)
print('q', q)
#print('q.shape',q.shape)
#print('chi',chi)
