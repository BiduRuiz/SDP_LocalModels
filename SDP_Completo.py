import picos as pic
import numpy as np
import math #Usar sqrt
from scipy.spatial import ConvexHull #Medições


def states():
    #Definir kets |00>, |01>, |10> e |11>
    ket_00 = np.array([[1],[0],[0],[0]])
    ket_01 = np.array([[0],[1],[0],[0]])
    ket_10 = np.array([[0],[0],[1],[0]])
    ket_11 = np.array([[0],[0],[0],[1]])
    #Definir ket psi = (|01>-|10>)/sqrt(2) e rho = |psi><psi|
    psi = (ket_01-ket_10)/math.sqrt(2)
    rho = psi*np.transpose(psi)
    #Definir matriz densidade I/4
    rho_sep = (np.eye(4))/4
    return rho, rho_sep

#rho,rho_sep = states()
#print(rho)
#print(rho_sep)

def measurements(n):
    #n é o ciclo de criação de vértice

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

    m_k = vert.shape[0]
    medicoes = np.zeros([m_k,2,2], dtype=complex)

    for i in range(m_k):
        med_00 = (1+vert[i][2])/2
        med_01 = (vert[i][0]-vert[i][1]*1j)/2
        med_10 = (vert[i][0]+vert[i][1]*1j)/2
        med_11 = (1-vert[i][2])/2
    
        medicoes[i] = [[med_00,med_01],[med_10,med_11]]
    
    #Poliedro
    hull = ConvexHull(vert)
    #Insphere radius
    r = np.min(np.abs(hull.equations[:, -1]))

    return medicoes,r
    
#medicoes,r = measurements(1)
#print(medicoes)
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

def SDP_LHS(m,k,rho,rho_sep,eta,detp,medicoes):

    P = pic.Problem()

    q = pic.RealVariable('q')

    chi = pic.HermitianVariable('chi',(4,4))

    sigma = [pic.HermitianVariable('Sigma_lambda[{}]'.format(i),(2,2)) for i in range(k**m)]

    rho_q = rho*q+(1-q)*rho_sep

    rho_eta = eta*chi+(1-eta)*(pic.partial_trace(rho_sep,subsystems=1,dimensions=(2,2)))@(pic.partial_trace(chi,subsystems=0,dimensions=(2,2)))

    est_det = [pic.sum([sigma[j]*detp[j][i] for j in range(k**m)]) for i in range(k*m)]

    est = [(np.kron(medicoes[i],np.eye(2)))*chi for i in range(k*m)]
    
    P.add_constraint(q<=1)

    P.add_constraint(q>=0)

    P.add_list_of_constraints([sigma[i]>>0 for i in range(k**m)]) 

    P.add_constraint(rho_q == rho_eta)

    P.add_list_of_constraints([pic.partial_trace(est[i],subsystems=0,dimensions=(2,2))==est_det[i] for i in range(k*m)])

    P.set_objective('max',q)

    solution = P.solve()

    return P, solution, q, chi, sigma, rho_q, rho_eta, est_det, est

rho,rho_sep = states()
medicoes,r = measurements(1)
detp = strategies_LHS(3,2)

P,solution,q,chi,sigma_lambda,rho_q,rho_eta,est_det,est = SDP_LHS(3,2,rho,rho_sep,r,detp,medicoes)
print(P)
print(solution)
#print(solution.primals)
print(q)
#print(chi)
#print(sigma_lambda)
#print(rho_q)
#print(rho_eta)
#print(est_det)
#print(est)