#Montar todas as possíveis estratégias 

import numpy as np

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
#est1 = strategies_LHS(2,2)

#print(est1)
#print(est1[1])
#print(est1[1][1])
#print(len(est1))

#est_2 = strategies_LHV(2,2,2,2)
#print(est_2)
#print(est_2.shape)
