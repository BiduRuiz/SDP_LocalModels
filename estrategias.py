#Montar todas as possíveis estratégias 

import numpy as np

def strategies(m,k):

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

#est1 = strategies(2,2)

#print(est1)
#print(len(est1))

#est_2 = strategies(6,2)
#print(est_2[0][1])
#print(est_2.shape)
