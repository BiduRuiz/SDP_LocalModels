import numpy as np
import math

#Definir kets |01> e |10>
ket_01 = np.array([[0],[1],[0],[0]])
ket_10 = np.array([[0],[0],[1],[0]])
#Definir ket psi = (|01>-|10>)/sqrt(2)
psi = (ket_01-ket_10)/math.sqrt(2)
#Definir matriz densidade I/2
rho_sep = (np.eye(2))/2
#Definir ket |2>
ket_2 = np.array([[0],[0],[1]])
rho_2 = ket_2@np.transpose(ket_2)
#Definir visibilidade
q = 1/2

rho_E = q*(psi@np.transpose(psi))+(1-q)*(np.kron(rho_sep,rho_2))
print(rho_E)