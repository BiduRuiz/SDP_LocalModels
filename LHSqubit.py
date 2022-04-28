print("Primeiro código LHS para estados de 2 qubits")
#Bibliotecas necessárias:
import numpy as np #Usar arrays
import math #Usar sqrt
import polyhedron as pol #Vértices e plot dos poliedros
import estrategias as est #Estratégias determinísticas
import SDP_LHS as sdp #SDP LHS

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

#Segunda etapa: Mapa ruidoso

#xi = tr_B(rho_sep)

#Terceira etapa: Nº de medições = 6 e resultados = 2

m = 6
k = 2

#Quarta etapa: Poliedro ótimo = icosaedro 12 vértices

vert = pol.icosaedro()
hull, eta = pol.polyhedron(vert)

#print(vert)
#print(vert[1])
#print(vert[1][0])
medicoes = np.zeros([12,2,2], dtype=complex)

#print(medicoes)

sum_med = np.zeros([2,2])

for i in range(12):
    med_00 = (1+vert[i][2])/2
    med_01 = (vert[i][0]-vert[i][1]*1j)/2
    med_10 = (vert[i][0]+vert[i][1]*1j)/2
    med_11 = (1-vert[i][2])/2
    
    medicoes[i] = [[med_00,med_01],[med_10,med_11]]
    
    #Garantir que a soma por medição é a matriz identidade!
    sum_med = sum_med + medicoes[i]

#print(sum_med)

#print(vert,eta)

#Quinta etapa: Montar as estratégias determinísticas

det_est = est.strategies_LHS(m,k)

#Sexta etapa: Aplicar SDP
print('Variáveis do problema')
print('k:',k)
print('m:',m)
print('rho:',rho)
print('rho.shape:',rho.shape)
print('rho_sep:',rho_sep)
print('rho_sep.shape:',rho_sep.shape)
print('eta:',eta)
#print('det_est:',det_est)
print('det_set.shape:',det_est.shape)
print('medicoes:',medicoes)
print('medicoes.shape:',medicoes.shape)
#print('sum_med:',sum_med)
#print('sum_med.shape:',sum_med.shape)
#pol.plot_polyhedron(vert)

sigma, q, chi, solution, prob, k_out,m_out,rho_out,rho_sep_out,eta_out,det_est_out,mea_out= sdp.SDP_LHS(k,m,rho,rho_sep,eta,det_est,medicoes)

print('Problema')
print(prob)
print('Solução do problema')
print('sigma',sigma)
print('q', q)
print('q.shape',q.shape)
print('chi',chi)
print('chi.shape',chi.shape)
print('solution:', solution)
print('Constantes do problema')
print('k:',k_out)
print('m:',m_out)
print('rho:',rho_out)
print('rho_sep:',rho_sep_out)
print('eta:',eta_out)
#print('det_est:',det_est_out)
print('medicoes:',mea_out[1])
