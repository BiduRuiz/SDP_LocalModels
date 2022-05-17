print("Primeiro código LHV para estados de 2 qubits")
#Bibliotecas necessárias:
import numpy as np #Usar arrays
import math #Usar sqrt
import polyhedron as pol #Vértices e plot dos poliedros
import estrategias as est #Estratégias determinísticas
import SDP as sdp #SDP LHS
import WernerClass as wc #Criando o estado

#Primeira etapa: Estado alvo -> classe de estados de Werner de dois qubits

rho, rho_sep = wc.WernerClass()

#Segunda etapa: Mapa ruidoso

#xi = tr_B(rho_sep)

#MEDIÇÕES ALICE

#Terceira etapa: Nº de medições = 6 e resultados = 2
m_A = 6
k_A = 2
#Quarta etapa: Poliedro ótimo = icosaedro 12 vértices
vert_A = pol.icosaedro()
hull_A, eta_A = pol.polyhedron(vert_A)

medicoes_A = np.zeros([12,2,2], dtype=complex)

sum_med_A = np.zeros([2,2])

for i in range(12):
    med_00 = (1+vert_A[i][2])/2
    med_01 = (vert_A[i][0]-vert_A[i][1]*1j)/2
    med_10 = (vert_A[i][0]+vert_A[i][1]*1j)/2
    med_11 = (1-vert_A[i][2])/2
    
    medicoes_A[i] = [[med_00,med_01],[med_10,med_11]]
    
    #Garantir que a soma por medição é a matriz identidade!
    sum_med_A = sum_med_A + medicoes_A[i]

#print(sum_med_A)

#print(vert_A,eta_A)

#MEDIÇÕES BOB

#Terceira etapa: Nº de medições = 6 e resultados = 2
m_B = 6
k_B = 2
#Quarta etapa: Poliedro ótimo = icosaedro 12 vértices
vert_B = pol.icosaedro()
hull_B, eta_B = pol.polyhedron(vert_B)

medicoes_B = np.zeros([12,2,2], dtype=complex)

sum_med_B = np.zeros([2,2])

for i in range(12):
    med_00 = (1+vert_B[i][2])/2
    med_01 = (vert_B[i][0]-vert_B[i][1]*1j)/2
    med_10 = (vert_B[i][0]+vert_B[i][1]*1j)/2
    med_11 = (1-vert_B[i][2])/2
    
    medicoes_B[i] = [[med_00,med_01],[med_10,med_11]]
    
    #Garantir que a soma por medição é a matriz identidade!
    sum_med_B = sum_med_B + medicoes_B[i]

#print(sum_med_B)

#print(vert_B,eta_B)

#Quinta etapa: Montar as estratégias determinísticas

det_est = est.strategies_LHV(m_A,k_A,m_B,k_B)

#Sexta etapa: Aplicar SDP

p_lambda, q, chi, solution, prob = sdp.SDP_LHV_alpha(k_A,m_A,k_B,m_B,rho,rho_sep,eta_A,eta_B,det_est,medicoes_A,medicoes_B)

print('Problema')
print(prob)
print('Solução do problema')
#print('p_lambda',p_lambda)
print('q', q)
#print('q.shape',q.shape)
print('chi',chi)
