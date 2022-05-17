print("Primeiro código LHS para estados de 2 qubits")
import picos as pic

print('Solvers supported on this installation of PICOS:',pic.solvers.all_solvers().keys())

print('Solvers available to PICOS on this machine:',pic.solvers.available_solvers())

#Bibliotecas necessárias:
import numpy as np #Usar arrays
import math #Usar sqrt
#import polyhedron as pol #Vértices e plot dos poliedros
import poly as pol #Vértices, plot dos poliedros e medicoes
import estrategias as est #Estratégias determinísticas
import SDP_LHS as sdp #SDP LHS
import WernerClass as wc #Criando o estado

#Primeira etapa: Estado alvo -> classe de estados de Werner de dois qubits

rho, rho_sep = wc.WernerClass()

#Segunda etapa: Mapa ruidoso

#xi = tr_B(rho_sep)

#Terceira etapa: Criando loop de medições

k = 2

n_ciclos = 1

#solutions = np.zeros(n_ciclos)
etas = np.zeros(n_ciclos)
chis = np.zeros([n_ciclos,4,4])

index = 0

for n in range(n_ciclos):

    vert = pol.vertices(n+1)
    hull, eta = pol.polyhedron(vert)
    #pol.plot_polyhedron(vert)
    
    etas[index] = eta

    m_k = vert.shape[0]
    medicoes, sum_med = pol.medicoes(vert)

    #Quinta etapa: Montar as estratégias determinísticas

    m = int(m_k/k)
    det_est = est.strategies_LHS(m,k)

    #Sexta etapa: Aplicar SDP
    
    sigma, chi, solution, prob = sdp.SDP_LHS(k,m,rho,rho_sep,eta,det_est,medicoes)
    
    #solutions[index] = q
    chis[index] = chi
    index = index+1
    
    print('Problema')
    print(prob)
    print('solution:', solution)
    
for i in range(n_ciclos):
    #print('q:',solutions[i],'eta:',etas[i],'chi:',chis[i])
    print('eta:',etas[i],'chi:',chis[i],'sigma:',sigma[0])
