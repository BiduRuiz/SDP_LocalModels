import numpy as np
import picos as pc

def SDP_LHS(k,m,rho,rho_sep,eta,estrategias,medicoes,identidade):

    #Estado de 2 qubits
    qub = 4
    #Identidade dim = 2
    iden_2 = np.array([[1,0],[0,1]]) 
    
    #Criando o SDP
    prob = pc.Problem()
    
    #Definindo as variáveis que queremos encontrar
    q = pc.RealVariable('q',1)
    sigma = [prob.add_variable("sigma[{}]".format(i), (2, 2)) for i in range(k**m)]
    chi = pc.HermitianVariable('chi',qub)
    
    #Objetivo do SDP: q* = max(q)
    prob.set_objective('max',q)
    
    #Definindo as restrições das variáveis
    prob.add_constraint(q>=0)
    prob.add_constraint(q<=1)
    prob.add_list_of_constraints([sigma[i]>=0 for i in range(k**m)],'i','0...k**m')
    
    #Definindo as outras restrições
    prob.add_constraint(q*rho+(1-q)*rho_sep == eta*chi+(1-eta)*pc.partial_trace(rho_sep,dim=4)@pc.partial_trace(chi, dim=4))
    prob.add_list_of_constraints([pc.partial_trace(np.kron(medicoes[i],iden_2)*chi, dim=4)==pc.sum([estrategias[j][i]*sigma[j]for j in range(k**m)]) for i in range(k*m)],'i','0...k*m')

    #Resolvendo o SDP
    prob.options.solver = 'gurobi'
    solution = prob.solve()
    
    return sigma, q, chi, solution, prob
