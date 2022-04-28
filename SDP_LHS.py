import numpy as np
import picos as pic

def SDP_LHS(k_out,m_mea,rho_ent,rho_loc,eta_in,estrategias,medicoes):


    #Criando o SDP
    prob = pic.Problem()

    #Definimos agora todas as variáveis e constantes dentro do PICOS
    #Constantes:
    k = pic.Constant('k',k_out)
    m = pic.Constant('m',m_mea)
    rho = pic.Constant('rho',rho_ent)
    rho_sep = pic.Constant('rho_sep',rho_loc)
    eta = pic.Constant('eta',eta_in)
    est = pic.Constant('est_det',estrategias)
    mea = [pic.Constant('meas[{}]'.format(i),medicoes[i])for i in range(k*m)]
    #Matrix identidade 2x2:
    iMat = pic.Constant('I', np.eye(2,dtype='complex'))
    
    #Variáveis
    q = pic.RealVariable('q')
    chi = pic.HermitianVariable('chi',(4,4))
    #Definindo o amsenble {Sigma}_lambda que queremos encontrar
    sigma = [prob.add_variable('sigma[{}]'.format(i), (2, 2), 'hermitian') for i in range(k**m)]
    
    #Objetivo do SDP: q* = max(q)
    prob.set_objective('max',q)
    
    #Definindo as restrições das variáveis
    prob.add_constraint(q>=0)
    prob.add_constraint(q<=1)
    prob.add_list_of_constraints([sigma[i]>>0 for i in range(k**m)])
    
    #Definindo as outras restrições
    prob.add_constraint(q*rho+(1-q)*rho_sep == eta*chi+(1-eta)*iMat@(pic.partial_trace(chi,subsystems=(1),dimensions=(2,2))))
    prob.add_list_of_constraints([pic.partial_trace(mea[i]@(pic.partial_trace(rho_sep,subsystems=(1),dimensions=(2,2)))*(chi),subsystems=(0),dimensions=(2,2))==pic.sum([estrategias[j][i]*sigma[j]for j in range(k**m)]) for i in range(k*m)])

    #Resolvendo o SDP
    prob.options.solver = 'mosek'
    solution = prob.solve()
    
    return sigma, q, chi, solution, prob, k, m, rho, rho_sep, eta, est, mea
