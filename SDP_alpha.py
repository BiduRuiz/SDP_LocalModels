import numpy as np
import picos as pic

def SDP_LHV_alpha(k_out_A,m_mea_A,k_out_B,m_mea_B,rho_ent,rho_loc,eta_in_A,eta_in_B,estrategias,medicoes_A,medicoes_B):

    #Criando o SDP
    prob = pic.Problem()
    
    #Definindo as constantes
    k_A = pic.Constant('k_A',k_out_A)
    m_A = pic.Constant('m_A',m_mea_A)
    k_B = pic.Constant('k_B',k_out_B)
    m_B = pic.Constant('m_B',m_mea_B)
    rho = pic.Constant('rho',rho_ent)
    rho_sep = pic.Constant('rho_sep',rho_loc)
    eta_A = pic.Constant('eta_A',eta_in_A)
    eta_B = pic.Constant('eta_B',eta_in_B)
    est = pic.Constant('est_det',estrategias)
    mea_A = [pic.Constant('meas_A[{}]'.format(i),medicoes_A[i])for i in range(k_A*m_A)]
    mea_B = [pic.Constant('meas_B[{}]'.format(i),medicoes_B[i])for i in range(k_B*m_B)]
    
    #Definindo as variáveis que queremos encontrar
    q = pic.RealVariable('q')
    p_lambda = [prob.add_variable('p_lambda[{}]'.format(i), (1, 1)) for i in range((k_A**m_A)*(k_B**m_B))]
    chi = pic.HermitianVariable('chi',(4,4))
    
    #Objetivo do SDP: q* = max(q)
    prob.set_objective('max',q)
    
    #Definindo as restrições das variáveis
    prob.add_constraint(q>=0)
    prob.add_constraint(q<=1)
    prob.add_list_of_constraints([p_lambda[i]>=0 for i in range((k_A**m_A)*(k_B**m_B))])
    
    #Definindo as outras restrições
    prob.add_constraint(q*rho+(1-q)*rho_sep == eta_A*eta_B*chi+eta_A*(1-eta_B)*pic.partial_trace(chi,subsystems=(1),dimensions=(2,2))@(pic.partial_trace(rho_sep,subsystems=(0), dimensions=(2,2))) + eta_B*(1-eta_A)*pic.partial_trace(rho_sep,subsystems=(1),dimensions=(2,2))@(pic.partial_trace(chi,subsystems=(0),dimensions=(2,2))) + (1-eta_A)*(1-eta_B)*pic.partial_trace(rho_sep,subsystems=(1),dimensions=(2,2))@(pic.partial_trace(rho_sep,subsystems=(0),dimensions=(2,2))))
    
    prob.add_list_of_constraints([[([pic.trace((mea_A[i]@(mea_B[k]))*chi)==pic.sum([est[j][i*k]*p_lambda[j]for j in range((k_A**m_A)*(k_B**m_B))]) for i in range(k_A*m_A)])] for k in range(k_B*m_B)])

    #Resolvendo o SDP
    solution = prob.solve()
    
    return p_lambda, q, chi, solution, prob
    
def SDP_LHS_alpha(k_out,m_mea,rho_ent,rho_loc,eta_in,estrategias,medicoes):


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
    chi = pic.ComplexVariable('chi',(4,4))
    #Definindo o amsenble {Sigma}_lambda que queremos encontrar
    sigma = [prob.add_variable('sigma[{}]'.format(i), (2, 2), 'hermitian') for i in range(k**m)]
    
    #Objetivo do SDP: q* = max(q)
    prob.set_objective('max',q)
    
    #Definindo as restrições das variáveis
    prob.add_constraint(q>=0)
    prob.add_constraint(q<=1)
    prob.add_list_of_constraints([sigma[i]>>0 for i in range(k**m)])
    
    #Definindo as outras restrições
    prob.add_constraint(q*rho+(1-q)*rho_sep == eta*chi+(1-eta)*(((pic.partial_trace(rho_sep,subsystems=(1),dimensions=(2,2))))@(pic.partial_trace(chi,subsystems=(0),dimensions=(2,2)))))
    prob.add_list_of_constraints([pic.partial_trace((mea[i]@(iMat))*chi,subsystems=(0),dimensions=(2,2)) == pic.sum([estrategias[j][i]*sigma[j]for j in range(k**m)]) for i in range(k*m)])

    #Resolvendo o SDP
    prob.options.solver = 'mosek'
    solution = prob.solve()
    
    return sigma, q, chi, solution, prob
