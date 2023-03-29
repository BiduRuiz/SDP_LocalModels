import functions as fc
import numpy as np
import time

start_time = time.time()

# Creating n states, with dimension d
n = 1
d = 4

# Creating the Werner State

ket_00 = np.array([[1],[0],[0],[0]])
ket_01 = np.array([[0],[1],[0],[0]])
ket_10 = np.array([[0],[0],[1],[0]])
ket_11 = np.array([[0],[0],[0],[1]])
#Create |psi> = (|01>-|10>)/sqrt(2) and rho = |psi><psi|
psi = (ket_01-ket_10)/np.sqrt(2)
rho = psi@np.transpose(psi)

# Creating the separable state
rho_sep = (np.eye(4))/4

# # Creating m measurements, each with 2 results
# # medicoes, eta = fc.pentakis(PLOT=False)
# medicoes, eta = fc.measurements(1,PLOT=False)
# m = int(medicoes.shape[0]/2)

# print(m,eta)

# # Creating the deterministic strategies
# detp = fc.strategies_LHS(m,2)

# # Creating the file to write the results
# f1 = open("results_"+str(n)+".txt", "w")
# f2 = open("rho_"+str(n)+".txt", "w")

# f1.write("Density matrix index"+"\t"+"Entanglement"+"\t"+"Locality\n")
# f2.write("Density matrix\n")

# for i in range(n):

#     # Using the functions created (Bures and Haar)

#     # rho = fc.rho_mixed(d)
    
#     w,v,ppt = fc.Ent_cert(rho)

#     P,solution,q = fc.SDP_LHS(m,2,rho,rho_sep,eta,detp,medicoes)

#     f2.writelines(str(rho)+"\n")
#     f1.writelines(str(i)+"\t"+str(ppt)+"\t"+str(q)+"\n")

# print(time.time() - start_time, "seconds")

# # Closing the files
# f1.close()
# f2.close()