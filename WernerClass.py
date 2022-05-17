import numpy as np 
import math #Usar sqrt

def WernerClass():

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
	
	return rho, rho_sep

