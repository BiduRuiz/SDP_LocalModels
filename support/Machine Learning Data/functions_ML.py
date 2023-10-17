# Construção do dataset em dataframe do Pandas

import pandas as pd
import numpy as np

def txt2dataframe(arquivo_txt):
  """ Essa função recebe o arquivo txt de resultados de input e retorna um dataframe do pandas """
  united = pd.read_csv(arquivo_txt,dtype='str', header=None)
  even_rows = united.iloc[::2]
  odd_rows = united.iloc[1::2]

  even_rows = even_rows.reset_index()
  odd_rows = odd_rows.reset_index()

  united = pd.concat([even_rows, odd_rows], axis=1)

  united = united.drop(['index'], axis=1)

  even_rows = united.iloc[::2]
  odd_rows = united.iloc[1::2]

  even_rows = even_rows.reset_index()
  odd_rows = odd_rows.reset_index()

  united = pd.concat([even_rows, odd_rows], axis=1)

  united = united.drop(['index'], axis=1)

  even_rows = united.iloc[::2]
  odd_rows = united.iloc[1::2]

  even_rows = even_rows.reset_index()
  odd_rows = odd_rows.reset_index()

  united = pd.concat([even_rows, odd_rows], axis=1)

  united = united.drop(['index'], axis=1)

  united.columns = ['a_1 a_2','a_3 a_4','a_5 a_6','a_7 a_8','a_9 a_10','a_11 a_12','a_13 a_14','a_15 a_16']

  united['a_1 a_2'] = united['a_1 a_2'].str.replace('[', '')
  united['a_3 a_4'] = united['a_3 a_4'].str.replace(']', '')
  united['a_5 a_6'] = united['a_5 a_6'].str.replace('[', '')
  united['a_7 a_8'] = united['a_7 a_8'].str.replace(']', '')
  united['a_9 a_10'] = united['a_9 a_10'].str.replace('[', '')
  united['a_11 a_12'] = united['a_11 a_12'].str.replace(']', '')
  united['a_13 a_14'] = united['a_13 a_14'].str.replace('[', '')
  united['a_15 a_16'] = united['a_15 a_16'].str.replace(']', '')

  united['a_1 a_2'] = united['a_1 a_2'].str.replace(' ', '')
  united['a_3 a_4'] = united['a_3 a_4'].str.replace(' ', '')
  united['a_5 a_6'] = united['a_5 a_6'].str.replace(' ', '')
  united['a_7 a_8'] = united['a_7 a_8'].str.replace(' ', '')
  united['a_9 a_10'] = united['a_9 a_10'].str.replace(' ', '')
  united['a_11 a_12'] = united['a_11 a_12'].str.replace(' ', '')
  united['a_13 a_14'] = united['a_13 a_14'].str.replace(' ', '')
  united['a_15 a_16'] = united['a_15 a_16'].str.replace(' ', '')

  new_1 = united['a_1 a_2'].str.split("j", expand = True)
  # new_1[0] = new_1[0].astype(str) + 'j'
  new_2 = united['a_3 a_4'].str.split("j", expand = True)
  # new_2[0] = new_2[0].astype(str) + 'j'
  new_3 = united['a_5 a_6'].str.split("j", expand = True)
  # new_3[0] = new_3[0].astype(str) + 'j'
  new_4 = united['a_7 a_8'].str.split("j", expand = True)
  # new_4[0] = new_4[0].astype(str) + 'j'
  new_5 = united['a_9 a_10'].str.split("j", expand = True)
  # new_5[0] = new_5[0].astype(str) + 'j'
  new_6 = united['a_11 a_12'].str.split("j", expand = True)
  # new_6[0] = new_6[0].astype(str) + 'j'
  new_7 = united['a_13 a_14'].str.split("j", expand = True)
  # new_7[0] = new_7[0].astype(str) + 'j'
  new_8 = united['a_15 a_16'].str.split("j", expand = True)
  # new_8[0] = new_8[0].astype(str) + 'j'

  new = pd.concat([new_1,new_2,new_3,new_4,new_5,new_6,new_7,new_8], axis=1)

  new = new.replace('',None)
  new.columns = np.arange(len(new.columns))

  value = False

  for i in range(len(new.columns)):
    value = new[i].isnull().values.all()
    if value == True:
      new = new.drop(i, axis=1)

  return new

def dataframe2array(dataframe):
  """ Essa função recebe o dataframe do pandas de input e retorna um array do numpy com as matrizes densidade """
  arr = dataframe.to_numpy()
  arr = arr.reshape(len(dataframe.columns)*len(dataframe.index))
  arr = list(filter(None, arr))
  arr = np.array([sub + 'j' for sub in arr])
  rhos = arr.reshape((len(dataframe.index),4,4))
  rhos = rhos.astype(complex)
  return rhos

def Input_ML_Complex(rhos):
  '''
  Montando o input do Machine learning COMPLEXO
  Essa função recebe o array com as matrizes densidade e gera dois arrays,
  um deles com a parte real
  outro deles com a parte imaginária
  '''
  Real_part = rhos.real
  Imag_part = rhos.imag
  return Real_part, Imag_part

# Montando o input do Machine learning REAL

def Input_ML_Real(rhos):
  '''
  Montando o input do Machine learning REAL
  Essa função recebe o array com as matrizes densidade e gera três arrays,
  são as decomposições do tipo 
  rho = (1/4)[I\otimes I
  + r\cdot\vec{\sigma}\otimes I 
  + I \otimes s \cdot\vec{\sigma} 
  + \sum_{n,m=1}^3 t_{nm}\sigma_n\otimes\sigma_m]
  '''
  r = np.zeros((rhos.shape[0],3))
  s = np.zeros((rhos.shape[0],3))
  T_matrix = np.zeros((rhos.shape[0],3,3))

  Pauli_matrices = np.array([[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]])

  for j in range(rhos.shape[0]):
    for i in range(3):
      r[j][i] = np.trace((np.kron(Pauli_matrices[i],np.eye(2)))@(rhos[j]))
      s[j][i] = np.trace((np.kron(np.eye(2),Pauli_matrices[i]))@(rhos[j]))
    for i in range(3):
      for k in range(3):
        T_matrix[j][i][k] = np.trace((np.kron(Pauli_matrices[i],Pauli_matrices[k]))@(rhos[j]))
  return r,s,T_matrix

def classification_LE(arquivo_txt):
  """ Essa função recebe o arquivo txt de resultados de input e retorna um dataframe do pandas """
  classes = pd.read_csv(arquivo_txt,dtype='str', header=None)

  return classification

def classification_LE(arquivo_txt):
  """ Essa função recebe o arquivo txt de resultados de input e retorna um dataframe do pandas """

  classes = pd.read_csv(arquivo_txt, dtype = str, header=None)

  classes = classes[0].str.split(' ', expand=True)

  classes = classes.drop(columns=0)

  classes.columns = ["Entanglement", "Locality", "Entanglement after"]

  classes = classes.astype(float)

  output = np.zeros(classes.shape[0])

  for i in range(classes.shape[0]):
    if classes['Entanglement'][i] == 1 and classes['Locality'][i] >= 1-10**(-4):
      output[i] = 1
  return output


def complete_dataset(r,s,T_matrix,output):

  complete_data = np.column_stack((r, s, T_matrix.reshape(T_matrix.shape[0],9),output.T))

  dataset = pd.DataFrame(complete_data, columns = ['r_x','r_y','r_z','s_x','s_y','s_z','T_xx','T_xy','T_xz','T_yx','T_yy','T_yz','T_zx','T_zy','T_zz','Entangled and Local'])

  return dataset


states_txt = '/content/drive/Shareddrives/Imortal/Mestrado/Pesquisa/Machine Learning/Dataset/states_1000.txt'
dataframe = txt2dataframe(states_txt)
# display(dataframe)
rhos = dataframe2array(dataframe)
# print(rhos)
r,s,T_matrix = Input_ML_Real(rhos)


classes_txt = '/content/drive/Shareddrives/Imortal/Mestrado/Pesquisa/Machine Learning/Dataset/info_1000.txt'

output = classification_LE(classes_txt)

df = complete_dataset(r,s,T_matrix,output)
display(df)
