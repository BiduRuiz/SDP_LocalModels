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