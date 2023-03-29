import functions as fc
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
#sns.set_theme(style="darkgrid",palette='rocket')

# print(sns.color_palette("rocket").as_hex())
# Colors of rocket: ['#35193e', '#701f57', '#ad1759', '#e13342', '#f37651', '#f6b48f']

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

# Palette options: cubehelix, Spectral, deep, rocket

# # First plot: Werner state

# r = [0.5773502691896258, 0.8164965809277259, 0.8628562094610168]
# q = [0.3333333367378584, 0.42040144526522283, 0.4407177404009949]
# n = [" m = 3\n 0.24s", " m = 9\n 5.81s", " m = 13\n 352.33s"]

# c = np.linspace(0,1,20)
# d = np.zeros(20)+0.5

# fig, ax = plt.subplots()

# for i in range(len(n)):
#     ax.annotate(n[i], (r[i], q[i]),xytext = (r[i]+0.01, q[i]-0.01))

# ax.scatter(r, q,color = '#f37651', edgecolor = '#35193e')
# ax.plot(c,d,color='#ad1759')

# plt.text(0.85,0.505, 'Analytical result')

# ax.set_ylabel('Lower bound for locality',fontsize=16)
# ax.set_xlabel('Radius of the inner sphere',fontsize=16)

# ax.set_xlim(0.5, 1)  # de 0 até 2 na horizontal
# ax.set_ylim(0.3, 0.52)  # de 0 até 4 na vertical

# ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

# ax.set_title('Werner state',fontsize=20)
# # ax.legend()
# # ax.grid()

# # ax.axis('equal')   # iguala as escalas horizontal e vertical

# # Salva um PNG de alta qualidade. Um pouco melhor para impressão
# fig.savefig('Werner.png', transparent=True, dpi=300, bbox_inches='tight')

# # Salva a figura em PDF. O DPI é irrelevante: os elementos do gráfico são
# # vetorizados, ficam com "resolução infinita". É o melhor para a maioria das
# # impressões. Não é uma boa ideia se houver transparências na figura (PDF 
# # não suporta transparências bem)
# fig.savefig('Werner.pdf', bbox_inches='tight') 

# # Segundo plot: Politopos das medições

# for i in range(3): med, eta = fc.measurements(1+i,PLOT=True)

# # Teste

# x = np.linspace(0, 2, 21)

# plt.plot(x, x, '--', label='linear')
# plt.plot(x, x**2, 'o', label='quadrático')
# plt.plot(x, x**3, '^', label='cúbico')

# plt.xlabel('teste x')
# plt.ylabel('teste y')

# plt.xlim(0, 2)  # de 0 até 2 na horizontal
# plt.ylim(0, 8)  # de 0 até 4 na vertical

# plt.title('Gráfico simples')
# plt.legend()
# # plt.grid()

# # plt.axis('equal')   # iguala as escalas horizontal e vertical

# # Salva um PNG de alta qualidade. Um pouco melhor para impressão
# plt.savefig('figura.png', dpi=300, bbox_inches='tight')

# # Salva a figura em PDF. O DPI é irrelevante: os elementos do gráfico são
# # vetorizados, ficam com "resolução infinita". É o melhor para a maioria das
# # impressões. Não é uma boa ideia se houver transparências na figura (PDF 
# # não suporta transparências bem)
# plt.savefig('figura.pdf', bbox_inches='tight') 

