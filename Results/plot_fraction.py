import numpy as np
import matplotlib.pyplot as plt

num_lines = sum(1 for line in open('summary.txt'))

f1 = open('summary.txt', 'r')

results = np.zeros((num_lines,4))

for i in range(num_lines):
    a = f1.readline()
    a = a.removesuffix("\n") 
    b = np.array(a.split(" ")).astype(np.float64)

    results[i] = b


fig, ax = plt.subplots(3,1)

ax[0].grid()

ax[0].scatter(results[:,0], results[:,1],marker='x')
ax[1].scatter(results[:,0], results[:,2],marker='.')
ax[2].scatter(results[:,0], results[:,3],marker='+')
ax[0].xscale('log')

ax[0].set(ylabel='%',
       title='Volume of states')
ax[1].set(ylabel='%')
ax[2].set(xlabel='log(n)', ylabel='%')

#fig.savefig("test.png")
plt.show()

