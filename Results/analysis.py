import numpy as np

num_lines = sum(1 for line in open('info.txt'))

f1 = open("info.txt", "r")

states = np.zeros(3)

for i in range(num_lines):
    a = f1.readline()
    a = a.removesuffix("\n") 
    b = np.array(a.split(" ")).astype(np.float64)
    
    if b[1]==0:
        states[0] = states[0]+1
    else:
        if b[3]==0:
            states[1] = states[1]+1
        else:
            states[2] = states[2]+1

print("Separable states: ",int(states[0]))
print("Non-local Entangled states: ",int(states[1]))
print("Local entangled states: ",int(states[2]))
        