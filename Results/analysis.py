import numpy as np

f2 = open('fractions.txt','w')
f3 = open('summary.txt','w')

for i in [1,10,50,100,500,1000,5000,10000,50000,100000,1000000]:

    num_lines = sum(1 for line in open('info_'+str(i)+'.txt'))

    f1 = open('info_'+str(i)+'.txt', 'r')

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

    output_0 = ("For n = "+str(i+1)+"\n")
    output_1 = ("Separable states: ",str(int(states[0])),"(",str(100*int(states[0])/num_lines),"%)","\n")
    output_2 = ("Non-local Entangled states: ",str(int(states[1])),"(",str(100*int(states[1])/num_lines),"%)","\n")
    output_3 = ("Local entangled states: ",str(int(states[2])),"(",str(100*int(states[2])/num_lines),"%)","\n")
    f2.writelines(output_0)
    f2.writelines(output_1)
    f2.writelines(output_2)
    f2.writelines(output_3)
    f2.writelines(str('-'*84)+"\n")
    f3.write(str(i+1)+' '+str(100*int(states[0])/num_lines)+' '+str(100*int(states[1])/num_lines)+' '+str(100*int(states[2])/num_lines)+'\n')

    f1.close()

f2.close()
f3.close()
            