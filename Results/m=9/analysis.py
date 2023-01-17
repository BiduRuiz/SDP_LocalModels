import numpy as np

f2 = open('fractions.txt','w')
f3 = open('summary.txt','w')

for i in [1500]:

    num_lines = sum(1 for line in open('info_'+str(i)+'.txt'))

    f1 = open('info_'+str(i)+'.txt', 'r')

    states = np.zeros(5)

    for i in range(num_lines):
        a = f1.readline()
        a = a.removesuffix("\n") 
        b = np.array(a.split(" ")).astype(np.float64)
        
        if b[1]==1:
            states[4] = states[4]+1

        if round(b[2],4)==1:
            states[3] = states[3]+1
            if b[1]==0:
                states[0] = states[0]+1
            else:
                if b[3]==0:
                    states[1] = states[1]+1
                else:
                    states[2] = states[2]+1

    output_0 = ("For n = "+str(i+1)+" Total states with alpha = 1 "+str(states[3])+" Total entangled states "+str(states[4])+"\n")
    output_1 = ("Separable states: ",str(int(states[0])),"(",str(100*int(states[0])/num_lines),"%)","\n")
    output_2 = ("Non-local Entangled states: ",str(int(states[1])),"(",str(100*int(states[1])/num_lines),"%)","\n")
    output_3 = ("Local entangled states: ",str(int(states[2])),"(",str(100*int(states[2])/num_lines),"%)","(",str(100*int(states[2])/states[4]),"%)","\n")
    f2.writelines(output_0)
    f2.writelines(output_1)
    f2.writelines(output_2)
    f2.writelines(output_3)
    f2.writelines(str('-'*84)+"\n")
    f3.write(str(i+1)+' '+str(100*int(states[0])/num_lines)+' '+str(100*int(states[1])/num_lines)+' '+str(100*int(states[2])/num_lines)+'\n')

    f1.close()

f2.close()
f3.close()
            