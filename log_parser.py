import sys
sys.stdout=open('output_CC_180712.csv','w')
import os
import re
from string import ascii_lowercase, ascii_uppercase

pattern = ascii_lowercase + ascii_uppercase + '_,+\n'
path_log = './checkpoints_CC/experiment_name'

if __name__ == '__main__':
    #print('pattern = ',pattern)
    filename = os.path.join(path_log,'loss_log.txt')
    print('G_GAN,G_L1,D_loss,\n')
    f = open(filename,'r')
    line = f.readline()
    while line != '':
        line = f.readline()
        #print(line)
        l = line.split(':')[4:]
        if len(l) < 3: continue
        m=[]
        #print(l)
        m.append(l[0][1:-5])
        m.append(l[1][1:-12])
        m.append(l[2][1:-2])
        '''
        for x in l:
            x.lstrip()
            x.rstrip()
            re.sub(pattern,'',x)
            m.append(x)
        '''
        for x in m:
            print(x,',',end='',sep='')
        print('')