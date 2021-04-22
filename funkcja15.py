import numpy as np

def funkcja15 (x):
    a=np.size(x)
    #print(a)
    y2=np.arange(1,(a+1))
    for i in range(a):
        if (x[i]<0):
            y2[i]=np.sin(x[i])
            #print(y2[i])
        if (x[i]>=0):
            y2[i]=np.sqrt(x[i])
    return y2