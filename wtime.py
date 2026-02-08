# Speed test: computing the determinant recursively or with LU decomposition.
# By L van Veen, Ontario Tech U, 2024.
import numpy as np
import time                              # For timing
import matplotlib.pyplot as plt
from LUP import LUP
from deter import deter

# Consider matrices of size n X n where n=2^istart .. 2^iend with recursive computation:
istart = 2
iend = 3
wtimerec=np.zeros((iend-istart+1,2))       
for i in range(istart,iend+1):
    n = 2**i
    A = np.random.rand(n,n)               # initialize random matrix
    start=time.time()                        # start timer
    detA = deter(A)                          # computation using cofactor expansion
    elapsed=time.time()-start                # measure elapsed time
    print("n=%d wall time=%f det=%e" %(n,elapsed,detA))         # output value of determinant
    wtimerec[i-istart,0]=n                   # store data for plotting
    wtimerec[i-istart,1]=elapsed

# Consider matrices of size n X n where n=2^istart .. 2^iend with LUP decomposition:
istart = 2
iend = 8
wtimeLUP=np.zeros((iend-istart+1,2))       
for i in range(istart,iend+1):
    n = 2**i
    A = np.random.rand(n,n)               # initialize random matrix
    start = time.time()                        # start timer
    L,U,P,sig,ok = LUP(A)                    # LU-decompose A=P^t LU
    if ok==1:
        detA = sig * np.prod(np.diag(U)) # compute determinant from diagonal of U
        elapsed = time.time()-start            # measure elapsed time
        print("n=%d wall time=%f det=%e" %(n,elapsed,detA))         # output value of determinant
        wtimeLUP[i-istart,0]=n                # store data for plotting
        wtimeLUP[i-istart,1]=elapsed
    else:
        print("Near-zero pivot in LU decomposition for n=%d\n" % (n))

plt.loglog(wtimerec[:,0],wtimerec[:,1],'-*r', label='recursive')
plt.loglog(wtimeLUP[:,0],wtimeLUP[:,1],'-*g', label='using LUP')
plt.loglog(wtimeLUP[:,0],1e-6*wtimeLUP[:,0]**3,'-k', label='n^3')
plt.legend()
plt.show()
