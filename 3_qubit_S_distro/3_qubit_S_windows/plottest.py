import pylab as py
import numpy as np
import time

##xlist=np.zeros(2000)
##ylist=np.zeros(2000)
##t1=time.time()
##for i in range(2000):
##    xlist[i]=i
##    ylist[i]=i
##py.scatter(xlist,ylist)
##t2=time.time()
##t3=(t1-t2)/60.0
##print t3," minutes"
##py.show()

t1=time.time()
for i in range(2000):
    q=float(i)
    py.scatter(q,q)

t2=time.time()
t3=(t1-t2)/60.0
print t3," minutes"
py.show()
