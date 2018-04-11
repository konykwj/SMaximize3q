#The Data Creation for the 3 quibt states.

import S_maximize_3q as S3
from math import cos
from math import sin
import time
import pylab as py
import numpy as np

S3.DC_custom_1p(0,25,500,psif,name)
S3.DC_custom_2p(0,25,500,psif,name)
S3.DC(1,0,10,200)


t1=time.time()
def psif(theta,phi):
    psif=2.0**-0.5*(cos(theta)*S3.lll+sin(theta)*S3.loo+cos(phi)*S3.olo+sin(phi)*S3.ool)
    return psif
name="topplate"
S3.DC_custom_2p(0,5,500,psif,name)
t2=time.time()
t3=(t2-t1)/60
print t3,"time for ",name," DC"

t1=time.time()
def psif(theta):
    psi=1.0/2.0**.5*(cos(theta)*(S3.lll+S3.loo)+sin(theta)*(S3.olo+S3.ool))
    return psi
name="Star"
S3.DC_custom_1p(0,25,500,psif,name)
t2=time.time()
t3=(t2-t1)/60
print t3,"time for ",name," DC"

