#A program to run graphing and analysis routines.

import S_maximize_3q as S3
from math import cos
from math import sin
import time
import pylab as py
import numpy as np



def p1S_max_tgl_check(name,n,ln,save,clear,color):
    if clear==1:
        py.clf()
    filename=S3.pathname+"/Figures/"+name+"<S>max vs tgl"+str(n)+" , "+str(ln)+" .png"

    ylist=np.zeros(ln)
    xlist=np.zeros(ln)
    for i in range(ln):
        print i
        theta=float(i)*2*np.pi/(ln-1)
        psi,speciallist,maxlist,xoptalist=S3.p1call_file(name,n,i,ln)
        x=maxlist[0]
        Fmax=abs(S3.Fa(x,psi))
        Fone=4*(1+(sin(2*theta))**2)**.5
        Ftwo=4*(1+(sin(theta))**2)**.5
        print Fmax,"Actual Data"
        print Fone," GGHZ Tangle test"
        print Ftwo ," MS tangle test"
        ylist[i]=Fmax

        d1=(psi[0,0]**2)*(psi[7,0]**2)+(psi[1,0]**2)*(psi[6,0]**2)+(psi[2,0]**2)*(psi[5,0]**2)+(psi[4,0]**2)*(psi[3,0]**2)
        d2=psi[0,0]*psi[7,0]*psi[3,0]*psi[4,0]+psi[0,0]*psi[7,0]*psi[5,0]*psi[2,0]+psi[0,0]*psi[7,0]*psi[6,0]*psi[1,0]+psi[3,0]*psi[4,0]*psi[5,0]*psi[2,0]+psi[3,0]*psi[4,0]*psi[6,0]*psi[1,0]+psi[5,0]*psi[2,0]*psi[6,0]*psi[1,0]
        d3=psi[0,0]*psi[6,0]*psi[5,0]*psi[3,0]+psi[7,0]*psi[1,0]*psi[2,0]*psi[4,0]
        Tgl=4*abs(d1-2*d2+4*d3)
        xlist[i]=Tgl
        print Tgl,"Real Tangle"
        Tgl2=(sin(2*theta))**2
        print Tgl2," Theory tangle"

    lable=name
    py.scatter(xlist,ylist,c=color,label=lable)
    
    py.xlabel('Three Tangle')
    py.ylabel('<S> Max')
    py.legend()
    if save==1:
        py.savefig(filename,dpi=200)
    
    return  

def p1Scomp_theta_check(name,n,ln,compnum):

        
    for i in range(ln):
        psi,speciallist,maxlist,xoptalist=S3.p1call_file(name,n,i,ln)
        
        for j in range(len(maxlist)):
            
            x=maxlist[j]
            theta=float(i)*2*np.pi/(ln-1)
            print "++++++++++++++++++++++++++++++"
            print theta
            print " "
        
            #Calculates the componenets of <S> 
            av=np.array([sin(x[0])*cos(x[1]),sin(x[0])*sin(x[1]),cos(x[0])]) #defines the vector associated with each measurement
            avp=np.array([sin(x[2])*cos(x[3]),sin(x[2])*sin(x[3]),cos(x[2])])#avp=unit vector for A' measurement 
            bv=np.array([sin(x[4])*cos(x[5]),sin(x[4])*sin(x[5]),cos(x[4])])#bv=unit vector for B measurement
            bvp=np.array([sin(x[6])*cos(x[7]),sin(x[6])*sin(x[7]),cos(x[6])])
            cv=np.array([sin(x[8])*cos(x[9]),sin(x[8])*sin(x[9]),cos(x[8])])
            cvp=np.array([sin(x[10])*cos(x[11]),sin(x[10])*sin(x[11]),cos(x[10])])

            A=(av[0]*S3.sigma1[0]+av[1]*S3.sigma1[1]+av[2]*S3.sigma1[2])/np.linalg.norm(av)        #creates each componenet of the bell inequality
            Ap=(avp[0]*S3.sigma1[0]+avp[1]*S3.sigma1[1]+avp[2]*S3.sigma1[2])/np.linalg.norm(avp)     #corresponds to the A' operator
            B=(bv[0]*S3.sigma2[0]+bv[1]*S3.sigma2[1]+bv[2]*S3.sigma2[2])/np.linalg.norm(bv)        #the linalg.norm() command normalizes the vector to ensure that there are only unit vectors
            Bp=(bvp[0]*S3.sigma2[0]+bvp[1]*S3.sigma2[1]+bvp[2]*S3.sigma2[2])/np.linalg.norm(bvp)
            C=(cv[0]*S3.sigma3[0]+cv[1]*S3.sigma3[1]+cv[2]*S3.sigma3[2])/np.linalg.norm(cv)
            Cp=(cvp[0]*S3.sigma3[0]+cvp[1]*S3.sigma3[1]+cvp[2]*S3.sigma3[2])/np.linalg.norm(cvp)
            

            P=C+Cp              #Defines the P,P' operator combinations of C,C' for ease of calculation
            Pp=C-Cp

            
            psidag=(psi.conjugate()).transpose() #|psi>dagge

            if compnum==2: #Breaks <S> into two different components, A(BP+B'P') and A'(BP'-B'P)

                S=np.dot(A,(np.dot(B,P)+np.dot(Bp,Pp)))+np.dot(Ap,(np.dot(B,Pp)-np.dot(Bp,P)))  #The Schvetlincthy operator 

                CmpA=np.dot(A,(np.dot(B,P)+np.dot(Bp,Pp)))
                CmpB=np.dot(Ap,(np.dot(B,Pp)-np.dot(Bp,P)))

                Faa=np.real(np.dot(psidag,np.dot(CmpA,psi)))  #The expectation value itself, <psi|S|psi>
                Fb=np.real(np.dot(psidag,np.dot(CmpB,psi)))  #The expectation value itself, <psi|S|psi>

                print Faa, " <A(BP+B'P')>"
                print Fb," <A'(BP'-B'P)>"

            if compnum==4:
                CmpA=np.dot(A,(np.dot(B,P)))
                CmpB=np.dot(A,(np.dot(Bp,Pp)))
                CmpC=np.dot(Ap,(np.dot(B,Pp)))
                CmpD=np.dot(Ap,(-np.dot(Bp,P)))

                Faa=np.real(np.dot(psidag,np.dot(CmpA,psi)))  
                Fb=np.real(np.dot(psidag,np.dot(CmpB,psi)))  
                Fc=np.real(np.dot(psidag,np.dot(CmpC,psi)))  
                Fd=np.real(np.dot(psidag,np.dot(CmpD,psi)))  

                print Faa[0,0], " <ABP>"
                print Fb[0,0], " <AB'P'>"
                print Fc[0,0]," <A'BP'>"
                print Fd[0,0]," <-A'B'P>"

            if compnum==8:
                CmpA=np.dot(A,(np.dot(B,C)))
                CmpB=np.dot(A,(np.dot(Bp,C)))
                CmpC=np.dot(Ap,(np.dot(B,C)))
                CmpD=np.dot(Ap,(-np.dot(Bp,C)))
                CmpA2=np.dot(A,(np.dot(B,Cp)))
                CmpB2=np.dot(A,(np.dot(Bp,-Cp)))
                CmpC2=np.dot(Ap,(np.dot(B,-Cp)))
                CmpD2=np.dot(Ap,(-np.dot(Bp,Cp)))

                Faa=np.real(np.dot(psidag,np.dot(CmpA,psi)))  
                Fb=np.real(np.dot(psidag,np.dot(CmpB,psi)))  
                Fc=np.real(np.dot(psidag,np.dot(CmpC,psi)))  
                Fd=np.real(np.dot(psidag,np.dot(CmpD,psi)))
                Faa2=np.real(np.dot(psidag,np.dot(CmpA2,psi)))  
                Fb2=np.real(np.dot(psidag,np.dot(CmpB2,psi)))  
                Fc2=np.real(np.dot(psidag,np.dot(CmpC2,psi)))  
                Fd2=np.real(np.dot(psidag,np.dot(CmpD2,psi)))

                print Faa[0,0], " <ABC>"
                print Fb[0,0], " <AB'C>"
                print Fc[0,0]," <A'BC>"
                print Fd[0,0]," <-A'B'C>"
                print Faa2[0,0], " <-ABC'>"
                print Fb2[0,0], " <-AB'C'>"
                print Fc2[0,0]," <-A'BC'>"
                print Fd2[0,0]," <A'B'C'>"
            print " "
    
    return

n=25
ln=500
name="GGHZ"
save=0
clear=1
color='blue'
S3.p1Scomp_compare_tgl_plot(name,n,ln)

py.show()
print "FIn"
