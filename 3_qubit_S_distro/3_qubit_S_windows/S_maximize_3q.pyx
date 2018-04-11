#Calculates the optimum operators for the maximum violation of Svetlichny's inequality
#for a given state.

import qutip as qt      #Python quantum toolbox. Very useful. Used to perform tensor, sigma, expect operations.
import scipy.optimize as optimize #brings in the package to maximize the function #The optimization routine called specifically.
import time             #Gives a sense of how long certain programs take.
import numpy as np      #The standard scientific mathematical module, numpy allows for fast array creation and indexing
import pickle           #Allows the data to be recorded in a way that it can be used later (pickling and unpickling- it's in the python documentation)
import pylab as py      #A plotting program that is used to graph various results

cimport numpy as np  #Uses the cython version of numpy for faster speeds


cdef extern from "math.h":  #Defines sin and cos as c functions for speed, and so that they can be called without any sort of extension
    double sin(double)

cdef extern from "math.h":
    double cos(double)

cdef int i,j                #Defines the counting variables, i,j as int for cython


#Operators

I=qt.qeye(2)                    #Defines the unit operator

A=qt.tensor(qt.sigmax(),I,I)    #Creates the pauli operators for each different qubit
B=qt.tensor(qt.sigmay(),I,I)
C=qt.tensor(qt.sigmaz(),I,I)    #Example: Q=Sigma z operator for qubit 1 
D=qt.tensor(I,qt.sigmax(),I)
E=qt.tensor(I,qt.sigmay(),I)
F=qt.tensor(I,qt.sigmaz(),I)
G=qt.tensor(I,I,qt.sigmax())
H=qt.tensor(I,I,qt.sigmay())
K=qt.tensor(I,I,qt.sigmaz())

u=A.shape

O=np.zeros(u,dtype=complex)  #Takes the above Qobj qutip classes and turns them into faster numpy arrays for minimizing
M=np.zeros(u,dtype=complex)
Q=np.zeros(u,dtype=complex)
R=np.zeros(u,dtype=complex)
S=np.zeros(u,dtype=complex)
T=np.zeros(u,dtype=complex)
U=np.zeros(u,dtype=complex)
V=np.zeros(u,dtype=complex)
W=np.zeros(u,dtype=complex)

for i in range(u[0]):       #The loop that creates the numpy arrays for the pauli operators
    for j in range(u[1]):
        O[i,j]=A[i,j]
        M[i,j]=B[i,j]
        Q[i,j]=C[i,j]
        R[i,j]=D[i,j]
        S[i,j]=E[i,j]
        T[i,j]=F[i,j]
        U[i,j]=G[i,j]
        V[i,j]=H[i,j]
        W[i,j]=K[i,j]

sigma1=np.array([O,M,Q],dtype=complex)    #Pauli vector, used to compute the dot product for each vector later
sigma2=np.array([R,S,T],dtype=complex)
sigma3=np.array([U,V,W],dtype=complex)

oooq=qt.tensor(qt.basis(2,0),qt.basis(2,0),qt.basis(2,0)) #Defines different states the overall state is made up of: ooo corresponds to |000>, lol corresponds to |101>
looq=qt.tensor(qt.basis(2,1),qt.basis(2,0),qt.basis(2,0)) #These are qobj classes, from qutip.
oloq=qt.tensor(qt.basis(2,0),qt.basis(2,1),qt.basis(2,0))
oolq=qt.tensor(qt.basis(2,0),qt.basis(2,0),qt.basis(2,1))
lloq=qt.tensor(qt.basis(2,1),qt.basis(2,1),qt.basis(2,0))
lolq=qt.tensor(qt.basis(2,1),qt.basis(2,0),qt.basis(2,1))
ollq=qt.tensor(qt.basis(2,0),qt.basis(2,1),qt.basis(2,1))
lllq=qt.tensor(qt.basis(2,1),qt.basis(2,1),qt.basis(2,1))

ooo=np.zeros([8,1],dtype=complex)
loo=np.zeros([8,1],dtype=complex)
olo=np.zeros([8,1],dtype=complex)
ool=np.zeros([8,1],dtype=complex)
llo=np.zeros([8,1],dtype=complex)
lol=np.zeros([8,1],dtype=complex)
oll=np.zeros([8,1],dtype=complex)
lll=np.zeros([8,1],dtype=complex)


for i in range(8):                  #Casts the different possible components of Psi into numpy arrays for faster computations.
    ooo[i,0]=oooq[i,0]
    loo[i,0]=looq[i,0]
    olo[i,0]=oloq[i,0]
    ool[i,0]=oolq[i,0]
    llo[i,0]=lloq[i,0]
    lol[i,0]=lolq[i,0]
    oll[i,0]=ollq[i,0]
    lll[i,0]=lllq[i,0]

#Expectation Value Function
    
def Fa(x, psi):  #A function that takes |psi> and some combination of angles for each basis vector and computes the expectation value of the Svetlinchny operator.

    av=np.array([sin(x[0])*cos(x[1]),sin(x[0])*sin(x[1]),cos(x[0])]) #defines the vector associated with each measurement
    avp=np.array([sin(x[2])*cos(x[3]),sin(x[2])*sin(x[3]),cos(x[2])])#avp=unit vector for A' measurement 
    bv=np.array([sin(x[4])*cos(x[5]),sin(x[4])*sin(x[5]),cos(x[4])])#bv=unit vector for B measurement
    bvp=np.array([sin(x[6])*cos(x[7]),sin(x[6])*sin(x[7]),cos(x[6])])
    cv=np.array([sin(x[8])*cos(x[9]),sin(x[8])*sin(x[9]),cos(x[8])])
    cvp=np.array([sin(x[10])*cos(x[11]),sin(x[10])*sin(x[11]),cos(x[10])])

    A=(av[0]*sigma1[0]+av[1]*sigma1[1]+av[2]*sigma1[2])/np.linalg.norm(av)        #creates each componenet of the bell inequality
    Ap=(avp[0]*sigma1[0]+avp[1]*sigma1[1]+avp[2]*sigma1[2])/np.linalg.norm(avp)     #corresponds to the A' operator
    B=(bv[0]*sigma2[0]+bv[1]*sigma2[1]+bv[2]*sigma2[2])/np.linalg.norm(bv)        #the linalg.norm() command normalizes the vector to ensure that there are only unit vectors
    Bp=(bvp[0]*sigma2[0]+bvp[1]*sigma2[1]+bvp[2]*sigma2[2])/np.linalg.norm(bvp)
    C=(cv[0]*sigma3[0]+cv[1]*sigma3[1]+cv[2]*sigma3[2])/np.linalg.norm(cv)
    Cp=(cvp[0]*sigma3[0]+cvp[1]*sigma3[1]+cvp[2]*sigma3[2])/np.linalg.norm(cvp)
    

    P=C+Cp              #Defines the P,P' operator combinations of C,C' for ease of calculation
    Pp=C-Cp

    S=np.dot(A,(np.dot(B,P)+np.dot(Bp,Pp)))+np.dot(Ap,(np.dot(B,Pp)-np.dot(Bp,P)))  #The Svetlichny operator 
    
    psidag=(psi.conjugate()).transpose() #|psi>dagger

    Fa=-np.real(np.dot(psidag,np.dot(S,psi)))  #The negative of the expectation value itself, <psi|S|psi>, written using numpy dot functions. The negative is to ensure it is minimized properly.
    Fa=Fa[0,0]
    return Fa

def S_maximize(int n,psi):  #The function that maximizes the quatnity <S> for a given state |psi>
    cdef int i,j            #A cython element that should define the loops as C elements for speed.
    
    xoptalist=np.zeros((n+1),dtype=np.ndarray) #An array that will be used to hold the raw computations for the minimized unit vectors

    for j in range (n+1): #For each state, since a random guess is used, and the function fmin_slsqp can be caught in local minima, in order to determine that actual minima
                          #This is iterated n+1 times. The larger N is, the more confidence that it is actually the global minima.
        ranges=np.array([[0.0,2*np.pi], [0.0,2*np.pi], [0.0,2*np.pi], [0.0, 2*np.pi],[0.0,2*np.pi], [0.0,2*np.pi], [0.0,2*np.pi], [0.0, 2*np.pi],[0.0,2*np.pi], [0.0,2*np.pi], [0.0,2*np.pi], [0.0, 2*np.pi]]) #Bounds for the minimization function. Parameters will not exceed these.

        x0=np.array([np.random.uniform(0.0,2*np.pi),np.random.uniform(0.0,2*np.pi),np.random.uniform(0.0,2*np.pi),np.random.uniform(0.0,2*np.pi),np.random.uniform(0.0,2*np.pi),np.random.uniform(0.0,2*np.pi),np.random.uniform(0.0,2*np.pi),np.random.uniform(0.0,2*np.pi),np.random.uniform(0.0,2*np.pi),np.random.uniform(0.0,2*np.pi),np.random.uniform(0.0,2*np.pi),np.random.uniform(0.0,2*np.pi)]) #The initial guess.

        xopta=optimize.fmin_slsqp(Fa,x0,bounds=ranges,args=([psi]),iter = 1000, acc = 1.0E-11,disp=0)#minimization function: uses slsqp algorithm, http://ab-initio.mit.edu/wiki/index.php/NLopt_Algorithms for information.
        #requires an intial guess, range, max number of iterations, accuracy, and whether or not a report is to be displayed (optional)
        #Also to note, the minimizeation function takes the minimum of -<S> so as to maximize <S>
        
        xoptalist[j]=xopta  
    
    return xoptalist

#A series of wavefunctions that can be called.

def GGHZ(theta):
    name="GGHZ"
    psi=cos(theta)*ooo+sin(theta)*lll 
    return psi,name

def MS1(theta):
    name="MS1"
    psi=1.0/2.0**.5*(ooo+cos(theta)*lol+sin(theta)*lll)
    return psi,name

def MS2(theta):
    name="MS2"
    psi=1.0/2.0**.5*(ooo+cos(theta)*llo+sin(theta)*lll)
    return psi,name

def W(theta,phi):
    name="W"
    psi=sin(theta)*cos(phi)*loo+sin(theta)*sin(phi)*olo+cos(theta)*ool
    return psi,name


#Data modification routines


#This structure removes duplicate vectors 
    
def unique_search(xoptalist):
    cdef int i,j,q,memnumb
    memnum=0  #Will keep a record of how many unique vectors there are so that the final array can be of the proper size. 

    memlist=np.zeros(len(xoptalist), dtype=int) # An array that will record which basis vector sets of xoptalist are unique.

    for i in range(len(xoptalist)):             #For the length of xoptalist
        if memlist[i]==0:                       #If memlist[i]=0, then it is not known to be unique, if it equals 1, then it is unique, and if 2 it is a duplicate.
            memlist[i]=1                        #sets each element of memlist that is unknown as unique as it comes to it. It begins at [0] and goes to the end,
                                                # so the only ones declared as unique are the ones that have already been checked against each other unique vector.
                                                #It basically takes a vector, declares it as unique, and removes any vectors that are the same in the data set. Then it moves on
                                                #to the next one that is unknown, declares it as unique and repeats the process till all vectors have been checked. 
            memnum=memnum+1                     #updates how many are unique. 
            for j in range (len(xoptalist)-(1+i)): #It runs over the number of vectors left after i
                q=j+1+i                         #Ensures that q does not bleed back into the ranges already checked
                if memlist[q]==0:               #If it is unknown if the vector is unique

                    xopta=xoptalist[i]          #Then it takes the ith element
                    xopta2=xoptalist[q]         #And whichever is the qth element
            
                    av1=np.array([sin(xopta[0])*cos(xopta[1]),sin(xopta[0])*sin(xopta[1]),cos(xopta[0])]) 
                    avp1=np.array([sin(xopta[2])*cos(xopta[3]),sin(xopta[2])*sin(xopta[3]),cos(xopta[2])])#avp=unit vector for A' measurement 
                    bv1=np.array([sin(xopta[4])*cos(xopta[5]),sin(xopta[4])*sin(xopta[5]),cos(xopta[4])])
                    bvp1=np.array([sin(xopta[6])*cos(xopta[7]),sin(xopta[6])*sin(xopta[7]),cos(xopta[6])])
                    cv1=np.array([sin(xopta[8])*cos(xopta[9]),sin(xopta[8])*sin(xopta[9]),cos(xopta[8])])
                    cvp1=np.array([sin(xopta[10])*cos(xopta[11]),sin(xopta[10])*sin(xopta[11]),cos(xopta[10])])
            
                    av2=np.array([sin(xopta2[0])*cos(xopta2[1]),sin(xopta2[0])*sin(xopta2[1]),cos(xopta2[0])])
                    avp2=np.array([sin(xopta2[2])*cos(xopta2[3]),sin(xopta2[2])*sin(xopta2[3]),cos(xopta2[2])])
                    bv2=np.array([sin(xopta2[4])*cos(xopta2[5]),sin(xopta2[4])*sin(xopta2[5]),cos(xopta2[4])])
                    bvp2=np.array([sin(xopta2[6])*cos(xopta2[7]),sin(xopta2[6])*sin(xopta2[7]),cos(xopta2[6])])
                    cv2=np.array([sin(xopta2[8])*cos(xopta2[9]),sin(xopta2[8])*sin(xopta2[9]),cos(xopta2[8])])
                    cvp2=np.array([sin(xopta2[10])*cos(xopta2[11]),sin(xopta2[10])*sin(xopta2[11]),cos(xopta2[10])])

                    av1=av1/np.linalg.norm(av1)
                    avp1=avp1/np.linalg.norm(avp1)
                    bv1=bv1/np.linalg.norm(bv1)
                    bvp1=bvp1/np.linalg.norm(bvp1)
                    cv1=cv1/np.linalg.norm(cv1)
                    cvp1=cvp1/np.linalg.norm(cvp1)
    
                    av2=av2/np.linalg.norm(av2)
                    avp2=avp2/np.linalg.norm(avp2)
                    bv2=bv2/np.linalg.norm(bv2)
                    bvp2=bvp2/np.linalg.norm(bvp2)
                    cv2=cv2/np.linalg.norm(cv2)
                    cvp2=cvp2/np.linalg.norm(cvp2)

                    bnd=.9999                           #And compares them. If their dot products are sufficiently close to 1, then they are determined to be the same vector,
                                                        # with the closeness determined by bnd. 
                    if (np.dot(av1,av2))>bnd and (np.dot(bv1,bv2))>bnd and (np.dot(cv1,cv2))>bnd and (np.dot(avp1,avp2))>bnd and (np.dot(bvp1,bvp2))>bnd and (np.dot(cvp1,cvp2)) >bnd:
                        memlist[q]=2                    
                        
    speciallist=np.zeros((memnum),dtype=np.ndarray)    #Defines an array that will hold the unique vector sets
    q=0                                                 #A counting varible.
    for i in range (len(xoptalist)):                    #Runs over the data set, pulling out the differnt possible variables
        if memlist[i]==1.0:
            speciallist[q]=xoptalist[i]
            q=q+1
    return speciallist

#Will single out the vectors that have the maximium violation, both positive and negative are kept by absolute value.

def max_cull(speciallist,psi):
    cdef int i,j,q
    
    ilist=[]            #A python list that holds which sets of vectors correspond to the maximum violation
    Fpntst=0.0          #The test variable
    tol=0.00001         #The tolerance. Used because there is some error associated with Fmin_Slsqp, and the computer will often yield something like 3E-20 as zero
    
    for j in range(len(speciallist)): #For each element in the unique vector list, <S> is calculated and comparted to Fpntst. 
        x=speciallist[j]
        Ftst=abs(Fa(x,psi))             #only the magnitude is considered here. 
        if Ftst-Fpntst>=tol:       #If the difference is greater than the tolerance, then it is declared to be a new max. 
            Fpntst=Ftst
            ilist=[j]                    #ilist is set to zero to represent this fact.
        if abs(Ftst-Fpntst)<tol:        #If the difference is less than the tolerance, then it is declared to yield the same max and the number of the vector is recorded.
            ilist.append(j)
    maxlist=np.zeros(len(ilist),dtype=np.ndarray) #Creates the array to hold all the maximized vectors.
    for q in range(len(ilist)): #Records all the vectors found which yileded the max violation.
        u=ilist[q]
        maxlist[q]=speciallist[u]
    return maxlist


#This will save the data as a file which can be unpickled later if desired.

def save_file_1p(name,n,q,ln,psi,speciallist,maxlist,xoptalist): #Used for states that can be expressed with only one parameter between the different elements. 
    filename=name+" 3 qubit pickled "+str(n)+" , "+str(q)+" , "+str(ln)+" .txt"
    f=open(filename,"w")
    pickle.dump(psi,f)
    pickle.dump(speciallist,f)
    pickle.dump(maxlist,f)
    pickle.dump(xoptalist,f)
    f.close()
    return

def save_file_2p(name,n,q,j,ln,psi,speciallist,maxlist,xoptalist): #Used for states that must be expressed with two parameters between the different elements. 
    filename=name+" 3 qubit pickled "+str(n)+" , "+str(q)+" , "+str(j)+" , "+str(ln)+" .txt"
    f=open(filename,"w")
    pickle.dump(psi,f)              #Despite it's name, pickling is something that is native to python. It allows the data to be stored in a way easy to recover within python. 
    pickle.dump(speciallist,f)
    pickle.dump(maxlist,f)
    pickle.dump(xoptalist,f)
    f.close()
    return


def ord_choose(ord,xoptalist,psi): #Allows for the changing of the order, whether the unique vectors are only of maximum value,
                     #or the maximum value vectors are only from the unique.
    if ord==0:
        speciallist=unique_search(xoptalist)
        maxlist=max_cull(speciallist,psi)
    if ord==1:
        maxlist=max_cull(xoptalist,psi)
        speciallist=unique_search(maxlist)
    return speciallist,maxlist


def DC(sc,ord,n,ln):  #The data creaton routine, calls the functions in order, allows choosing
                      # between different states by the following: 1=GGHZ, 2=MS1, 3=MS2, 4=W
                      #Will save the information in a series of files, then will call these files later. 
    t1=time.time()
    if sc==1:
        for q in range (ln):
            theta=float(q)*2*np.pi/(float(ln)-1)
            psi,name=GGHZ(theta)
            xoptalist=S_maximize(n,psi)
            speciallist,maxlist=ord_choose(ord,xoptalist,psi)            
            save_file_1p(name,n,q,ln,psi,speciallist,maxlist,xoptalist)
            print 100*(float(q)/float(ln))
    if sc==2:
        for q in range (ln):
            theta=float(q)*2*np.pi/(float(ln)-1)
            psi,name=MS1(theta)
            xoptalist=S_maximize(n,psi)
            speciallist,maxlist=ord_choose(ord,xoptalist,psi)
            save_file_1p(name,n,q,ln,psi,speciallist,maxlist,xoptalist)
            print 100*(float(q)/float(ln))
    if sc==3:
        for q in range (ln):
            theta=float(q)*2*np.pi/(float(ln)-1)
            psi,name=MS2(theta)
            xoptalist=S_maximize(n,psi)
            speciallist,maxlist=ord_choose(ord,xoptalist,psi)
            save_file_1p(name,n,q,ln,psi,speciallist,maxlist,xoptalist)
            print 100*(float(q)/float(ln))
    if sc==4:
        pct=0
        for q in range (ln):
            theta=float(q)*2*np.pi/(float(ln)-1)
            for j in range(ln):
                phi=float(j)*2*np.pi/(float(ln)-1)
                psi,name=W(theta,phi)
                xoptalist=S_maximize(n,psi)
                speciallist,maxlist=ord_choose(ord,xoptalist,psi)
                save_file_2p(name,n,q,j,ln,psi,speciallist,maxlist,xoptalist)
                print 100*(float(pct)/float(ln)**2)
                pct=pct+1
    t2=time.time()
    t3=(t2-t1)/60.0
    print t3," minutes to complete"
    return




def DC_custom_2p(ord,n,ln,psif,name):   #Allows one to use a custom 2-parameter psi function and name, but still have it automated
    t1=time.time()
    pct=0
    for q in range (ln):
        theta=float(q)*2*np.pi/(float(ln)-1)
        for j in range(ln):
            phi=float(j)*2*np.pi/(float(ln)-1)
            psi=psif(theta,phi)
            xoptalist=S_maximize(n,psi)
            speciallist,maxlist=ord_choose(ord,xoptalist,psi)
            save_file_2p(name,n,q,j,ln,psi,speciallist,maxlist,xoptalist)
            print 100*(float(pct)/float(ln)**2)
            pct=pct+1
    t2=time.time()
    t3=(t2-t1)/60.0
    print t3," minutes to complete"
    return




def DC_custom_1p(ord,n,ln,psif,name):  #Allows one to use a custom 1-parameter psi function and name, but still have it automated
    t1=time.time()
    for q in range(ln):
        theta=float(q)*2*np.pi/(float(ln)-1)
        psi=psif(theta)
        xoptalist=S_maximize(n,psi)
        speciallist,maxlist=ord_choose(ord,xoptalist,psi)
        save_file_1p(name,n,q,ln,psi,speciallist,maxlist,xoptalist)
        print 100*(float(q)/float(ln))
    t2=time.time()
    t3=(t2-t1)/60.0
    print t3," minutes to complete"
    return




def p1call_file(name,n,q,ln):           #Allows one to call a specific run of a 1 parameter calculation in order to check different elements
    filename=name+" 3 qubit pickled "+str(n)+" , "+str(q)+" , "+str(ln)+" .txt"
    f2=open(filename, 'rb')
    psi=pickle.load(f2)
    speciallist=pickle.load(f2)
    maxlist=pickle.load(f2)
    xoptalist=pickle.load(f2)
    f2.close
    return psi,speciallist,maxlist,xoptalist

def p2call_file(name,n,q,j,ln):     #Allows one to call a specific run of a 2 parameter calculation in order to check different elements
    filename=name+" 3 qubit pickled "+str(n)+" , "+str(q)+" , "+str(j)+" , "+str(ln)+" .txt"
    f2=open(filename, 'rb')
    psi=pickle.load(f2)
    speciallist=pickle.load(f2)
    maxlist=pickle.load(f2)
    xoptalist=pickle.load(f2)
    f2.close
    return psi,speciallist,maxlist,xoptalist




