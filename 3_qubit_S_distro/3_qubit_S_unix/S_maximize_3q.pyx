#Calculates the optimum operators for the maximum violation of Svetlichny's inequality
#for a given state.

import qutip as qt      #Python quantum toolbox. Very useful. Used to perform tensor, sigma, expect operations.
import scipy.optimize as optimize #brings in the package to maximize the function #The optimization routine called specifically.
import time             #Gives a sense of how long certain programs take.
import numpy as np      #The standard scientific mathematical module, numpy allows for fast array creation and indexing
import pickle           #Allows the data to be recorded in a way that it can be used later (pickling and unpickling- it's in the python documentation)
import pylab as py      #A plotting program that is used to graph various results
import sys

cimport numpy as np  #Uses the cython version of numpy for faster speeds


cdef extern from "math.h":  #Defines sin and cos as c functions for speed, and so that they can be called without any sort of extension
    double sin(double)

cdef extern from "math.h":
    double cos(double)

cdef int i,j                #Defines the counting variables, i,j as int for cython

pathname=sys.path[0]

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
    filename=pathname+"/Data/"+name+"/"+name+" 3 qubit pickled "+str(n)+" , "+str(q)+" , "+str(ln)+" .txt"
    f=open(filename,"w")
    pickle.dump(psi,f)
    pickle.dump(speciallist,f)
    pickle.dump(maxlist,f)
    pickle.dump(xoptalist,f)
    f.close()
    return

def save_file_2p(name,n,q,j,ln,psi,speciallist,maxlist,xoptalist): #Used for states that must be expressed with two parameters between the different elements. 
    filename=pathname+"/Data/"+name+"/"+name+" 3 qubit pickled "+str(n)+" , "+str(q)+" , "+str(j)+" , "+str(ln)+" .txt"
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
    filename=pathname+"/Data/"+name+"/"+name+" 3 qubit pickled "+str(n)+" , "+str(q)+" , "+str(ln)+" .txt"
    f2=open(filename, 'rb')
    psi=pickle.load(f2)
    speciallist=pickle.load(f2)
    maxlist=pickle.load(f2)
    xoptalist=pickle.load(f2)
    f2.close
    return psi,speciallist,maxlist,xoptalist

def p2call_file(name,n,q,j,ln):     #Allows one to call a specific run of a 2 parameter calculation in order to check different elements
    filename=pathname+"/Data/"+name+"/"+name+" 3 qubit pickled "+str(n)+" , "+str(q)+" , "+str(j)+" , "+str(ln)+" .txt"
    f2=open(filename, 'rb')
    psi=pickle.load(f2)
    speciallist=pickle.load(f2)
    maxlist=pickle.load(f2)
    xoptalist=pickle.load(f2)
    f2.close
    return psi,speciallist,maxlist,xoptalist



def vec_print1p(f,printlist,psi,name,n,ln,q): #This will print out a list of all the maximized vectors for a one parameter state found to a folder called 'printout.' It is human-readable only, not for python to use.
    theta=float(q)*2*np.pi/(float(ln)-1)
    f.write( "=================================")
    f.write('\n')
    f.write("Psi= ")
    f.write('\n')
    f.write(str(psi))
    f.write('\n')
    f.write(str(theta)+" =Value of theta  "+str(q)+" =q")
    f.write('\n')
    f.write(str(len(printlist))+"= number of unique max vectors")
    f.write('\n')
    for i in range(len(printlist)):            
        xopta=printlist[i]
        
        av=np.array([sin(xopta[0])*cos(xopta[1]),sin(xopta[0])*sin(xopta[1]),cos(xopta[0])])
        avp=np.array([sin(xopta[2])*cos(xopta[3]),sin(xopta[2])*sin(xopta[3]),cos(xopta[2])])
        bv=np.array([sin(xopta[4])*cos(xopta[5]),sin(xopta[4])*sin(xopta[5]),cos(xopta[4])])
        bvp=np.array([sin(xopta[6])*cos(xopta[7]),sin(xopta[6])*sin(xopta[7]),cos(xopta[6])])
        cv=np.array([sin(xopta[8])*cos(xopta[9]),sin(xopta[8])*sin(xopta[9]),cos(xopta[8])])
        cvp=np.array([sin(xopta[10])*cos(xopta[11]),sin(xopta[10])*sin(xopta[11]),cos(xopta[10])])
        f.write("---------------------------------")
        f.write('\n')
        f.write("Vector number "+str(i+1))
        f.write('\n')
        f.write("a= "+str(av/np.linalg.norm(av)))
        f.write('\n')
        f.write("a'= "+str(avp/np.linalg.norm(avp)))
        f.write('\n')
        f.write("b= "+str(bv/np.linalg.norm(bv)))
        f.write('\n')
        f.write("b'= "+str(bvp/np.linalg.norm(bvp)))
        f.write('\n')
        f.write("c= "+str(cv/np.linalg.norm(cv)))
        f.write('\n')
        f.write("c'= "+str(cvp/np.linalg.norm(cvp)))
        f.write('\n')
        f.write("<S>= "+str(abs(Fa(xopta,psi))))
        f.write('\n')
        
    f.write( "=================================")
    return


def vec_print2p(f,printlist,psi,name,n,ln,q,j):   #This will print out a list of all the maximized vectors for a two parameter state found to a folder called 'printout.' It is human-readable only, not for python to use.
    theta=float(q)*2*np.pi/(float(ln)-1)
    phi=float(j)*2*np.pi/(float(ln)-1)

    f.write( "=================================")
    f.write('\n')
    f.write("Psi= ")
    f.write('\n')
    f.write(str(psi))
    f.write('\n')
    f.write(str(theta)+" =Value of theta  "+str(q)+" =q")
    f.write('\n')
    f.write(str(phi)+" =Value of phi  "+str(j)+" =j")
    f.write('\n')
    f.write(str(len(printlist))+"= number of unique max vectors")
    f.write('\n')
    for i in range(len(printlist)):            
        xopta=printlist[i]
        
        av=np.array([sin(xopta[0])*cos(xopta[1]),sin(xopta[0])*sin(xopta[1]),cos(xopta[0])])
        avp=np.array([sin(xopta[2])*cos(xopta[3]),sin(xopta[2])*sin(xopta[3]),cos(xopta[2])])
        bv=np.array([sin(xopta[4])*cos(xopta[5]),sin(xopta[4])*sin(xopta[5]),cos(xopta[4])])
        bvp=np.array([sin(xopta[6])*cos(xopta[7]),sin(xopta[6])*sin(xopta[7]),cos(xopta[6])])
        cv=np.array([sin(xopta[8])*cos(xopta[9]),sin(xopta[8])*sin(xopta[9]),cos(xopta[8])])
        cvp=np.array([sin(xopta[10])*cos(xopta[11]),sin(xopta[10])*sin(xopta[11]),cos(xopta[10])])
        f.write("---------------------------------")
        f.write('\n')
        f.write("Vector number "+str(i+1))
        f.write('\n')
        f.write("a= "+str(av/np.linalg.norm(av)))
        f.write('\n')
        f.write("a'= "+str(avp/np.linalg.norm(avp)))
        f.write('\n')
        f.write("b= "+str(bv/np.linalg.norm(bv)))
        f.write('\n')
        f.write("b'= "+str(bvp/np.linalg.norm(bvp)))
        f.write('\n')
        f.write("c= "+str(cv/np.linalg.norm(cv)))
        f.write('\n')
        f.write("c'= "+str(cvp/np.linalg.norm(cvp)))
        f.write('\n')
        f.write("<S>= "+str(abs(Fa(xopta,psi))))
        f.write('\n')
        
    f.write( "=================================")
    return


def vecprint(n,ln,name,param): #This is the control function for the printing functions. It allows the choice between 1 and 2 parameters.
    pth=sys.path[0]
    filename=str(pth)+"/Printout/"+name+" 3 qubit pickled "+str(n)+" , "+str(ln)+" .txt"
    f=open(filename,"wb")
    f.write(name+" trial, "+str(n)+"=n, "+str(ln)+"=ln")
    f.write('\n')
    for q in range (ln):
        if param==1:
            psi,speciallist,maxlist,xoptalist=p1call_file(name,n,q,ln)
            printlist=maxlist
            vec_print1p(f,printlist,psi,name,n,ln,q)

        if param==2:
            for j in range(ln):
                psi,speciallist,maxlist,xoptalist=p2call_file(name,n,q,j,ln)
                printlist=maxlist
                vec_print2p(f,printlist,psi,name,n,ln,q,j)
    f.close()
    return




def p1tgl_theta_plot(name,n,ln,save,clear): #This function plots the three tangle versus theta of a one parameter state. 
    if clear==1:
        py.clf()
    filename=pathname+"/Figures/"+name+"tgl vs theta"+str(n)+" , "+str(ln)+" .png" #This is saved in the figures folder.

    tglist=np.zeros(ln)
    xlist=np.zeros(ln)
    for i in range(ln):
        psi,speciallist,maxlist,xoptalist=p1call_file(name,n,i,ln)
        d1=(psi[0,0]**2)*(psi[7,0]**2)+(psi[1,0]**2)*(psi[6,0]**2)+(psi[2,0]**2)*(psi[5,0]**2)+(psi[4,0]**2)*(psi[3,0]**2)
        d2=psi[0,0]*psi[7,0]*psi[3,0]*psi[4,0]+psi[0,0]*psi[7,0]*psi[5,0]*psi[2,0]+psi[0,0]*psi[7,0]*psi[6,0]*psi[1,0]+psi[3,0]*psi[4,0]*psi[5,0]*psi[2,0]+psi[3,0]*psi[4,0]*psi[6,0]*psi[1,0]+psi[5,0]*psi[2,0]*psi[6,0]*psi[1,0]
        d3=psi[0,0]*psi[6,0]*psi[5,0]*psi[3,0]+psi[7,0]*psi[1,0]*psi[2,0]*psi[4,0]
        Tgl=4*abs(d1-2*d2+4*d3)
        tglist[i]=Tgl
        theta=float(i)*2*np.pi/(float(ln)-1)
        xlist[i]=theta
    lable="Three Tangle, "+name
    py.plot(xlist,tglist,label=lable)
    py.xlabel('Theta')
    py.ylabel('Tangle')
    py.legend()
    if save==1:
        py.savefig(filename,dpi=200) 
    return





def p1S_max_theta_plot(name,n,ln,save,clear): #This function plots  <S> max versus theta for a one parameter state
    if clear==1:
        py.clf()
    filename=pathname+"/Figures/"+name+"<S> max vs theta"+str(n)+" , "+str(ln)+" .png"

    ylist=np.zeros(ln)
    xlist=np.zeros(ln)
    for i in range(ln):
        psi,speciallist,maxlist,xoptalist=p1call_file(name,n,i,ln)
        theta=float(i)*2*np.pi/(ln-1)
        xlist[i]=theta
        x=maxlist[0]
        Fmax=abs(Fa(x,psi))
        ylist[i]=Fmax

    lable=name
    py.plot(xlist,ylist,label=lable)

    py.xlabel('Theta')
    py.ylabel('<S> max')
    py.legend()
    if save==1:
        py.savefig(filename,dpi=200)
    return




def p1S_max_tgl_plot(name,n,ln,save,clear,color): #This function plots <S> max versus the three tangle for a one parameters state.
    if clear==1:
        py.clf()
    filename=pathname+"/Figures/"+name+"<S>max vs tgl"+str(n)+" , "+str(ln)+" .png"

    ylist=np.zeros(ln)
    xlist=np.zeros(ln)
    for i in range(ln):
        psi,speciallist,maxlist,xoptalist=p1call_file(name,n,i,ln)
        x=maxlist[0]
        Fmax=abs(Fa(x,psi))
        ylist[i]=Fmax

        d1=(psi[0,0]**2)*(psi[7,0]**2)+(psi[1,0]**2)*(psi[6,0]**2)+(psi[2,0]**2)*(psi[5,0]**2)+(psi[4,0]**2)*(psi[3,0]**2)
        d2=psi[0,0]*psi[7,0]*psi[3,0]*psi[4,0]+psi[0,0]*psi[7,0]*psi[5,0]*psi[2,0]+psi[0,0]*psi[7,0]*psi[6,0]*psi[1,0]+psi[3,0]*psi[4,0]*psi[5,0]*psi[2,0]+psi[3,0]*psi[4,0]*psi[6,0]*psi[1,0]+psi[5,0]*psi[2,0]*psi[6,0]*psi[1,0]
        d3=psi[0,0]*psi[6,0]*psi[5,0]*psi[3,0]+psi[7,0]*psi[1,0]*psi[2,0]*psi[4,0]
        Tgl=4*abs(d1-2*d2+4*d3)
        xlist[i]=Tgl

    lable=name
    py.scatter(xlist,ylist,s=10,c=color,linewidth=0.5,label=lable)
    
    py.xlabel('Three Tangle')
    py.ylabel('<S> Max')
    py.legend()
    if save==1:
        py.savefig(filename,dpi=200)
    
    return    
    



def p1Scomp_theta_plot(name,n,ln,compnum): #This function plots the different components of <S> for a one parameter state versus theta, saving each componenet in a different folder.
                                           #Compnum determines how many elements <S> will be broken into, with values of 2,4, or 8. Below is shown how the breakdown occurs.
    py.clf()
    
    ylist=[]
    xlist=[]

    yAlist=[]
    yBlist=[]
    yClist=[]
    yDlist=[]
    yElist=[]
    yFlist=[]
    yGlist=[]
    yHlist=[]
        
    for i in range(ln):
        psi,speciallist,maxlist,xoptalist=p1call_file(name,n,i,ln)
        
        for j in range(len(maxlist)):
            x=maxlist[j]
            theta=float(i)*2*np.pi/(ln-1)
            Fmax=abs(Fa(x,psi))
    
            xlist.append(theta)
            ylist.append(Fmax)
        
            #Calculates the componenets of <S> 
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

            
            psidag=(psi.conjugate()).transpose() #|psi>dagge

            if compnum==2: #Breaks <S> into two different components, A(BP+B'P') and A'(BP'-B'P)

                CmpA=np.dot(A,(np.dot(B,P)+np.dot(Bp,Pp)))
                CmpB=np.dot(Ap,(np.dot(B,Pp)-np.dot(Bp,P)))

                Faa=np.real(np.dot(psidag,np.dot(CmpA,psi)))  #The expectation value itself, <psi|S|psi>
                yAlist.append(Faa[0,0])
                Fb=np.real(np.dot(psidag,np.dot(CmpB,psi)))  #The expectation value itself, <psi|S|psi>
                yBlist.append(Fb[0,0])

            if compnum==4: #Breaks <S> into four different components, ABP,AB'P', A'BP', and -A'B'P
                CmpA=np.dot(A,(np.dot(B,P)))
                CmpB=np.dot(A,(np.dot(Bp,Pp)))
                CmpC=np.dot(Ap,(np.dot(B,Pp)))
                CmpD=np.dot(Ap,(-np.dot(Bp,P)))

                Faa=np.real(np.dot(psidag,np.dot(CmpA,psi)))  
                yAlist.append(Faa[0,0])
                Fb=np.real(np.dot(psidag,np.dot(CmpB,psi)))  
                yBlist.append(Fb[0,0])
                Fc=np.real(np.dot(psidag,np.dot(CmpC,psi)))  
                yClist.append(Fb[0,0])
                Fd=np.real(np.dot(psidag,np.dot(CmpD,psi)))  
                yDlist.append(Fb[0,0])

            if compnum==8:#Breaks <S> into eight different components in the same way as above, but splitting P and P' into C+C' and C-C'

                CmpA=np.dot(A,(np.dot(B,C)))
                CmpB=np.dot(A,(np.dot(Bp,C)))
                CmpC=np.dot(Ap,(np.dot(B,C)))
                CmpD=np.dot(Ap,(-np.dot(Bp,C)))
                CmpA2=np.dot(A,(np.dot(B,Cp)))
                CmpB2=np.dot(A,(np.dot(Bp,-Cp)))
                CmpC2=np.dot(Ap,(np.dot(B,-Cp)))
                CmpD2=np.dot(Ap,(-np.dot(Bp,Cp)))

                Faa=np.real(np.dot(psidag,np.dot(CmpA,psi)))  
                yAlist.append(Faa[0,0])
                Fb=np.real(np.dot(psidag,np.dot(CmpB,psi)))  
                yBlist.append(Fb[0,0])
                Fc=np.real(np.dot(psidag,np.dot(CmpC,psi)))  
                yClist.append(Fb[0,0])
                Fd=np.real(np.dot(psidag,np.dot(CmpD,psi)))  
                yDlist.append(Fb[0,0])
                Faa2=np.real(np.dot(psidag,np.dot(CmpA2,psi)))  
                yElist.append(Faa2[0,0])
                Fb2=np.real(np.dot(psidag,np.dot(CmpB2,psi)))  
                yFlist.append(Fb2[0,0])
                Fc2=np.real(np.dot(psidag,np.dot(CmpC2,psi)))  
                yGlist.append(Fb2[0,0])
                Fd2=np.real(np.dot(psidag,np.dot(CmpD2,psi)))  
                yHlist.append(Fb2[0,0])
    

    

    lable="Maximum <S> "+name #Saves the maximum violation as well for comparison.
    py.xlabel('Theta')
    py.ylabel('<S> component max')
    py.xlim(0,2*np.pi)
    py.scatter(xlist,ylist,s=10,c='black',linewidth=0.5,label=lable)
    py.legend()
    filename=pathname+"/Figures/SComp/"+name+"<S>comp max vs theta"+str(n)+" , "+str(ln)+" max.png"
    py.savefig(filename,dpi=200)
    
    if compnum==2: #These structures graph the data, depending on which compnum was chosen. It is all saved to the folder SComp under the figures directory.
        
        py.clf()
        lable="<A(BP+B'P')> "+name
        py.xlabel('Theta')
        py.ylabel('<S> component max')
        py.xlim(0,2*np.pi)
        py.scatter(xlist,yAlist,s=10,c='red',linewidth=0.5,label=lable)
        py.legend()
        filename=pathname+"/Figures/SComp/"+name+"<S>comp max vs theta"+str(n)+" , "+str(ln)+" A2.png"
        py.savefig(filename,dpi=200)

        py.clf()
        lable="<A'(BP'-B'P)> "+name
        py.xlabel('Theta')
        py.ylabel('<S> component max')
        py.xlim(0,2*np.pi)
        py.scatter(xlist,yBlist,s=10,c='blue',linewidth=0.5,label=lable)
        py.legend()
        filename=pathname+"/Figures/SComp/"+name+"<S>comp max vs theta"+str(n)+" , "+str(ln)+" B2.png"
        py.savefig(filename,dpi=200)

    if compnum==4:

        py.clf()
        lable="<ABP> "+name
        py.xlabel('Theta')
        py.ylabel('<S> component max')
        py.xlim(0,2*np.pi)
        py.scatter(xlist,yAlist,s=10,c='red',linewidth=0.5,label=lable)
        py.legend()
        filename=pathname+"/Figures/SComp/"+name+"<S>comp max vs theta"+str(n)+" , "+str(ln)+" A4.png"
        py.savefig(filename,dpi=200)

        py.clf()
        lable="<AB'P'> "+name
        py.xlabel('Theta')
        py.ylabel('<S> component max')
        py.xlim(0,2*np.pi)
        py.scatter(xlist,yBlist,s=10,c='blue',linewidth=0.5,label=lable)
        py.legend()
        filename=pathname+"/Figures/SComp/"+name+"<S>comp max vs theta"+str(n)+" , "+str(ln)+" B4.png"
        py.savefig(filename,dpi=200)

        py.clf()
        lable="<A'BP'> "+name
        py.xlabel('Theta')
        py.ylabel('<S> component max')
        py.xlim(0,2*np.pi)
        py.scatter(xlist,yClist,s=10,c='orange',linewidth=0.5,label=lable)
        py.legend()
        filename=pathname+"/Figures/SComp/"+name+"<S>comp max vs theta"+str(n)+" , "+str(ln)+" C4.png"
        py.savefig(filename,dpi=200)

        py.clf()
        lable="<-A'B'P> "+name
        py.xlabel('Theta')
        py.ylabel('<S> component max')
        py.xlim(0,2*np.pi)
        py.scatter(xlist,yDlist,s=10,c='green',linewidth=0.5,label=lable)
        py.legend()
        filename=pathname+"/Figures/SComp/"+name+"<S>comp max vs theta"+str(n)+" , "+str(ln)+" D4.png"
        py.savefig(filename,dpi=200)

    if compnum==8:

        py.clf()
        lable="<ABC> "+name
        py.xlabel('Theta')
        py.ylabel('<S> component max')
        py.xlim(0,2*np.pi)
        py.scatter(xlist,yAlist,s=10,c='red',linewidth=0.5,label=lable)
        py.legend()
        filename=pathname+"/Figures/SComp/"+name+"<S>comp max vs theta"+str(n)+" , "+str(ln)+" A8.png"
        py.savefig(filename,dpi=200)

        py.clf()
        lable="<AB'C> "+name
        py.xlabel('Theta')
        py.ylabel('<S> component max')
        py.xlim(0,2*np.pi)
        py.scatter(xlist,yBlist,s=10,c='blue',linewidth=0.5,label=lable)
        py.legend()
        filename=pathname+"/Figures/SComp/"+name+"<S>comp max vs theta"+str(n)+" , "+str(ln)+" B8.png"
        py.savefig(filename,dpi=200)

        py.clf()
        lable="<A'BC> "+name
        py.xlabel('Theta')
        py.ylabel('<S> component max')
        py.xlim(0,2*np.pi)
        py.scatter(xlist,yClist,s=10,c='orange',linewidth=0.5,label=lable)
        py.legend()
        filename=pathname+"/Figures/SComp/"+name+"<S>comp max vs theta"+str(n)+" , "+str(ln)+" C8.png"
        py.savefig(filename,dpi=200)

        py.clf()
        lable="<-A'B'C> "+name
        py.xlabel('Theta')
        py.ylabel('<S> component max')
        py.xlim(0,2*np.pi)
        py.scatter(xlist,yDlist,s=10,c='green',linewidth=0.5,label=lable)
        py.legend()
        filename=pathname+"/Figures/SComp/"+name+"<S>comp max vs theta"+str(n)+" , "+str(ln)+" D8.png"
        py.savefig(filename,dpi=200)

        py.clf()
        lable="<ABC> "+name
        py.xlabel('Theta')
        py.ylabel('<S> component max')
        py.xlim(0,2*np.pi)
        py.scatter(xlist,yElist,s=10,c='red',linewidth=0.5,label=lable)
        py.legend()
        filename=pathname+"/Figures/SComp/"+name+"<S>comp max vs theta"+str(n)+" , "+str(ln)+" E8.png"
        py.savefig(filename,dpi=200)

        py.clf()
        lable="<-AB'C'> "+name
        py.xlabel('Theta')
        py.ylabel('<S> component max')
        py.xlim(0,2*np.pi)
        py.scatter(xlist,yFlist,s=10,c='blue',linewidth=0.5,label=lable)
        py.legend()
        filename=pathname+"/Figures/SComp/"+name+"<S>comp max vs theta"+str(n)+" , "+str(ln)+" F8.png"
        py.savefig(filename,dpi=200)

        py.clf()
        lable="<-A'BC'> "+name
        py.xlabel('Theta')
        py.ylabel('<S> component max')
        py.xlim(0,2*np.pi)
        py.scatter(xlist,yGlist,s=10,c='orange',linewidth=0.5,label=lable)
        py.legend()
        filename=pathname+"/Figures/SComp/"+name+"<S>comp max vs theta"+str(n)+" , "+str(ln)+" G8.png"
        py.savefig(filename,dpi=200)

        py.clf()
        lable="<A'B'C'> "+name
        py.xlabel('Theta')
        py.ylabel('<S> component')
        py.xlim(0,2*np.pi)
        py.scatter(xlist,yHlist,s=10,c='green',linewidth=0.5,label=lable)
        py.legend()
        filename=pathname+"/Figures/SComp/"+name+"<S>comp max vs theta"+str(n)+" , "+str(ln)+" H8.png"
        py.savefig(filename,dpi=200)
    
    return




def p1Scomp_tgl_plot(name,n,ln,compnum): #This function plots the different components of <S> for a one parameter state versus the three tangle, saving each componenet in a different folder.
                                           #Compnum determines how many elements <S> will be broken into, with values of 2,4, or 8. Below is shown how the breakdown occurs.

    py.clf()

    ylist=[]
    xlist=[]

    yAlist=[]
    yBlist=[]
    yClist=[]
    yDlist=[]
    yElist=[]
    yFlist=[]
    yGlist=[]
    yHlist=[]

    for i in range(ln):
        psi,speciallist,maxlist,xoptalist=p1call_file(name,n,i,ln)

        for j in range(len(maxlist)):
            x=maxlist[j]
            Fmax=abs(Fa(x,psi))
            ylist.append(Fmax)
        
            d1=(psi[0,0]**2)*(psi[7,0]**2)+(psi[1,0]**2)*(psi[6,0]**2)+(psi[2,0]**2)*(psi[5,0]**2)+(psi[4,0]**2)*(psi[3,0]**2)
            d2=psi[0,0]*psi[7,0]*psi[3,0]*psi[4,0]+psi[0,0]*psi[7,0]*psi[5,0]*psi[2,0]+psi[0,0]*psi[7,0]*psi[6,0]*psi[1,0]+psi[3,0]*psi[4,0]*psi[5,0]*psi[2,0]+psi[3,0]*psi[4,0]*psi[6,0]*psi[1,0]+psi[5,0]*psi[2,0]*psi[6,0]*psi[1,0]
            d3=psi[0,0]*psi[6,0]*psi[5,0]*psi[3,0]+psi[7,0]*psi[1,0]*psi[2,0]*psi[4,0]
            Tgl=4*abs(d1-2*d2+4*d3)
            
            xlist.append(Tgl)
                        
            #Calculates the componenets of <S> 
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

            
            psidag=(psi.conjugate()).transpose() #|psi>dagge

            if compnum==2: #Breaks <S> into two different components, A(BP+B'P') and A'(BP'-B'P)

                S=np.dot(A,(np.dot(B,P)+np.dot(Bp,Pp)))+np.dot(Ap,(np.dot(B,Pp)-np.dot(Bp,P)))  #The Schvetlincthy operator 

                CmpA=np.dot(A,(np.dot(B,P)+np.dot(Bp,Pp)))
                CmpB=np.dot(Ap,(np.dot(B,Pp)-np.dot(Bp,P)))

                Faa=np.real(np.dot(psidag,np.dot(CmpA,psi)))  #The expectation value itself, <psi|S|psi>
                yAlist.append(Faa[0,0])
                Fb=np.real(np.dot(psidag,np.dot(CmpB,psi)))  #The expectation value itself, <psi|S|psi>
                yBlist.append(Fb[0,0])

            if compnum==4: #Breaks <S> into four different components, ABP,AB'P', A'BP', and -A'B'P
                CmpA=np.dot(A,(np.dot(B,P)))
                CmpB=np.dot(A,(np.dot(Bp,Pp)))
                CmpC=np.dot(Ap,(np.dot(B,Pp)))
                CmpD=np.dot(Ap,(-np.dot(Bp,P)))

                Faa=np.real(np.dot(psidag,np.dot(CmpA,psi)))  
                yAlist.append(Faa[0,0])
                Fb=np.real(np.dot(psidag,np.dot(CmpB,psi)))  
                yBlist.append(Fb[0,0])
                Fc=np.real(np.dot(psidag,np.dot(CmpC,psi)))  
                yClist.append(Fb[0,0])
                Fd=np.real(np.dot(psidag,np.dot(CmpD,psi)))  
                yDlist.append(Fb[0,0])


            if compnum==8: #Breaks <S> into eight different components, same as above but with P,P' being C+C', C-C'

                CmpA=np.dot(A,(np.dot(B,C)))
                CmpB=np.dot(A,(np.dot(Bp,C)))
                CmpC=np.dot(Ap,(np.dot(B,C)))
                CmpD=np.dot(Ap,(-np.dot(Bp,C)))
                CmpA2=np.dot(A,(np.dot(B,Cp)))
                CmpB2=np.dot(A,(np.dot(Bp,-Cp)))
                CmpC2=np.dot(Ap,(np.dot(B,-Cp)))
                CmpD2=np.dot(Ap,(-np.dot(Bp,Cp)))

                Faa=np.real(np.dot(psidag,np.dot(CmpA,psi)))  
                yAlist.append(Faa[0,0])
                Fb=np.real(np.dot(psidag,np.dot(CmpB,psi)))  
                yBlist.append(Fb[0,0])
                Fc=np.real(np.dot(psidag,np.dot(CmpC,psi)))  
                yClist.append(Fb[0,0])
                Fd=np.real(np.dot(psidag,np.dot(CmpD,psi)))  
                yDlist.append(Fb[0,0])
                Faa2=np.real(np.dot(psidag,np.dot(CmpA2,psi)))  
                yElist.append(Faa2[0,0])
                Fb2=np.real(np.dot(psidag,np.dot(CmpB2,psi)))  
                yFlist.append(Fb2[0,0])
                Fc2=np.real(np.dot(psidag,np.dot(CmpC2,psi)))  
                yGlist.append(Fb2[0,0])
                Fd2=np.real(np.dot(psidag,np.dot(CmpD2,psi)))  
                yHlist.append(Fb2[0,0])
    


    lable="Maximum <S> "+name   #Plots the maximum violation as well as the components. 
    py.xlabel('Tangle')
    py.ylabel('<S> component max')
    py.xlim(-.01,1.01)
    py.scatter(xlist,ylist,s=10,c='black',linewidth=0.5,label=lable)
    py.legend()
    filename=pathname+"/Figures/SComp/"+name+"<S>comp max vs tgl"+str(n)+" , "+str(ln)+" max.png"
    py.savefig(filename,dpi=200)
    
    if compnum==2: #Plots each component in a seperate file in the folder SComp in Figures. 

        py.clf()
        lable="<A(BP+B'P')> "+name
        py.xlabel('Tangle')
        py.ylabel('<S> component max')
        py.xlim(-.01,1.01)
        py.scatter(xlist,yAlist,s=10,c='red',linewidth=0.5,label=lable)
        py.legend()
        filename=pathname+"/Figures/SComp/"+name+"<S>comp max vs tgl"+str(n)+" , "+str(ln)+" A2.png"
        py.savefig(filename,dpi=200)

        py.clf()
        lable="<A'(BP'-B'P)> "+name
        py.xlabel('Tangle')
        py.ylabel('<S> component max')
        py.xlim(-.01,1.01)
        py.scatter(xlist,yBlist,s=10,c='blue',linewidth=0.5,label=lable)
        py.legend()
        filename=pathname+"/Figures/SComp/"+name+"<S>comp max vs tgl"+str(n)+" , "+str(ln)+" B2.png"
        py.savefig(filename,dpi=200)

    if compnum==4:

        py.clf()
        lable="<ABP> "+name
        py.xlabel('Tangle')
        py.ylabel('<S> component max')
        py.xlim(-.01,1.01)
        py.scatter(xlist,yAlist,s=10,c='green',linewidth=0.5,label=lable)
        py.legend()
        filename=pathname+"/Figures/SComp/"+name+"<S>comp max vs tgl"+str(n)+" , "+str(ln)+" A4.png"
        py.savefig(filename,dpi=200)

        py.clf()
        lable="<AB'P'> "+name
        py.xlabel('Tangle')
        py.ylabel('<S> component max')
        py.xlim(-.01,1.01)
        py.scatter(xlist,yBlist,s=10,c='red',linewidth=0.5,label=lable)
        py.legend()
        filename=pathname+"/Figures/SComp/"+name+"<S>comp max vs tgl"+str(n)+" , "+str(ln)+" B4.png"
        py.savefig(filename,dpi=200)

        py.clf()
        lable="<A'BP'> "+name
        py.xlabel('Tangle')
        py.ylabel('<S> component max')
        py.xlim(-.01,1.01)
        py.scatter(xlist,yClist,s=10,c='blue',linewidth=0.5,label=lable)
        py.legend()
        filename=pathname+"/Figures/SComp/"+name+"<S>comp max vs tgl"+str(n)+" , "+str(ln)+" C4.png"
        py.savefig(filename,dpi=200)

        py.clf()
        lable="<-A'B'P> "+name
        py.xlabel('Tangle')
        py.ylabel('<S> component max')
        py.xlim(-.01,1.01)
        py.scatter(xlist,yDlist,s=10,c='orange',linewidth=0.5,label=lable)
        py.legend()
        filename=pathname+"/Figures/SComp/"+name+"<S>comp max vs tgl"+str(n)+" , "+str(ln)+" D4.png"
        py.savefig(filename,dpi=200)

    if compnum==8:

        py.clf()
        lable="<ABC> "+name
        py.xlabel('Tangle')
        py.ylabel('<S> component max')
        py.xlim(-.01,1.01)
        py.scatter(xlist,yAlist,s=10,c='red',linewidth=0.5,label=lable)
        py.legend()
        filename=pathname+"/Figures/SComp/"+name+"<S>comp max vs tgl"+str(n)+" , "+str(ln)+" A8.png"
        py.savefig(filename,dpi=200)

        py.clf()
        lable="<AB'C> "+name
        py.xlabel('Tangle')
        py.ylabel('<S> component max')
        py.xlim(-.01,1.01)
        py.scatter(xlist,yBlist,s=10,c='blue',linewidth=0.5,label=lable)
        py.legend()
        filename=pathname+"/Figures/SComp/"+name+"<S>comp max vs tgl"+str(n)+" , "+str(ln)+" B8.png"
        py.savefig(filename,dpi=200)

        py.clf()
        lable="<A'BC> "+name
        py.xlabel('Tangle')
        py.ylabel('<S> component max')
        py.xlim(-.01,1.01)
        py.scatter(xlist,yClist,s=10,c='orange',linewidth=0.5,label=lable)
        py.legend()
        filename=pathname+"/Figures/SComp/"+name+"<S>comp max vs tgl"+str(n)+" , "+str(ln)+" C8.png"
        py.savefig(filename,dpi=200)

        py.clf()
        lable="<-A'B'C> "+name
        py.xlabel('tangle')
        py.ylabel('<S> component max')
        py.xlim(-.01,1.01)
        py.scatter(xlist,yDlist,s=10,c='green',linewidth=0.5,label=lable)
        py.legend()
        filename=pathname+"/Figures/SComp/"+name+"<S>comp max vs tgl"+str(n)+" , "+str(ln)+" D8.png"
        py.savefig(filename,dpi=200)

        py.clf()
        lable="<ABC> "+name
        py.xlabel('Tangle')
        py.ylabel('<S> component max')
        py.xlim(-.01,1.01)
        py.scatter(xlist,yElist,s=10,c='red',linewidth=0.5,label=lable)
        py.legend()
        filename=pathname+"/Figures/SComp/"+name+"<S>comp max vs tgl"+str(n)+" , "+str(ln)+" E8.png"
        py.savefig(filename,dpi=200)

        py.clf()
        lable="<-AB'C'> "+name
        py.xlabel('Tangle')
        py.ylabel('<S> component max')
        py.xlim(-.01,1.01)
        py.scatter(xlist,yFlist,s=10,c='blue',linewidth=0.5,label=lable)
        py.legend()
        filename=pathname+"/Figures/SComp/"+name+"<S>comp max vs tgl"+str(n)+" , "+str(ln)+" F8.png"
        py.savefig(filename,dpi=200)

        py.clf()
        lable="<-A'BC'> "+name
        py.xlabel('Tangle')
        py.ylabel('<S> component max')
        py.xlim(-.01,1.01)
        py.scatter(xlist,yGlist,s=10,c='orange',linewidth=0.5,label=lable)
        py.legend()
        filename=pathname+"/Figures/SComp/"+name+"<S>comp max vs tgl"+str(n)+" , "+str(ln)+" G8.png"
        py.savefig(filename,dpi=200)

        py.clf()
        lable="<A'B'C'> "+name
        py.xlabel('Tangle')
        py.ylabel('<S> component max')
        py.xlim(-.01,1.01)
        py.scatter(xlist,yHlist,s=10,c='green',linewidth=0.5,label=lable)
        py.legend()
        filename=pathname+"/Figures/SComp/"+name+"<S>comp max vs tgl"+str(n)+" , "+str(ln)+" H8.png"
        py.savefig(filename,dpi=200)
    
    return




def p1Veccomp_theta_plot(name,n,ln): #This plots the components of each maximized unit vector in a seperate file. Saved in the folder VComp under Figures. Only for one parameter states.
    maxveclist=np.zeros(ln,dtype=np.ndarray) #Holds all the maximum unit vectors from the trials
    xlist=[]
    psilist=np.zeros(ln,dtype=np.ndarray) #Holds all the |psi> from the trials
    for q in range(ln):                 #sets up psilist and maxveclist with all the data needed to calculate
        psi,speciallist,maxlist,xoptalist=p1call_file(name,n,q,ln)
        psilist[q]=psi
        maxveclist[q]=maxlist

    alist=[] #Defines the lists that will be used to hold all the information
    aplist=[]
    bplist=[]
    blist=[]
    clist=[]
    cplist=[]
    Maxlist=np.zeros(ln)
    Xmaxlist=np.zeros(ln)

    for i in range (ln):    #Calculates theta and the max violation for each |psi>, along with calculating all the vector components and appending them to their repsecitive lists for each maximized vector
        theta=float(i)*2*np.pi/(ln-1)
        Xmaxlist[i]=theta
        for q in range (len(maxveclist[i])):
            
            op=maxveclist[i]
            x=op[q]


            av=np.array([sin(x[0])*cos(x[1]),sin(x[0])*sin(x[1]),cos(x[0])]) #defines the vector associated with each measurement
            avp=np.array([sin(x[2])*cos(x[3]),sin(x[2])*sin(x[3]),cos(x[2])])#bv=unit vector for B measurement
            bv=np.array([sin(x[4])*cos(x[5]),sin(x[4])*sin(x[5]),cos(x[4])])#avp=unit vector for A' measurement
            bvp=np.array([sin(x[6])*cos(x[7]),sin(x[6])*sin(x[7]),cos(x[6])])
            cv=np.array([sin(x[8])*cos(x[9]),sin(x[8])*sin(x[9]),cos(x[8])])
            cvp=np.array([sin(x[10])*cos(x[11]),sin(x[10])*sin(x[11]),cos(x[10])])

            av=av/np.linalg.norm(av)
            bv=bv/np.linalg.norm(bv)
            cv=cv/np.linalg.norm(cv)
            avp=avp/np.linalg.norm(avp)
            bvp=bvp/np.linalg.norm(bvp)
            cvp=cvp/np.linalg.norm(cvp)
            
            for j in range(3):
                alist.append(av[j])
                aplist.append(avp[j])
                blist.append(bv[j])
                bplist.append(bvp[j])
                clist.append(cv[j])
                cplist.append(cvp[j])
                xlist.append(theta)

        psi=psilist[i]
        Maxlist[i]=abs(Fa(x,psi))

    for i in range(6):     #This pulls out each element, x,y,z for each unit vector 
        Alist=[]
        Blist=[]
        Clist=[]
        Xlist=[]

        if i==0:
            cname="a"
            for j in range(int(len(xlist)/3.0)):
                Alist.append(alist[3*j-0])
                Blist.append(alist[3*j-1])
                Clist.append(alist[3*j-2])
                Xlist.append(xlist[3*j])
        if i==1:
            cname="ap"
            for j in range(int(len(xlist)/3.0)):
                Alist.append(aplist[3*j-0])
                Blist.append(aplist[3*j-1])
                Clist.append(aplist[3*j-2])
                Xlist.append(xlist[3*j])
        if i==2:
            cname="b"
            for j in range(int(len(xlist)/3.0)):
                Alist.append(blist[3*j-0])
                Blist.append(blist[3*j-1])
                Clist.append(blist[3*j-2])
                Xlist.append(xlist[3*j])
        if i==3:
            cname="bp"
            for j in range(int(len(xlist)/3.0)):
                Alist.append(bplist[3*j-0])
                Blist.append(bplist[3*j-1])
                Clist.append(bplist[3*j-2])
                Xlist.append(xlist[3*j])
        if i==4:
            cname="c"
            for j in range(int(len(xlist)/3.0)):
                Alist.append(clist[3*j-0])
                Blist.append(clist[3*j-1])
                Clist.append(clist[3*j-2])
                Xlist.append(xlist[3*j])
        if i==5:
            cname="cp"
            for j in range(int(len(xlist)/3.0)):
                Alist.append(cplist[3*j-0])
                Blist.append(cplist[3*j-1])
                Clist.append(cplist[3*j-2])
                Xlist.append(xlist[3*j])

        py.clf()
        v=[0,2*np.pi,-1.1,1.1] #This sets the axis to be constant so all the graphs can be overlayed without trouble.

        py.axis(v)              #These structures plot the data for each vector. It is iterated over the total number of vectors.
        Alabel=cname+"x"
        py.scatter(Xlist,Alist,s=10,c='r',linewidth=0.5,label=Alabel)
        py.xlabel('Theta')
        py.ylabel('Vector component')
        py.legend()
        figname=pathname+"/Figures/VComp/"+name+"vec comp vs theta"+str(n)+" , "+str(ln)+" "+Alabel+".png"
        py.savefig(figname,dpi=200)
        py.clf()
        Blabel=cname+"y"
        py.axis(v)

        py.scatter(Xlist,Blist,s=10,c='g',linewidth=0.5,label=Blabel)
        py.xlabel('Theta')
        py.ylabel('Vector component')
        py.legend()
        figname=pathname+"/Figures/VComp/"+name+"vec comp vs theta"+str(n)+" , "+str(ln)+" "+Blabel+".png"
        py.savefig(figname,dpi=200)
        py.clf()
        Clabel=cname+"z"
        py.axis(v)

        py.scatter(Xlist,Clist,s=10,c='b',linewidth=0.5,label=Clabel)
        py.xlabel('Theta')
        py.ylabel('Vector component')
        py.legend()
        figname=pathname+"/Figures/VComp/"+name+"vec comp vs theta"+str(n)+" , "+str(ln)+" "+Clabel+".png"
        py.savefig(figname,dpi=200)
        py.clf()


    py.xlim(0,2*np.pi)
    f = py.figure()
    ax = f.add_subplot(111)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    Maxlabel="<S>"
    py.plot(Xmaxlist,Maxlist,c='orange',label=Maxlabel)
    py.xlabel('Theta')
    py.ylabel('Maximum violation')
    py.legend()
    figname=pathname+"/Figures/VComp/"+name+"vec comp vs theta"+str(n)+" , "+str(ln)+" "+Maxlabel+".png"
    py.savefig(figname,dpi=200)
    py.clf()
    
    return





def p1Veccomp_tgl_plot(name,n,ln): #This is the same function as above, but plotting <S>max versus the three tangle rather than versus theta. Also only works on a one parameter state.
    maxveclist=np.zeros(ln,dtype=np.ndarray)
    xlist=[]
    psilist=np.zeros(ln,dtype=np.ndarray)
    for q in range(ln):
        psi,speciallist,maxlist,xoptalist=p1call_file(name,n,q,ln)
        psilist[q]=psi
        maxveclist[q]=maxlist

    alist=[]
    aplist=[]
    bplist=[]
    blist=[]
    clist=[]
    cplist=[]
    Maxlist=np.zeros(ln)
    Xmaxlist=np.zeros(ln)
    
    for i in range (len(maxveclist)):
        psi=psilist[i]
        d1=(psi[0,0]**2)*(psi[7,0]**2)+(psi[1,0]**2)*(psi[6,0]**2)+(psi[2,0]**2)*(psi[5,0]**2)+(psi[4,0]**2)*(psi[3,0]**2)
        d2=psi[0,0]*psi[7,0]*psi[3,0]*psi[4,0]+psi[0,0]*psi[7,0]*psi[5,0]*psi[2,0]+psi[0,0]*psi[7,0]*psi[6,0]*psi[1,0]+psi[3,0]*psi[4,0]*psi[5,0]*psi[2,0]+psi[3,0]*psi[4,0]*psi[6,0]*psi[1,0]+psi[5,0]*psi[2,0]*psi[6,0]*psi[1,0]
        d3=psi[0,0]*psi[6,0]*psi[5,0]*psi[3,0]+psi[7,0]*psi[1,0]*psi[2,0]*psi[4,0]
        Tgl=4*abs(d1-2*d2+4*d3)
        Xmaxlist[i]=Tgl
        for q in range (len(maxveclist[i])):
            
            op=maxveclist[i]
            x=op[q]


            av=np.array([sin(x[0])*cos(x[1]),sin(x[0])*sin(x[1]),cos(x[0])]) #defines the vector associated with each measurement
            avp=np.array([sin(x[2])*cos(x[3]),sin(x[2])*sin(x[3]),cos(x[2])])#bv=unit vector for B measurement
            bv=np.array([sin(x[4])*cos(x[5]),sin(x[4])*sin(x[5]),cos(x[4])])#avp=unit vector for A' measurement
            bvp=np.array([sin(x[6])*cos(x[7]),sin(x[6])*sin(x[7]),cos(x[6])])
            cv=np.array([sin(x[8])*cos(x[9]),sin(x[8])*sin(x[9]),cos(x[8])])
            cvp=np.array([sin(x[10])*cos(x[11]),sin(x[10])*sin(x[11]),cos(x[10])])

            av=av/np.linalg.norm(av)
            bv=bv/np.linalg.norm(bv)
            cv=cv/np.linalg.norm(cv)
            avp=avp/np.linalg.norm(avp)
            bvp=bvp/np.linalg.norm(bvp)
            cvp=cvp/np.linalg.norm(cvp)
            
            for j in range(3):
                alist.append(av[j])
                aplist.append(avp[j])
                blist.append(bv[j])
                bplist.append(bvp[j])
                clist.append(cv[j])
                cplist.append(cvp[j])
                xlist.append(Tgl)
        Maxlist[i]=abs(Fa(x,psi))
      
    for i in range(6):
        Alist=[]
        Blist=[]
        Clist=[]
        Xlist=[]

        if i==0:
            cname="a"
            for j in range(int(len(xlist)/3.0)):
                Alist.append(alist[3*j-0])
                Blist.append(alist[3*j-1])
                Clist.append(alist[3*j-2])
                Xlist.append(xlist[3*j])
        if i==1:
            cname="ap"
            for j in range(int(len(xlist)/3.0)):
                Alist.append(aplist[3*j-0])
                Blist.append(aplist[3*j-1])
                Clist.append(aplist[3*j-2])
                Xlist.append(xlist[3*j])
        if i==2:
            cname="b"
            for j in range(int(len(xlist)/3.0)):
                Alist.append(blist[3*j-0])
                Blist.append(blist[3*j-1])
                Clist.append(blist[3*j-2])
                Xlist.append(xlist[3*j])
        if i==3:
            cname="bp"
            for j in range(int(len(xlist)/3.0)):
                Alist.append(bplist[3*j-0])
                Blist.append(bplist[3*j-1])
                Clist.append(bplist[3*j-2])
                Xlist.append(xlist[3*j])
        if i==4:
            cname="c"
            for j in range(int(len(xlist)/3.0)):
                Alist.append(clist[3*j-0])
                Blist.append(clist[3*j-1])
                Clist.append(clist[3*j-2])
                Xlist.append(xlist[3*j])
        if i==5:
            cname="cp"
            for j in range(int(len(xlist)/3.0)):
                Alist.append(cplist[3*j-0])
                Blist.append(cplist[3*j-1])
                Clist.append(cplist[3*j-2])
                Xlist.append(xlist[3*j])


        py.clf()
        
        v=[-.1,1.1,-1.1,1.1]

        py.axis(v)
        Alabel=cname+"x"

        py.scatter(Xlist,Alist,s=10,c='r',linewidth=0.5,label=Alabel)
        py.xlabel('Tangle')
        py.ylabel('Vector component')
        py.legend()
        figname=pathname+"/Figures/VComp/"+name+"vec comp vs tangle"+str(n)+" , "+str(ln)+" "+Alabel+".png"
        py.savefig(figname,dpi=200)
        py.clf()
        Blabel=cname+"y"
        py.axis(v)

        py.scatter(Xlist,Blist,s=10,c='g',linewidth=0.5,label=Blabel)
        py.xlabel('Tangle')
        py.ylabel('Vector component')
        py.legend()
        figname=pathname+"/Figures/VComp/"+name+"vec comp vs tangle"+str(n)+" , "+str(ln)+" "+Blabel+".png"
        py.savefig(figname,dpi=200)
        py.clf()
        Clabel=cname+"z"
        py.axis(v)

        py.scatter(Xlist,Clist,s=10,c='b',linewidth=0.5,label=Clabel)
        py.xlabel('Tangle')
        py.ylabel('Vector component')
        py.legend()
        figname=pathname+"/Figures/VComp/"+name+"vec comp vs tangle"+str(n)+" , "+str(ln)+" "+Clabel+".png"
        py.savefig(figname,dpi=200)
        py.clf()

    f = py.figure()
    ax = f.add_subplot(111)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    py.xlim(-.1,1.1)
    Maxlabel="<S>"
    py.plot(Xmaxlist,Maxlist,c='orange',label=Maxlabel)
    py.xlabel('Three Tangle')
    py.ylabel('Maximum violation')
    py.legend()
    figname=pathname+"/Figures/VComp/"+name+"vec comp vs tangle"+str(n)+" , "+str(ln)+" "+Maxlabel+".png"
    py.savefig(figname,dpi=200)
    py.clf()

    return



def p2S_max_tgl_plot(name,n,ln,save,clear,color): #Plots the maximum <S> violation versus the three tangle for a two parameter system.
    if clear==1:
        py.clf()
    filename=pathname+"/Figures/"+name+"<S>max vs tgl"+str(n)+" , "+str(ln)+" .png"
    length=int(float(ln)**2)
    ylist=np.zeros(length)
    xlist=np.zeros(length)
    j=0
    Ftest=0.0
    for i in range(ln):
        print 100.0*float(i)/float(ln)
        for q in range(ln):
            psi,speciallist,maxlist,xoptalist=p2call_file(name,n,i,q,ln)
            Fptst=0.0
            x=maxlist[0]
            Ftst=abs(Fa(x,psi))
            ylist[j]=Ftst

            d1=(psi[0,0]**2)*(psi[7,0]**2)+(psi[1,0]**2)*(psi[6,0]**2)+(psi[2,0]**2)*(psi[5,0]**2)+(psi[4,0]**2)*(psi[3,0]**2)
            d2=psi[0,0]*psi[7,0]*psi[3,0]*psi[4,0]+psi[0,0]*psi[7,0]*psi[5,0]*psi[2,0]+psi[0,0]*psi[7,0]*psi[6,0]*psi[1,0]+psi[3,0]*psi[4,0]*psi[5,0]*psi[2,0]+psi[3,0]*psi[4,0]*psi[6,0]*psi[1,0]+psi[5,0]*psi[2,0]*psi[6,0]*psi[1,0]
            d3=psi[0,0]*psi[6,0]*psi[5,0]*psi[3,0]+psi[7,0]*psi[1,0]*psi[2,0]*psi[4,0]
            Tgl=4*abs(d1-2*d2+4*d3)
            xlist[j]=Tgl
            j=j+1

    lable=name
    
    py.scatter(xlist,ylist,s=10,c=color,linewidth=0.5,label=lable)
    
    py.xlabel('Three Tangle')
    py.ylabel('<S> Max')
    py.legend()
    if save==1:
        py.savefig(filename,dpi=200)
    print Ftest
    return



def p2Veccomp_tgl_plot(name,n,ln): #This is the same as p1Veccomp_tgl_plot(name,n,ln), but works only for two parameter states.
    length=float(ln)**2
    maxveclist=np.zeros(length,dtype=np.ndarray)
    xlist=[]
    psilist=np.zeros(length,dtype=np.ndarray)
    k=0
    for q in range(ln):
        for j in range(ln):
            psi,speciallist,maxlist,xoptalist=p2call_file(name,n,q,j,ln)
            psilist[k]=psi
            maxveclist[k]=maxlist
            k=k+1

    alist=[]
    aplist=[]
    bplist=[]
    blist=[]
    clist=[]
    cplist=[]
    Maxlist=np.zeros(length)
    Xmaxlist=np.zeros(length)
    
    for i in range (len(maxveclist)):
        psi=psilist[i]
        d1=(psi[0,0]**2)*(psi[7,0]**2)+(psi[1,0]**2)*(psi[6,0]**2)+(psi[2,0]**2)*(psi[5,0]**2)+(psi[4,0]**2)*(psi[3,0]**2)
        d2=psi[0,0]*psi[7,0]*psi[3,0]*psi[4,0]+psi[0,0]*psi[7,0]*psi[5,0]*psi[2,0]+psi[0,0]*psi[7,0]*psi[6,0]*psi[1,0]+psi[3,0]*psi[4,0]*psi[5,0]*psi[2,0]+psi[3,0]*psi[4,0]*psi[6,0]*psi[1,0]+psi[5,0]*psi[2,0]*psi[6,0]*psi[1,0]
        d3=psi[0,0]*psi[6,0]*psi[5,0]*psi[3,0]+psi[7,0]*psi[1,0]*psi[2,0]*psi[4,0]
        Tgl=4*abs(d1-2*d2+4*d3)
        Xmaxlist[i]=Tgl
        for q in range (len(maxveclist[i])):
            
            op=maxveclist[i]
            x=op[q]


            av=np.array([sin(x[0])*cos(x[1]),sin(x[0])*sin(x[1]),cos(x[0])]) #defines the vector associated with each measurement
            avp=np.array([sin(x[2])*cos(x[3]),sin(x[2])*sin(x[3]),cos(x[2])])#bv=unit vector for B measurement
            bv=np.array([sin(x[4])*cos(x[5]),sin(x[4])*sin(x[5]),cos(x[4])])#avp=unit vector for A' measurement
            bvp=np.array([sin(x[6])*cos(x[7]),sin(x[6])*sin(x[7]),cos(x[6])])
            cv=np.array([sin(x[8])*cos(x[9]),sin(x[8])*sin(x[9]),cos(x[8])])
            cvp=np.array([sin(x[10])*cos(x[11]),sin(x[10])*sin(x[11]),cos(x[10])])

            av=av/np.linalg.norm(av)
            bv=bv/np.linalg.norm(bv)
            cv=cv/np.linalg.norm(cv)
            avp=avp/np.linalg.norm(avp)
            bvp=bvp/np.linalg.norm(bvp)
            cvp=cvp/np.linalg.norm(cvp)
            
            for j in range(3):
                alist.append(av[j])
                aplist.append(avp[j])
                blist.append(bv[j])
                bplist.append(bvp[j])
                clist.append(cv[j])
                cplist.append(cvp[j])
                xlist.append(Tgl)
        Maxlist[i]=abs(Fa(x,psi))
      
    for i in range(6):
        Alist=[]
        Blist=[]
        Clist=[]
        Xlist=[]

        if i==0:
            cname="a"
            for j in range(int(len(xlist)/3.0)):
                Alist.append(alist[3*j-0])
                Blist.append(alist[3*j-1])
                Clist.append(alist[3*j-2])
                Xlist.append(xlist[3*j])
        if i==1:
            cname="ap"
            for j in range(int(len(xlist)/3.0)):
                Alist.append(aplist[3*j-0])
                Blist.append(aplist[3*j-1])
                Clist.append(aplist[3*j-2])
                Xlist.append(xlist[3*j])
        if i==2:
            cname="b"
            for j in range(int(len(xlist)/3.0)):
                Alist.append(blist[3*j-0])
                Blist.append(blist[3*j-1])
                Clist.append(blist[3*j-2])
                Xlist.append(xlist[3*j])
        if i==3:
            cname="bp"
            for j in range(int(len(xlist)/3.0)):
                Alist.append(bplist[3*j-0])
                Blist.append(bplist[3*j-1])
                Clist.append(bplist[3*j-2])
                Xlist.append(xlist[3*j])
        if i==4:
            cname="c"
            for j in range(int(len(xlist)/3.0)):
                Alist.append(clist[3*j-0])
                Blist.append(clist[3*j-1])
                Clist.append(clist[3*j-2])
                Xlist.append(xlist[3*j])
        if i==5:
            cname="cp"
            for j in range(int(len(xlist)/3.0)):
                Alist.append(cplist[3*j-0])
                Blist.append(cplist[3*j-1])
                Clist.append(cplist[3*j-2])
                Xlist.append(xlist[3*j])


        py.clf()
        
        v=[-.1,1.1,-1.1,1.1]

        py.axis(v)
        Alabel=cname+"x"

        py.scatter(Xlist,Alist,s=10,c='r',linewidth=0.5,label=Alabel)
        py.xlabel('Tangle')
        py.ylabel('Vector component')
        py.legend()
        figname=pathname+"/Figures/VComp/"+name+"vec comp vs tangle"+str(n)+" , "+str(ln)+" "+Alabel+".png"
        py.savefig(figname,dpi=200)
        py.clf()
        Blabel=cname+"y"
        py.axis(v)

        py.scatter(Xlist,Blist,s=10,c='g',linewidth=0.5,label=Blabel)
        py.xlabel('Tangle')
        py.ylabel('Vector component')
        py.legend()
        figname=pathname+"/Figures/VComp/"+name+"vec comp vs tangle"+str(n)+" , "+str(ln)+" "+Blabel+".png"
        py.savefig(figname,dpi=200)
        py.clf()
        Clabel=cname+"z"
        py.axis(v)

        py.scatter(Xlist,Clist,s=10,c='b',linewidth=0.5,label=Clabel)
        py.xlabel('Tangle')
        py.ylabel('Vector component')
        py.legend()
        figname=pathname+"/Figures/VComp/"+name+"vec comp vs tangle"+str(n)+" , "+str(ln)+" "+Clabel+".png"
        py.savefig(figname,dpi=200)
        py.clf()

    f = py.figure()
    ax = f.add_subplot(111)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    py.xlim(-.1,1.1)
    Maxlabel="<S>"
    py.plot(Xmaxlist,Maxlist,c='orange',label=Maxlabel)
    py.xlabel('Three Tangle')
    py.ylabel('Maximum violation')
    py.legend()
    figname=pathname+"/Figures/VComp/"+name+"vec comp vs tangle"+str(n)+" , "+str(ln)+" "+Maxlabel+".png"
    py.savefig(figname,dpi=200)
    py.clf()

    return


def p2Scomp_tgl_plot(name,n,ln,compnum): #This is the same as p1Scomp_tgl_plot(name,n,ln,compnum), but works only for two parameter states.

    py.clf()

    ylist=[]
    xlist=[]

    yAlist=[]
    yBlist=[]
    yClist=[]
    yDlist=[]
    yElist=[]
    yFlist=[]
    yGlist=[]
    yHlist=[]

    
    for q in range(ln):
        for i in range(ln):
            psi,speciallist,maxlist,xoptalist=p2call_file(name,n,q,i,ln)

            for j in range(len(maxlist)):
                x=maxlist[j]
                Fmax=abs(Fa(x,psi))
                ylist.append(Fmax)
            
                d1=(psi[0,0]**2)*(psi[7,0]**2)+(psi[1,0]**2)*(psi[6,0]**2)+(psi[2,0]**2)*(psi[5,0]**2)+(psi[4,0]**2)*(psi[3,0]**2)
                d2=psi[0,0]*psi[7,0]*psi[3,0]*psi[4,0]+psi[0,0]*psi[7,0]*psi[5,0]*psi[2,0]+psi[0,0]*psi[7,0]*psi[6,0]*psi[1,0]+psi[3,0]*psi[4,0]*psi[5,0]*psi[2,0]+psi[3,0]*psi[4,0]*psi[6,0]*psi[1,0]+psi[5,0]*psi[2,0]*psi[6,0]*psi[1,0]
                d3=psi[0,0]*psi[6,0]*psi[5,0]*psi[3,0]+psi[7,0]*psi[1,0]*psi[2,0]*psi[4,0]
                Tgl=4*abs(d1-2*d2+4*d3)
                
                xlist.append(Tgl)
                            
                #Calculates the componenets of <S> 
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

                
                psidag=(psi.conjugate()).transpose() #|psi>dagge

                if compnum==2: #Breaks <S> into two different components, A(BP+B'P') and A'(BP'-B'P)

                    S=np.dot(A,(np.dot(B,P)+np.dot(Bp,Pp)))+np.dot(Ap,(np.dot(B,Pp)-np.dot(Bp,P)))  #The Schvetlincthy operator 

                    CmpA=np.dot(A,(np.dot(B,P)+np.dot(Bp,Pp)))
                    CmpB=np.dot(Ap,(np.dot(B,Pp)-np.dot(Bp,P)))

                    Faa=np.real(np.dot(psidag,np.dot(CmpA,psi)))  #The expectation value itself, <psi|S|psi>
                    yAlist.append(Faa[0,0])
                    Fb=np.real(np.dot(psidag,np.dot(CmpB,psi)))  #The expectation value itself, <psi|S|psi>
                    yBlist.append(Fb[0,0])

                if compnum==4:
                    CmpA=np.dot(A,(np.dot(B,P)))
                    CmpB=np.dot(A,(np.dot(Bp,Pp)))
                    CmpC=np.dot(Ap,(np.dot(B,Pp)))
                    CmpD=np.dot(Ap,(-np.dot(Bp,P)))

                    Faa=np.real(np.dot(psidag,np.dot(CmpA,psi)))  
                    yAlist.append(Faa[0,0])
                    Fb=np.real(np.dot(psidag,np.dot(CmpB,psi)))  
                    yBlist.append(Fb[0,0])
                    Fc=np.real(np.dot(psidag,np.dot(CmpC,psi)))  
                    yClist.append(Fb[0,0])
                    Fd=np.real(np.dot(psidag,np.dot(CmpD,psi)))  
                    yDlist.append(Fb[0,0])

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
                    yAlist.append(Faa[0,0])
                    Fb=np.real(np.dot(psidag,np.dot(CmpB,psi)))  
                    yBlist.append(Fb[0,0])
                    Fc=np.real(np.dot(psidag,np.dot(CmpC,psi)))  
                    yClist.append(Fb[0,0])
                    Fd=np.real(np.dot(psidag,np.dot(CmpD,psi)))  
                    yDlist.append(Fb[0,0])
                    Faa2=np.real(np.dot(psidag,np.dot(CmpA2,psi)))  
                    yElist.append(Faa2[0,0])
                    Fb2=np.real(np.dot(psidag,np.dot(CmpB2,psi)))  
                    yFlist.append(Fb2[0,0])
                    Fc2=np.real(np.dot(psidag,np.dot(CmpC2,psi)))  
                    yGlist.append(Fb2[0,0])
                    Fd2=np.real(np.dot(psidag,np.dot(CmpD2,psi)))  
                    yHlist.append(Fb2[0,0])


    lable="Maximum <S> "+name
    py.xlabel('Tangle')
    py.ylabel('<S> component max')
    py.xlim(-.01,1.01)
    py.scatter(xlist,ylist,s=10,c='black',linewidth=0.5,label=lable)
    py.legend()
    filename=pathname+"/Figures/SComp/"+name+"<S>comp max vs tgl"+str(n)+" , "+str(ln)+" max.png"
    py.savefig(filename,dpi=200)
    
    if compnum==2:

        py.clf()
        lable="<A(BP+B'P')> "+name
        py.xlabel('Tangle')
        py.ylabel('<S> component max')
        py.xlim(-.01,1.01)
        py.scatter(xlist,yAlist,s=10,c='red',linewidth=0.5,label=lable)
        py.legend()
        filename=pathname+"/Figures/SComp/"+name+"<S>comp max vs tgl"+str(n)+" , "+str(ln)+" A2.png"
        py.savefig(filename,dpi=200)

        py.clf()
        lable="<A'(BP'-B'P)> "+name
        py.xlabel('Tangle')
        py.ylabel('<S> component max')
        py.xlim(-.01,1.01)
        py.scatter(xlist,yBlist,s=10,c='blue',linewidth=0.5,label=lable)
        py.legend()
        filename=pathname+"/Figures/SComp/"+name+"<S>comp max vs tgl"+str(n)+" , "+str(ln)+" B2.png"
        py.savefig(filename,dpi=200)

    if compnum==4:

        py.clf()
        lable="<ABP> "+name
        py.xlabel('Tangle')
        py.ylabel('<S> component max')
        py.xlim(-.01,1.01)
        py.scatter(xlist,yAlist,s=10,c='green',linewidth=0.5,label=lable)
        py.legend()
        filename=pathname+"/Figures/SComp/"+name+"<S>comp max vs tgl"+str(n)+" , "+str(ln)+" A4.png"
        py.savefig(filename,dpi=200)

        py.clf()
        lable="<AB'P'> "+name
        py.xlabel('Tangle')
        py.ylabel('<S> component max')
        py.xlim(-.01,1.01)
        py.scatter(xlist,yBlist,s=10,c='red',linewidth=0.5,label=lable)
        py.legend()
        filename=pathname+"/Figures/SComp/"+name+"<S>comp max vs tgl"+str(n)+" , "+str(ln)+" B4.png"
        py.savefig(filename,dpi=200)

        py.clf()
        lable="<A'BP'> "+name
        py.xlabel('Tangle')
        py.ylabel('<S> component max')
        py.xlim(-.01,1.01)
        py.scatter(xlist,yClist,s=10,c='blue',linewidth=0.5,label=lable)
        py.legend()
        filename=pathname+"/Figures/SComp/"+name+"<S>comp max vs tgl"+str(n)+" , "+str(ln)+" C4.png"
        py.savefig(filename,dpi=200)

        py.clf()
        lable="<-A'B'P> "+name
        py.xlabel('Tangle')
        py.ylabel('<S> component max')
        py.xlim(-.01,1.01)
        py.scatter(xlist,yDlist,s=10,c='orange',linewidth=0.5,label=lable)
        py.legend()
        filename=pathname+"/Figures/SComp/"+name+"<S>comp max vs tgl"+str(n)+" , "+str(ln)+" D4.png"
        py.savefig(filename,dpi=200)

    if compnum==8:

        py.clf()
        lable="<ABC> "+name
        py.xlabel('Tangle')
        py.ylabel('<S> component max')
        py.xlim(-.01,1.01)
        py.scatter(xlist,yAlist,s=10,c='red',linewidth=0.5,label=lable)
        py.legend()
        filename=pathname+"/Figures/SComp/"+name+"<S>comp max vs tgl"+str(n)+" , "+str(ln)+" A8.png"
        py.savefig(filename,dpi=200)

        py.clf()
        lable="<AB'C> "+name
        py.xlabel('Tangle')
        py.ylabel('<S> component max')
        py.xlim(-.01,1.01)
        py.scatter(xlist,yBlist,s=10,c='blue',linewidth=0.5,label=lable)
        py.legend()
        filename=pathname+"/Figures/SComp/"+name+"<S>comp max vs tgl"+str(n)+" , "+str(ln)+" B8.png"
        py.savefig(filename,dpi=200)

        py.clf()
        lable="<A'BC> "+name
        py.xlabel('Tangle')
        py.ylabel('<S> component max')
        py.xlim(-.01,1.01)
        py.scatter(xlist,yClist,s=10,c='orange',linewidth=0.5,label=lable)
        py.legend()
        filename=pathname+"/Figures/SComp/"+name+"<S>comp max vs tgl"+str(n)+" , "+str(ln)+" C8.png"
        py.savefig(filename,dpi=200)

        py.clf()
        lable="<-A'B'C> "+name
        py.xlabel('Tangle')
        py.ylabel('<S> component max')
        py.xlim(-.01,1.01)
        py.scatter(xlist,yDlist,s=10,c='green',linewidth=0.5,label=lable)
        py.legend()
        filename=pathname+"/Figures/SComp/"+name+"<S>comp max vs tgl"+str(n)+" , "+str(ln)+" D8.png"
        py.savefig(filename,dpi=200)

        py.clf()
        lable="<ABC> "+name
        py.xlabel('Tangle')
        py.ylabel('<S> component max')
        py.xlim(-.01,1.01)
        py.scatter(xlist,yElist,s=10,c='red',linewidth=0.5,label=lable)
        py.legend()
        filename=pathname+"/Figures/SComp/"+name+"<S>comp max vs tgl"+str(n)+" , "+str(ln)+" E8.png"
        py.savefig(filename,dpi=200)

        py.clf()
        lable="<-AB'C'> "+name
        py.xlabel('Tangle')
        py.ylabel('<S> component max')
        py.xlim(-.01,1.01)
        py.scatter(xlist,yFlist,s=10,c='blue',linewidth=0.5,label=lable)
        py.legend()
        filename=pathname+"/Figures/SComp/"+name+"<S>comp max vs tgl"+str(n)+" , "+str(ln)+" F8.png"
        py.savefig(filename,dpi=200)

        py.clf()
        lable="<-A'BC'> "+name
        py.xlabel('Tangle')
        py.ylabel('<S> component max')
        py.xlim(-.01,1.01)
        py.scatter(xlist,yGlist,s=10,c='orange',linewidth=0.5,label=lable)
        py.legend()
        filename=pathname+"/Figures/SComp/"+name+"<S>comp max vs tgl"+str(n)+" , "+str(ln)+" G8.png"
        py.savefig(filename,dpi=200)

        py.clf()
        lable="<A'B'C'> "+name
        py.xlabel('Tangle')
        py.ylabel('<S> component max')
        py.xlim(-.01,1.01)
        py.scatter(xlist,yHlist,s=10,c='green',linewidth=0.5,label=lable)
        py.legend()
        filename=pathname+"/Figures/SComp/"+name+"<S>comp max vs tgl"+str(n)+" , "+str(ln)+" H8.png"
        py.savefig(filename,dpi=200)
    
    return






def p2S_max_theta_plot(name,n,ln,save,clear,color): #This plots the <S> violation versus theta for a two parameter state, calculating it over ever value of the second parameter, phi.
    if clear==1:
        py.clf()
    filename=pathname+"/Figures/"+name+"<S>max vs theta"+str(n)+" , "+str(ln)+" .png"
    length=int(float(ln)**2)
    ylist=np.zeros(length)
    xlist=np.zeros(length)
    j=0
    Ftest=0.0
    for i in range(ln):
        print 100.0*float(i)/float(ln)
        theta=float(i)*2*np.pi/(ln-1)
        for q in range(ln):
            psi,speciallist,maxlist,xoptalist=p2call_file(name,n,i,q,ln)
            Fptst=0.0
            x=maxlist[0]
            Ftst=abs(Fa(x,psi))
            ylist[j]=Ftst

            xlist[j]=theta
            j=j+1

    lable=name
    
    py.scatter(xlist,ylist,s=10,c=color,linewidth=0.5,label=lable)
    
    py.xlabel('Theta')
    py.ylabel('<S> Max')
    py.legend()
    if save==1:
        py.savefig(filename,dpi=200)
    print Ftest
    return






def p2S_max_phi_plot(name,n,ln,save,clear,color): #This does the same as above, but plots with respect to the second parameter phi, rather than theta.
    if clear==1:
        py.clf()
    filename=pathname+"/Figures/"+name+"<S>max vs phi"+str(n)+" , "+str(ln)+" .png"
    length=int(float(ln)**2)
    ylist=np.zeros(length)
    xlist=np.zeros(length)
    j=0
    Ftest=0.0
    for i in range(ln):
        print 100.0*float(i)/float(ln)
        phi=float(i)*2*np.pi/(ln-1)
        for q in range(ln):
            psi,speciallist,maxlist,xoptalist=p2call_file(name,n,q,i,ln)
            Fptst=0.0
            x=maxlist[0]
            Ftst=abs(Fa(x,psi))
            ylist[j]=Ftst

            xlist[j]=phi
            j=j+1

    lable=name
    
    py.scatter(xlist,ylist,s=10,c=color,linewidth=0.5,label=lable)
    
    py.xlabel('Phi')
    py.ylabel('<S> Max')
    py.legend()
    if save==1:
        py.savefig(filename,dpi=200)
    print Ftest
    return





def p2Scomp_compare_tgl_plot(name,n,ln): #This plots all the breakdowns of <S> with respect to tangle on one graph for a two parameter state. 

    py.clf()

    ylist=[]
    xlist=[]

    yA2list=[]
    yA4list=[]
    yA8list=[]


    
    for q in range(ln):
        for i in range(ln):
            psi,speciallist,maxlist,xoptalist=p2call_file(name,n,q,i,ln)

            for j in range(len(maxlist)):
                x=maxlist[j]
                Fmax=abs(Fa(x,psi))
                ylist.append(Fmax)
            
                d1=(psi[0,0]**2)*(psi[7,0]**2)+(psi[1,0]**2)*(psi[6,0]**2)+(psi[2,0]**2)*(psi[5,0]**2)+(psi[4,0]**2)*(psi[3,0]**2)
                d2=psi[0,0]*psi[7,0]*psi[3,0]*psi[4,0]+psi[0,0]*psi[7,0]*psi[5,0]*psi[2,0]+psi[0,0]*psi[7,0]*psi[6,0]*psi[1,0]+psi[3,0]*psi[4,0]*psi[5,0]*psi[2,0]+psi[3,0]*psi[4,0]*psi[6,0]*psi[1,0]+psi[5,0]*psi[2,0]*psi[6,0]*psi[1,0]
                d3=psi[0,0]*psi[6,0]*psi[5,0]*psi[3,0]+psi[7,0]*psi[1,0]*psi[2,0]*psi[4,0]
                Tgl=4*abs(d1-2*d2+4*d3)
                
                xlist.append(Tgl)
                            
                #Calculates the componenets of <S> 
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

                
                psidag=(psi.conjugate()).transpose() #|psi>dagge


                S=np.dot(A,(np.dot(B,P)+np.dot(Bp,Pp)))+np.dot(Ap,(np.dot(B,Pp)-np.dot(Bp,P)))  #The Schvetlincthy operator 

                CmpA=np.dot(A,(np.dot(B,P)+np.dot(Bp,Pp)))

                Faa=np.real(np.dot(psidag,np.dot(CmpA,psi)))  #The expectation value itself, <psi|S|psi>
                yA2list.append(Faa[0,0])

                CmpA=np.dot(A,(np.dot(B,P)))

                Faa=np.real(np.dot(psidag,np.dot(CmpA,psi)))  
                yA4list.append(Faa[0,0])

                CmpA=np.dot(A,(np.dot(B,C)))

                Faa=np.real(np.dot(psidag,np.dot(CmpA,psi)))  
                yA8list.append(Faa[0,0])


    lable="Maximum <S> "+name
    py.xlabel('Tangle')
    py.ylabel('<S> component max')
    py.xlim(-.01,1.01)
    py.scatter(xlist,ylist,s=10,c='black',linewidth=0.5,label=lable)

    lable="<A(BP+B'P')> "+name

    py.xlim(-.01,1.01)
    py.scatter(xlist,yA2list,s=10,c='red',linewidth=0.5,label=lable)

    lable="<ABP> "+name

    py.xlim(-.01,1.01)
    py.scatter(xlist,yA4list,s=10,c='green',linewidth=0.5,label=lable)

    lable="<ABC> "+name
    py.xlim(-.01,1.01)
    py.scatter(xlist,yA8list,s=10,c='blue',linewidth=0.5,label=lable)
    py.legend()
    filename=pathname+"/Figures/SComp/"+name+"<S>comp compare vs tgl"+str(n)+" , "+str(ln)+".png"
    py.savefig(filename,dpi=200)
    
    return


def p1Scomp_compare_tgl_plot(name,n,ln): #This plots all the breakdowns of <S> with respect to tangle on one graph for a one parameter state. 

    py.clf()

    ylist=[]
    xlist=[]

    yA2list=[]
    yA4list=[]
    yA8list=[]


    
    for i in range(ln):
        psi,speciallist,maxlist,xoptalist=p1call_file(name,n,i,ln)

        for j in range(len(maxlist)):
            x=maxlist[j]
            Fmax=abs(Fa(x,psi))
            ylist.append(Fmax)
        
            d1=(psi[0,0]**2)*(psi[7,0]**2)+(psi[1,0]**2)*(psi[6,0]**2)+(psi[2,0]**2)*(psi[5,0]**2)+(psi[4,0]**2)*(psi[3,0]**2)
            d2=psi[0,0]*psi[7,0]*psi[3,0]*psi[4,0]+psi[0,0]*psi[7,0]*psi[5,0]*psi[2,0]+psi[0,0]*psi[7,0]*psi[6,0]*psi[1,0]+psi[3,0]*psi[4,0]*psi[5,0]*psi[2,0]+psi[3,0]*psi[4,0]*psi[6,0]*psi[1,0]+psi[5,0]*psi[2,0]*psi[6,0]*psi[1,0]
            d3=psi[0,0]*psi[6,0]*psi[5,0]*psi[3,0]+psi[7,0]*psi[1,0]*psi[2,0]*psi[4,0]
            Tgl=4*abs(d1-2*d2+4*d3)
            
            xlist.append(Tgl)
                        
            #Calculates the componenets of <S> 
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

            
            psidag=(psi.conjugate()).transpose() #|psi>dagge


            S=np.dot(A,(np.dot(B,P)+np.dot(Bp,Pp)))+np.dot(Ap,(np.dot(B,Pp)-np.dot(Bp,P)))  #The Schvetlincthy operator 

            CmpA=np.dot(A,(np.dot(B,P)+np.dot(Bp,Pp)))

            Faa=np.real(np.dot(psidag,np.dot(CmpA,psi)))  #The expectation value itself, <psi|S|psi>
            yA2list.append(Faa[0,0])

            CmpA=np.dot(A,(np.dot(B,P)))

            Faa=np.real(np.dot(psidag,np.dot(CmpA,psi)))  
            yA4list.append(Faa[0,0])

            CmpA=np.dot(A,(np.dot(B,C)))

            Faa=np.real(np.dot(psidag,np.dot(CmpA,psi)))  
            yA8list.append(Faa[0,0])


    lable="Maximum <S> "+name
    py.xlabel('Tangle')
    py.ylabel('<S> component max')
    py.xlim(-.01,1.01)
    py.scatter(xlist,ylist,s=10,c='black',linewidth=0.5,label=lable)


    lable="<A(BP+B'P')> "+name

    py.xlim(-.01,1.01)
    py.scatter(xlist,yA2list,s=10,c='red',linewidth=0.5,label=lable)

    lable="<ABP> "+name

    py.xlim(-.01,1.01)
    py.scatter(xlist,yA4list,s=10,c='green',linewidth=0.5,label=lable)

    lable="<ABC> "+name
    py.xlim(-.01,1.01)
    py.scatter(xlist,yA8list,s=10,c='blue',linewidth=0.5,label=lable)
    py.legend()
    filename=pathname+"/Figures/SComp/"+name+"<S>comp compare vs tgl"+str(n)+" , "+str(ln)+".png"
    py.savefig(filename,dpi=200)
    
    return






def p1Scomp_compare_theta_plot(name,n,ln): #This plots all the breakdowns of <S> with respect to theta on one graph for a two parameter state. 
    py.clf()
    
    ylist=[]
    xlist=[]

    yA2list=[]
    yA4list=[]
    yA8list=[]
        
    for i in range(ln):
        psi,speciallist,maxlist,xoptalist=p1call_file(name,n,i,ln)
        
        for j in range(len(maxlist)):
            x=maxlist[j]
            theta=float(i)*2*np.pi/(ln-1)
            Fmax=abs(Fa(x,psi))
    
            xlist.append(theta)
            ylist.append(Fmax)
        
            #Calculates the componenets of <S> 
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

            
            psidag=(psi.conjugate()).transpose() #|psi>dagge


            CmpA=np.dot(A,(np.dot(B,P)+np.dot(Bp,Pp)))

            Faa=np.real(np.dot(psidag,np.dot(CmpA,psi)))  #The expectation value itself, <psi|S|psi>
            yA2list.append(Faa[0,0])

            CmpA=np.dot(A,(np.dot(B,P)))

            Faa=np.real(np.dot(psidag,np.dot(CmpA,psi)))  
            yA4list.append(Faa[0,0])
            
            CmpA=np.dot(A,(np.dot(B,C)))

            Faa=np.real(np.dot(psidag,np.dot(CmpA,psi)))  
            yA8list.append(Faa[0,0])

    lable="Maximum <S> "+name
    py.xlabel('Theta')
    py.ylabel('<S> component max')
    py.xlim(0,2*np.pi)
    py.scatter(xlist,ylist,s=10,c='black',linewidth=0.5,label=lable)
    
    lable="<A(BP+B'P')> "+name
    py.xlim(0,2*np.pi)
    py.scatter(xlist,yA2list,s=10,c='red',linewidth=0.5,label=lable)

    lable="<ABP> "+name
    py.xlim(0,2*np.pi)
    py.scatter(xlist,yA4list,s=10,c='blue',linewidth=0.5,label=lable)

    lable="<ABC> "+name
    py.xlim(0,2*np.pi)
    py.scatter(xlist,yA8list,s=10,c='green',linewidth=0.5,label=lable)
    py.legend()
    filename=pathname+"/Figures/SComp/"+name+"<S>comp compare vs theta"+str(n)+" , "+str(ln)+".png"
    py.savefig(filename,dpi=200)

    
    return






def p2vecnumber_tgl_plot(name,n,ln,save,clear,color): #This plot the number of maximized vectors found for a given |psi> versus the three tangle for a two parameter state.
    if clear==1:
        py.clf()
    filename=pathname+"/Figures/"+name+"vector number vs tgl"+str(n)+" , "+str(ln)+" .png"
    length=int(float(ln)**2)
    ylist=np.zeros(length)
    xlist=np.zeros(length)
    j=0
    Ftest=0.0
    for i in range(ln):
        print 100.0*float(i)/float(ln)
        for q in range(ln):
            psi,speciallist,maxlist,xoptalist=p2call_file(name,n,i,q,ln)
            x=len(maxlist)
            
            ylist[j]=x

            d1=(psi[0,0]**2)*(psi[7,0]**2)+(psi[1,0]**2)*(psi[6,0]**2)+(psi[2,0]**2)*(psi[5,0]**2)+(psi[4,0]**2)*(psi[3,0]**2)
            d2=psi[0,0]*psi[7,0]*psi[3,0]*psi[4,0]+psi[0,0]*psi[7,0]*psi[5,0]*psi[2,0]+psi[0,0]*psi[7,0]*psi[6,0]*psi[1,0]+psi[3,0]*psi[4,0]*psi[5,0]*psi[2,0]+psi[3,0]*psi[4,0]*psi[6,0]*psi[1,0]+psi[5,0]*psi[2,0]*psi[6,0]*psi[1,0]
            d3=psi[0,0]*psi[6,0]*psi[5,0]*psi[3,0]+psi[7,0]*psi[1,0]*psi[2,0]*psi[4,0]
            Tgl=4*abs(d1-2*d2+4*d3)
            xlist[j]=Tgl
            j=j+1

    lable=name
    
    py.scatter(xlist,ylist,s=10,c=color,linewidth=0.5,label=lable)
    
    py.xlabel('Three Tangle')
    py.ylabel('Number of Maximized vectors')
    py.legend()
    if save==1:
        py.savefig(filename,dpi=200)
    return




def p1vecnumber_tgl_plot(name,n,ln,save,clear,color): #This plot the number of maximized vectors found for a given |psi> versus the three tangle for a one parameter state.
    if clear==1:
        py.clf()
    filename=pathname+"/Figures/"+name+"vector number vs tgl"+str(n)+" , "+str(ln)+" .png"
    ylist=np.zeros(ln)
    xlist=np.zeros(ln)
    Ftest=0.0
    for i in range(ln):
        psi,speciallist,maxlist,xoptalist=p1call_file(name,n,i,ln)
        x=len(maxlist)
        
        ylist[i]=x

        d1=(psi[0,0]**2)*(psi[7,0]**2)+(psi[1,0]**2)*(psi[6,0]**2)+(psi[2,0]**2)*(psi[5,0]**2)+(psi[4,0]**2)*(psi[3,0]**2)
        d2=psi[0,0]*psi[7,0]*psi[3,0]*psi[4,0]+psi[0,0]*psi[7,0]*psi[5,0]*psi[2,0]+psi[0,0]*psi[7,0]*psi[6,0]*psi[1,0]+psi[3,0]*psi[4,0]*psi[5,0]*psi[2,0]+psi[3,0]*psi[4,0]*psi[6,0]*psi[1,0]+psi[5,0]*psi[2,0]*psi[6,0]*psi[1,0]
        d3=psi[0,0]*psi[6,0]*psi[5,0]*psi[3,0]+psi[7,0]*psi[1,0]*psi[2,0]*psi[4,0]
        Tgl=4*abs(d1-2*d2+4*d3)
        xlist[i]=Tgl

    lable=name
    
    py.scatter(xlist,ylist,s=10,c=color,linewidth=0.5,label=lable)
    
    py.xlabel('Three Tangle')
    py.ylabel('Number of Maximized vectors')
    py.legend()
    if save==1:
        py.savefig(filename,dpi=200)
    return


def p1vecnumber_theta_plot(name,n,ln,save,clear,color): #This plot the number of maximized vectors found for a given |psi> versus the theta for a one parameter state.
    if clear==1:
        py.clf()
    filename=pathname+"/Figures/"+name+"vector number vs theta"+str(n)+" , "+str(ln)+" .png"
    ylist=np.zeros(ln)
    xlist=np.zeros(ln)
    Ftest=0.0
    for i in range(ln):
        psi,speciallist,maxlist,xoptalist=p1call_file(name,n,i,ln)
        x=len(maxlist)
        
        ylist[i]=x
        theta=float(i)*2*np.pi/(ln-1)
 
        xlist[i]=theta

    lable=name
    
    py.scatter(xlist,ylist,s=10,c=color,linewidth=0.5,label=lable)
    
    py.xlabel('Theta')
    py.ylabel('Number of Maximized vectors')
    py.legend()
    if save==1:
        py.savefig(filename,dpi=200)
    return




def p1Thetaccomp_theta_plot(name,n,ln): #This function, in a similar way to p1Veccomp_theta_plot, will plot the angles of the maximized unit vectors versus theta for a one parameter state.
                                        #Values are plotted as angle/pi to make it easier to read.
    maxveclist=np.zeros(ln,dtype=np.ndarray)
    xlist=[]
    Xlist=[]
    Maxlist=np.zeros(ln)
    Xmaxlist=np.zeros(ln)

    for q in range(ln):
        psi,speciallist,maxlist,xoptalist=p1call_file(name,n,q,ln)

        theta=float(q)*2*np.pi/(ln-1)
        Xmaxlist[q]=theta
        for i in range (len(maxlist)):
            Xlist.append(theta)
            x=maxlist[i]
            
            xlist.append(x)
        x=maxlist[0]
        Maxlist[i]=abs(Fa(x,psi))

    for i in range(6):
        if i==0:
            label='a'
            Alist=[]
            Blist=[]
            for j in range(len(xlist)):
                x=xlist[j]
                Alist.append(x[0]/np.pi)
                Blist.append(x[1]/np.pi)
        if i==1:
            label='ap'
            Alist=[]
            Blist=[]
            for j in range(len(xlist)):
                x=xlist[j]
                Alist.append(x[2]/np.pi)
                Blist.append(x[3]/np.pi)
        if i==2:
            label='b'
            Alist=[]
            Blist=[]
            for j in range(len(xlist)):
                x=xlist[j]
                Alist.append(x[4]/np.pi)
                Blist.append(x[5]/np.pi)
        if i==3:
            label='bp'
            Alist=[]
            Blist=[]
            for j in range(len(xlist)):
                x=xlist[j]
                Alist.append(x[6]/np.pi)
                Blist.append(x[7]/np.pi)
        if i==4:
            label='c'
            Alist=[]
            Blist=[]
            for j in range(len(xlist)):
                x=xlist[j]
                Alist.append(x[8]/np.pi)
                Blist.append(x[9]/np.pi)
        if i==5:
            label='cp'
            Alist=[]
            Blist=[]
            for j in range(len(xlist)):
                x=xlist[j]
                Alist.append(x[10]/np.pi)
                Blist.append(x[11]/np.pi)

        py.clf()
        v=[-0.1,2.0*np.pi+.1,-0.1,2.1]
        py.axis(v)
        Alabel=label+"theta"
        py.scatter(Xlist,Alist,s=10,c='r',linewidth=0.5,label=Alabel)
        py.xlabel('Parameter Theta')
        py.ylabel('Vector angle')
        py.legend()
        figname=pathname+"/Figures/AComp/"+name+"vec angles vs theta"+str(n)+" , "+str(ln)+" "+Alabel+".png"
        py.savefig(figname,dpi=200)

        py.clf()
        Blabel=label+"phi"
        py.axis(v)
        py.scatter(Xlist,Blist,s=10,c='g',linewidth=0.5,label=Blabel)
        py.xlabel('Parameter Theta')
        py.ylabel('Vector angle')
        py.legend()
        figname=pathname+"/Figures/AComp/"+name+"vec angles vs theta"+str(n)+" , "+str(ln)+" "+Blabel+".png"
        py.savefig(figname,dpi=200)
        
    py.clf()
    py.xlim(-0.1,2.0*np.pi+.1)
    f = py.figure()
    ax = f.add_subplot(111)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    Maxlabel="<S>"
    py.plot(Xmaxlist,Maxlist,c='orange',label=Maxlabel)
    py.xlabel('Parameter Theta')
    py.ylabel('Maximum violation')
    py.legend()
    figname=pathname+"/Figures/AComp/"+name+"vec angles vs theta"+str(n)+" , "+str(ln)+" "+Maxlabel+".png"
    py.savefig(figname,dpi=200)
    py.clf()
    
    return







def p1Thetaccomp_tgl_plot(name,n,ln): #This function, in a similar way to p1Veccomp_tgl_plot, will plot the angles of the maximized unit vectors versus the three tangle for a one parameter state.
                                        #Values are plotted as angle/pi to make it easier to read.
    maxveclist=np.zeros(ln,dtype=np.ndarray)
    xlist=[]
    Xlist=[]
    Maxlist=np.zeros(ln)
    Xmaxlist=np.zeros(ln)
    
    for q in range(ln):
        psi,speciallist,maxlist,xoptalist=p1call_file(name,n,q,ln)

        
        d1=(psi[0,0]**2)*(psi[7,0]**2)+(psi[1,0]**2)*(psi[6,0]**2)+(psi[2,0]**2)*(psi[5,0]**2)+(psi[4,0]**2)*(psi[3,0]**2)
        d2=psi[0,0]*psi[7,0]*psi[3,0]*psi[4,0]+psi[0,0]*psi[7,0]*psi[5,0]*psi[2,0]+psi[0,0]*psi[7,0]*psi[6,0]*psi[1,0]+psi[3,0]*psi[4,0]*psi[5,0]*psi[2,0]+psi[3,0]*psi[4,0]*psi[6,0]*psi[1,0]+psi[5,0]*psi[2,0]*psi[6,0]*psi[1,0]
        d3=psi[0,0]*psi[6,0]*psi[5,0]*psi[3,0]+psi[7,0]*psi[1,0]*psi[2,0]*psi[4,0]
        Tgl=4*abs(d1-2*d2+4*d3)

        Xmaxlist[q]=Tgl
        
        for i in range (len(maxlist)):
            Xlist.append(Tgl)
            x=maxlist[i]
            xlist.append(x)
        x=maxlist[0]
        Maxlist[q]=abs(Fa(x,psi))

        

    for i in range(6):
        if i==0:
            label='a'
            Alist=[]
            Blist=[]
            for j in range(len(xlist)):
                x=xlist[j]
                Alist.append(x[0]/np.pi)
                Blist.append(x[1]/np.pi)
        if i==1:
            label='ap'
            Alist=[]
            Blist=[]
            for j in range(len(xlist)):
                x=xlist[j]
                Alist.append(x[2]/np.pi)
                Blist.append(x[3]/np.pi)
        if i==2:
            label='b'
            Alist=[]
            Blist=[]
            for j in range(len(xlist)):
                x=xlist[j]
                Alist.append(x[4]/np.pi)
                Blist.append(x[5]/np.pi)
        if i==3:
            label='bp'
            Alist=[]
            Blist=[]
            for j in range(len(xlist)):
                x=xlist[j]
                Alist.append(x[6]/np.pi)
                Blist.append(x[7]/np.pi)
        if i==4:
            label='c'
            Alist=[]
            Blist=[]
            for j in range(len(xlist)):
                x=xlist[j]
                Alist.append(x[8]/np.pi)
                Blist.append(x[9]/np.pi)
        if i==5:
            label='cp'
            Alist=[]
            Blist=[]
            for j in range(len(xlist)):
                x=xlist[j]
                Alist.append(x[10]/np.pi)
                Blist.append(x[11]/np.pi)

        py.clf()
        v=[-0.1,1.1,-0.1,2.1]
        py.axis(v)
        Alabel=label+"theta"
        py.scatter(Xlist,Alist,s=10,c='r',linewidth=0.5,label=Alabel)
        py.xlabel('Tangle')
        py.ylabel('Vector angle')
        py.legend()
        figname=pathname+"/Figures/AComp/"+name+"vec angles vs tgl"+str(n)+" , "+str(ln)+" "+Alabel+".png"
        py.savefig(figname,dpi=200)

        py.clf()
        Blabel=label+"phi"
        py.axis(v)
        py.scatter(Xlist,Blist,s=10,c='g',linewidth=0.5,label=Blabel)
        py.xlabel('Tangle')
        py.ylabel('Vector angle')
        py.legend()
        figname=pathname+"/Figures/AComp/"+name+"vec angles vs tgl"+str(n)+" , "+str(ln)+" "+Blabel+".png"
        py.savefig(figname,dpi=200)
        
    py.clf()
    py.xlim(-0.1,1.1)
    f = py.figure()
    ax = f.add_subplot(111)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    Maxlabel="<S>"
    py.plot(Xmaxlist,Maxlist,c='orange',label=Maxlabel)
    py.xlabel('Tangle')
    py.ylabel('Maximum violation')
    py.legend()
    figname=pathname+"/Figures/AComp/"+name+"vec angles vs tgl"+str(n)+" , "+str(ln)+" "+Maxlabel+".png"
    py.savefig(figname,dpi=200)
    py.clf()
    
    return






def p2Thetaccomp_tgl_plot(name,n,ln): #This function, in a similar way to p2Veccomp_tgl_plot, will plot the angles of the maximized unit vectors versus tangle for a two parameter state.
                                        #Values are plotted as angle/pi to make it easier to read.
    xlist=[]
    Xlist=[]
    Maxlist=np.zeros(ln)
    Xmaxlist=np.zeros(ln)
    for q in range(ln):
        for j in range(ln):
            psi,speciallist,maxlist,xoptalist=p2call_file(name,n,q,j,ln)

            
            
            d1=(psi[0,0]**2)*(psi[7,0]**2)+(psi[1,0]**2)*(psi[6,0]**2)+(psi[2,0]**2)*(psi[5,0]**2)+(psi[4,0]**2)*(psi[3,0]**2)
            d2=psi[0,0]*psi[7,0]*psi[3,0]*psi[4,0]+psi[0,0]*psi[7,0]*psi[5,0]*psi[2,0]+psi[0,0]*psi[7,0]*psi[6,0]*psi[1,0]+psi[3,0]*psi[4,0]*psi[5,0]*psi[2,0]+psi[3,0]*psi[4,0]*psi[6,0]*psi[1,0]+psi[5,0]*psi[2,0]*psi[6,0]*psi[1,0]
            d3=psi[0,0]*psi[6,0]*psi[5,0]*psi[3,0]+psi[7,0]*psi[1,0]*psi[2,0]*psi[4,0]
            Tgl=4*abs(d1-2*d2+4*d3)

            Xmaxlist[q]=Tgl
            
            for i in range (len(maxlist)):
                Xlist.append(Tgl)
                x=maxlist[i]
                xlist.append(x)
            x=maxlist[0]    
            Maxlist[q]=abs(Fa(x,psi))
        

    for i in range(6):
        if i==0:
            label='a'
            Alist=[]
            Blist=[]
            for j in range(len(xlist)):
                x=xlist[j]
                Alist.append(x[0]/np.pi)
                Blist.append(x[1]/np.pi)
        if i==1:
            label='ap'
            Alist=[]
            Blist=[]
            for j in range(len(xlist)):
                x=xlist[j]
                Alist.append(x[2]/np.pi)
                Blist.append(x[3]/np.pi)
        if i==2:
            label='b'
            Alist=[]
            Blist=[]
            for j in range(len(xlist)):
                x=xlist[j]
                Alist.append(x[4]/np.pi)
                Blist.append(x[5]/np.pi)
        if i==3:
            label='bp'
            Alist=[]
            Blist=[]
            for j in range(len(xlist)):
                x=xlist[j]
                Alist.append(x[6]/np.pi)
                Blist.append(x[7]/np.pi)
        if i==4:
            label='c'
            Alist=[]
            Blist=[]
            for j in range(len(xlist)):
                x=xlist[j]
                Alist.append(x[8]/np.pi)
                Blist.append(x[9]/np.pi)
        if i==5:
            label='cp'
            Alist=[]
            Blist=[]
            for j in range(len(xlist)):
                x=xlist[j]
                Alist.append(x[10]/np.pi)
                Blist.append(x[11]/np.pi)

        py.clf()
        v=[-0.1,1.1,0.0,2.1]
        py.axis(v)
        Alabel=label+"theta"
        py.scatter(Xlist,Alist,s=10,c='r',linewidth=0.5,label=Alabel)
        py.xlabel('Tangle')
        py.ylabel('Vector angle')
        py.legend()
        figname=pathname+"/Figures/AComp/"+name+"vec angles vs tgl"+str(n)+" , "+str(ln)+" "+Alabel+".png"
        py.savefig(figname,dpi=200)

        py.clf()
        Blabel=label+"phi"
        py.axis(v)
        py.scatter(Xlist,Blist,s=10,c='g',linewidth=0.5,label=Blabel)
        py.xlabel('Tangle')
        py.ylabel('Vector angle')
        py.legend()
        figname=pathname+"/Figures/AComp/"+name+"vec angles  vs tgl"+str(n)+" , "+str(ln)+" "+Blabel+".png"
        py.savefig(figname,dpi=200)
        
    py.clf()
    py.xlim(-0.1,1.1)
    f = py.figure()
    ax = f.add_subplot(111)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    Maxlabel="<S>"
    py.plot(Xmaxlist,Maxlist,c='orange',label=Maxlabel)
    py.xlabel('Tangle')
    py.ylabel('Maximum violation')
    py.legend()
    figname=pathname+"/Figures/AComp/"+name+"vec angles vs tgl"+str(n)+" , "+str(ln)+" "+Maxlabel+".png"
    py.savefig(figname,dpi=200)
    py.clf()
    
    return


