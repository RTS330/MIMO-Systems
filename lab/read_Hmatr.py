# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 20:12:36 2023

@author: alex
"""

import scipy.io as sp
import numpy as np
from numpy.linalg import inv
from numpy import linalg as la
Hmatr = sp.loadmat("H16_4_35.mat")
hl=list(Hmatr.values())
h=hl[3]
h1=h[:,0,0]
h2=h[:,0,1]
h3=h[:,0,2]
h4=h[:,0,3]
h1=h1/la.norm(h1)
h2=h2/la.norm(h2)
h3=h3/la.norm(h3)
h4=h4/la.norm(h4)
H=np.zeros([16,4],dtype = 'complex_')
W=np.zeros([16,4],dtype = 'complex_')
W0=np.zeros([16,4],dtype = 'complex_')
H[:,0]=h1
H[:,1]=h2
H[:,2]=h3
H[:,3]=h4
H=H.T
H1=H.conj().T
H2=H@H1
W=H1@inv(H2)
w1=W[:,0]
w1=w1/la.norm(w1)
w2=W[:,1]
w2=w2/la.norm(w2)
w3=W[:,2]
w3=w3/la.norm(w3)
w4=W[:,3]
w4=w4/la.norm(w4)
 

 

r=h2.T@w1
print(r)
W0[:,0]=w1;
W0[:,1]=w2;
W0[:,2]=w3;
W0[:,3]=w4;
Ik=H@W0
print(Ik)

