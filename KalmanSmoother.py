# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 10:49:01 2017

@author: ganesh
"""
import numpy as np
import sys

def KalmanRTSSmoother(MMf,MMp,PPf,PPp,A):
#% Description:
#%   Perform Kalman Filter prediction step. The model is
#%
#%     x[k] = A*x[k-1] + Bw[k-1],  w ~ N(0,Q).
#%     y[k] = C*x[k]   + v[k]      v ~ N(0,R)
#
#%Example
#% Y is measurement
#% x=x0
#% P=P0;
#% [MMf,MMp,PPf,PPp]=KalmanLoop(x,P,A,B,Y,Q,R,D,u)
#% [MMs,PPs]=KalmanRTSSmoother(MMf,MMp,PPf,PPp,A)

    N = MMf.shape[1]
    N1 = MMf.shape[0]
    MMs = np.zeros((N1,N), dtype=float)
    MMs[:,-1] = MMf[:,-1]
    PPs = [None]*N    #cell(1,N);
    PPs[N-1]=PPf[-1]
    Xs=MMf[:,-1]
    Ps=PPf[-1]
    for i in range(N-2, 0, -1):    # =N-1:-1:1
        Xf = MMf[:,i]    #% forwardan gelen filtered deger
        Pf = PPf[i]      #% forwardan gelen filtered deger
        
        Xp = MMp[:,i+1]  #% forwarla bir sonraki zamandan predicted deger
        Pp = PPp[i+1]    #% forwarla bir sonraki zamandan predicted deger

        J = np.dot(np.dot(Pf,A.T),np.linalg.inv(Pp))
        Xs = Xf + np.dot(J, (Xs-Xp))
        Ps = Pf + np.dot(np.dot(J, (Ps-Pp)), J.T)
        MMs[:,i] = Xs
        PPs[i] = Ps
    return MMs,PPs
    
def KalmanUpdate(X,P,m,C,R):
#% Kalman Filter update step
#%
#
#%   [X,P,S] = KF_UPDATE(X,P,m,C,R)
#%
#% Description:
#%   Perform Kalman Filter prediction step. The model is
#%
#%     x[k] = A*x[k-1]  + Bw[k-1],  w ~ N(0,Q).
#%     y[k] = C*x[k]   + v[k],      v ~ N(0,R)

# update step
    z = np.dot(C, X)   # Mesurament prediction
    v = m - z          # Innovation
    S = np.dot(np.dot(C,P),C.T) + R     # C * P * C' + R;    % innovation Covar
    K = np.dot(np.dot(P,C.T), np.linalg.inv(S))   #P * C' * inv(S);   % Kalman Gain
    X = X + np.dot(K,v)
    P = P - np.dot(np.dot(K, S), K.T)
    return X,P



def KalmanPredict(x,P,A,B,Q):
#%Kalman Filter  Prediction  step
#
#%   [X,P] = KF_PREDICT(X,P,A,B,D,Q,U)
# 
#% Description:
#%   Perform Kalman Filter prediction step. The model is
#%
#%     x[k] = A*x[k-1]  + Bw[k-1],  w ~ N(0,Q).
#%     y[k] = C*x[k]   + v[k]                 v ~ N(0,R)
    x = np.dot(A,x)
    P = np.dot(np.dot(A, P), A.T) + np.dot(np.dot(B, Q), B.T)  # A * P * A' + B * Q * B'
    return x,P 
    
def KalmanLoop(x,P,A,B,C,Y,Q,R):
    N = Y.shape[1]    #Number of Measurements
    MMf = np.zeros((2,N))
    PPf = [] #cell(1,N);
    MMp = np.zeros((2,N))
    PPp = [] #cell(1,N)
    # Perform prediction
    for i in range(N):
        (x,P) = KalmanPredict(x,P,A,B,Q)
        MMp[:,i, None] = x
        PPp.append(P)
        (x,P) = KalmanUpdate(x,P,Y[0,i],C,R)
        MMf[:,i, None] = x
        PPf.append(P)
    return MMf,MMp,PPf,PPp

def kalmansmooth(*arg):
    if len(arg) < 1:
        raise ValueError('kalmansmooth needs at least one argument: Usage: kalmansmooth(signal_to_smooth, smooth_factor)')
    else:
        tst_trg = arg[0]
    if len(arg) > 1:
        R = arg[1]
    else:
        R = 0.25; #Change R value, The Bigger R, The more smoothed data        
    # function that performs Kalman smoothing on input data
    data_sm = [];
    for iter1 in range(tst_trg.shape[0]):
        EstdData = tst_trg[None,iter1,:]
        Q = 300
        T = 0.1
        A = np.asarray([[1, T],[0, 1]])
        B = np.asarray([[(T**2)/2],[T]])
        C = np.asarray([[1, 0]])
        P0= np.eye(2)
        x0= np.asarray([[EstdData[0,0]],[0]])
        x = x0
        P = P0
        (MMf,MMp,PPf,PPp) = KalmanLoop(x,P,A,B,C,EstdData,Q,R)
        MMs,PPs = KalmanRTSSmoother(MMf,MMp,PPf,PPp,A)
        data_sm.append(MMs[0,:, None])
    tst_trg_sm = np.concatenate(data_sm, axis=1)
    return tst_trg_sm.T

if __name__ == "__main__":
    kalmansmooth(sys.argv[1], sys.argv[2])
    