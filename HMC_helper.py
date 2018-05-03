import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

def lognorm(X,theta,alpha):
    return np.sum(-(.5*np.log(2*np.pi*alpha)) - ((X - theta)**2) / (2*alpha))

def newG(X,theta,alpha,G):
    D = X.shape[1]
    Xtheta = X.dot(theta)
    logitXtheta = 1 / (1 + np.exp(-Xtheta))
    prodlXtheta = logitXtheta * (1 - logitXtheta)
    I_G = la.inv(G)
    
    return Xtheta,logitXtheta,prodlXtheta,I_G

def partial_derivs(X,logitXtheta,prodlXtheta,I_G):
    n,D = X.shape
    Z1 = np.zeros((n,D))
    I_Deriv = {}
    TraceI_Deriv = np.zeros(D)
    
    for d in range(D):
        Z = ((1 - 2*logitXtheta)*(X[:,d]).reshape(-1,1))
        
        for a in range(D):
            Z1[:,a] = (X[:,a].reshape(-1,1)*prodlXtheta*Z).reshape(n,)
        I_Deriv[d] = I_G.dot(Z1.T.dot(X))
        TraceI_Deriv[d] = np.trace(I_Deriv[d])
    return I_Deriv, TraceI_Deriv

def update_momentum(X,y,theta,Pro_momentum,TraceI_Deriv,I_Deriv,I_G,logitXtheta,alpha,step_size):
    n,D = X.shape
    MI_DerivI_GM = np.zeros(D)
    I_Gmomentum = I_G.dot(Pro_momentum)
    for d in range(D):
        MI_DerivI_GM[d] = .5*((Pro_momentum.T.dot(I_Deriv[d].dot(I_Gmomentum))))
    dHdTheta = X.T.dot(y - logitXtheta) - (np.eye(D)*(1/alpha)).dot(theta)
    Pro_momentum = Pro_momentum +(step_size/2)*dHdTheta
    
    return Pro_momentum

def update_parameter(X,theta,Pro_momentum,I_G,G,nnewton,alpha,step_size):
    new_theta = theta
    OI_Gmomentum = I_G.dot(Pro_momentum)
    for FixedIter in range(nnewton):
        Xtheta,logitXtheta,prodlXtheta,I_G = newG(X,new_theta,alpha,G)
        I_Gmomentum = I_G.dot(Pro_momentum)
        new_theta = theta +  (step_size/2) *(I_Gmomentum)        
    return new_theta
