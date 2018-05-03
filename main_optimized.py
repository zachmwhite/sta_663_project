import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import helpers_optimized as hp
import numba
from numba import jit

@jit
def rmhmc(X,y,niter = 6000,burnin = 1000,nleapfrog = 6,nnewton = 1,alpha = 100):
    """
    
    """
    step_size = 3 / nleapfrog
    n,D = X.shape

    theta = np.ones((D,1))*1e-3
    thetasamples = np.zeros((niter - burnin,D))

    Xtheta = X.dot(theta)
    Cur_LL = hp.lognorm(np.zeros(D),theta.reshape(D,),alpha) + Xtheta.T.dot(y) - np.sum(np.log(1 + np.exp(Xtheta)))

    for it in range(niter):
        print("Iteration Num: ", it)
        new_theta = theta

        # Calculating G and partial derivatives
        Xtheta,logitXtheta,prodlXtheta,G,I_G = hp.newG(X,new_theta,alpha)      
        I_Deriv,TraceI_Deriv = hp.partial_derivs(X,logitXtheta,prodlXtheta,I_G)        
        Pro_momentum = np.random.randn(1,D).dot(la.cholesky(G)).T

        O_momentum = Pro_momentum
        O_G = G
        O_I_G = I_G  

        nsteps = np.ceil(np.random.rand(1)*nleapfrog)    
        # Leapfrog Steps
        for j in np.arange(nsteps):

            # Update momentum (first step)
            Pro_momentum = hp.update_momentum(X,y,new_theta,Pro_momentum,TraceI_Deriv,I_Deriv,I_G,logitXtheta,alpha,step_size)

            # Update theta parameters (second step)
            new_theta = hp.update_parameter(X,new_theta,Pro_momentum,I_G,nnewton,alpha,step_size)

            # Calculate new G based on the new parameters
            Xtheta,logitXtheta,prodlXtheta,G,I_G = hp.newG(X,new_theta,alpha)

            # Calculate Partial Derivatives for DG/dtheta
            I_Deriv,TraceI_Deriv = hp.partial_derivs(X,logitXtheta,prodlXtheta,I_G)

            # Update momentum again (third step)
            Pro_momentum = hp.update_momentum(X,y,new_theta,Pro_momentum,TraceI_Deriv,I_Deriv,I_G,logitXtheta,alpha,step_size)

            if(j == (nsteps - 1)):
                print("new_theta: ",new_theta.T)

        # Proposed value
        Pro_LL = hp.lognorm(np.zeros(D),new_theta.reshape(D,),alpha) + Xtheta.T.dot(y) - np.sum(np.log(1 + np.exp(Xtheta)))     
        Pro_D = .5*(np.log(2) + D * np.log(np.pi) + np.log(la.det(G)))
        Pro_H = -Pro_LL + Pro_D + (Pro_momentum.T.dot(I_G.dot(Pro_momentum)))/2

        # Current value
        Cur_D = .5* (np.log(2) + D * np.log(np.pi) + np.log(la.det(O_G)))    
        Cur_H = - Cur_LL + Cur_D + (O_momentum.T.dot(O_I_G.dot(O_momentum)))/2

        r = -Pro_H + Cur_H

        u = np.log(np.random.rand(1))
        print("r: ",r," u: ",u)
        if (r > 0 or r > u):        
            Cur_LL = Pro_LL
            theta = new_theta

        if it >= burnin:
            thetasamples[it-burnin,:] = theta.reshape(D,)

    return thetasamples