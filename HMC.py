import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import HMC_helper as hp

def hmc(X,y,niter = 6000,burnin = 1000,nleapfrog = 6,nnewton = 1,alpha = 100,initv=20):
    """
    
    """
    step_size = 3 / nleapfrog
    n,D = X.shape

    G = np.eye(D)*initv

    theta = np.ones((D,1))*1e-3
    thetasamples = np.zeros(D)

    Xtheta = X.dot(theta)
    Cur_LL = hp.lognorm(np.zeros(D),theta.reshape(D,),alpha) + Xtheta.T.dot(y) - np.sum(np.log(1 + np.exp(Xtheta)))

        print("Iteration Num: ", it)
        new_theta = theta

        # Calculating G and partial derivatives
        Xtheta,logitXtheta,prodlXtheta,I_G = hp.newG(X,new_theta,alpha,G)      
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
            new_theta = hp.update_parameter(X,new_theta,Pro_momentum,I_G,G,nnewton,alpha,step_size)

            # Calculate new G based on the new parameters
            Xtheta,logitXtheta,prodlXtheta,I_G = hp.newG(X,new_theta,alpha,G) 
            
            # Calculate Partial Derivatives for DG/dtheta
            I_Deriv,TraceI_Deriv = hp.partial_derivs(X,logitXtheta,prodlXtheta,I_G)

            # Update momentum again (third step)
            Pro_momentum = hp.update_momentum(X,y,new_theta,Pro_momentum,TraceI_Deriv,I_Deriv,I_G,logitXtheta,alpha,step_size)

            if(j == (nsteps - 1)):
                print("new_theta: ",new_theta.T)

        # Proposed value
        Pro_LL = hp.lognorm(np.zeros(D),new_theta.reshape(D,),alpha) + X.dot(new_theta).T.dot(y) - np.sum(np.log(1 + np.exp(X.dot(new_theta))))     
        Pro_H = -Pro_LL

        # Current value
        Cur_H = - Cur_LL

        print("p: ",Pro_H," c: ",Cur_H)

        
        r = -Pro_H + Cur_H

        u = np.log(np.random.rand(1))
        print("r: ",r," u: ",u)
        if (r > 0 or r > u):        
            Cur_LL = Pro_LL
            theta = new_theta

        if it >= burnin:
            
            thetasamples = np.r_[thetasamples,theta.reshape(D,)]

            
    return thetasamples