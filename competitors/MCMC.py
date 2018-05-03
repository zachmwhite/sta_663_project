import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import HMC_helper as hp

def rmhmc(X,y,niter = 6000,burnin = 1000,nleapfrog = 6,nnewton = 1,alpha = 100):
    """
    
    """
    step_size = 3 / nleapfrog
    n,D = X.shape

    G = np.eye(D)*50

    theta = np.ones((D,1))*1e-3
    thetasamples = np.zeros((niter - burnin,D))

    Xtheta = X.dot(theta)
    Cur_LL = hp.lognorm(np.zeros(D),theta.reshape(D,),alpha) + Xtheta.T.dot(y) - np.sum(np.log(1 + np.exp(Xtheta)))

    for it in range(niter):
        print("Iteration Num: ", it)
        new_theta = np.random.normal(theta,step_size)
        print("new_theta: ",new_theta.T)

        Xtheta = X.dot(new_theta)
        # Proposed value
        Pro_LL = hp.lognorm(np.zeros(D),new_theta.reshape(D,),alpha) + Xtheta.T.dot(y) - np.sum(np.log(1 + np.exp(Xtheta)))     

        r = -Cur_LL + Pro_LL

        u = np.log(np.random.rand(1))
        print("r: ",r," u: ",u)
        if (r > 0 or r > u):        
            Cur_LL = Pro_LL
            theta = new_theta

        if it >= burnin:
            thetasamples[it-burnin,:] = theta.reshape(D,)

    return thetasamples