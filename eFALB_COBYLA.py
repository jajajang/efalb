import numpy as np
import numpy.random as ra
import numpy.linalg as la
from scipy.optimize import fsolve
from matplotlib import pyplot as plt
from scipy.stats import ortho_group
from scipy.optimize import minimize
from timeit import default_timer as timer
d=2
d1=d
d2=d
sigma=0.01
time = 2500

def con(t):
    return 1-la.norm(t)
cons = {'type':'ineq', 'fun': con}

def create_unitvec(d_1, d_2):
    v1 = np.random.rand(d_1)
    v1_hat = v1 / np.linalg.norm(v1)
    v2 = np.random.rand(d_2)
    v2_hat = v2 / np.linalg.norm(v2)
    return v1_hat,v2_hat

def argmax_finder(V,theta_hat,d1,d2,beta):
    x_old=np.zeros(d1)
    z_old=np.zeros(d2)
    x,z=create_unitvec(d1,d2)
    u, s, vh = np.linalg.svd(np.reshape(theta_hat,(d1,d2)), full_matrices=False)
    xinit=u[:,0]
    zinit=vh[0,:]
    eps=0.001
    count=0
    while (la.norm(x_old-x)>eps or la.norm(z_old-z)>eps) and count<120:
        #fix z first
        def newfx(x):
            newW = np.outer(x, z).ravel()
            return -np.dot(newW, theta_hat) - beta * np.sqrt(newW @ V @ newW.T)
        result=minimize(newfx,0.5*xinit,method='COBYLA',constraints=cons)
        x_old=x
        x=result.x
        def newfz(z):
            newW = np.outer(x, z).ravel()
            return -np.dot(newW, theta_hat) - beta * np.sqrt(newW @ V @ newW.T)
        result=minimize(newfz,0.5*zinit,method='COBYLA',constraints=cons)
        z_old=z
        z=result.x
        count=count+1
    return x,z

timespent=np.zeros(10)
#Nlist=[10,20,40,80]
#for itera in range(0,4):
experi = [0] * 10
for exper in range(0,10):
    start=timer()
    cum_regret=np.zeros(time)
    beta=np.sqrt(d)
    V=np.identity(d1*d2)/(d1*d2)
    invV=np.identity(d1*d2)*(d1*d2)
    theta_hat=np.zeros((d1,d2)).ravel()

    uinit,vinit=create_unitvec(d1,d2)
    L=np.outer(uinit,vinit)
    opti=1
    instant_regret=0
    cum_reg=0
    WTy=np.zeros(d*d)
    for t in range(0,time):
        x_t=np.zeros(d)
        z_t=np.zeros(d)
        if t==0:
            x_t, z_t=create_unitvec(d1,d2) #random vector change
        else:
            x_t, z_t=argmax_finder(invV,theta_hat,d1,d2,beta)

        reward=x_t@L@z_t+ra.normal(0,sigma)
        instant_regret=opti-x_t@L@z_t
        cum_reg+=instant_regret
        cum_regret[t] = cum_reg

        w_t=(np.outer(x_t,z_t)).ravel()
        V=V+np.outer(w_t,w_t)
        invV=la.inv(V)

        WTy+=reward*w_t
        theta_hat = np.dot(invV, WTy)
        beta=sigma*np.sqrt(np.log(la.det(V)/4*10000))+1
        if t%20==0:
            print(t)
            print(cum_reg)
            print(uinit)
            print(x_t)
            print(vinit)
            print(z_t)
    experi[exper]=cum_regret
    timeline=np.arange(time)
    end=timer()
    print(end-start)
    timespent[exper]=end-start
    #plt.plot(np.arange(time), cum_regret)
np.savetxt('eFALB_COBYLA.csv', experi, delimiter=',')

#plt.legend(["eps=8*eps_0", "eps=4*eps_0", "eps=2eps_0", "eps=eps_0"])
#plt.show()

