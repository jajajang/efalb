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

def create_action(gap):
    elem=np.arange(-1,1,gap)
    N=len(elem)
    X=[0]*(N*N)
    i=0
    j=0
    while i<N:
        while j<N:
            X[i*N+j]=np.array([elem[i],elem[j]])
            j=j+1
        i=i+1
        j=0
    return np.array(X)
def create_action_circle(gap):
    elem=np.arange(-1,1,gap)
    N=len(elem)
    X=[0]*(N*N)
    i=0
    j=0
    s=0
    while i<N:
        while j<N:
            if la.norm(np.array([elem[i],elem[j]]))<=1:
                X[s]=np.array([elem[i],elem[j]])
                s=s+1
            j=j+1
        i=i+1
        j=0
    return np.array(X[:s])

#Nlist=[10,20,40,80]
#for itera in range(0,4):
timespent=np.zeros(10)
experi = [0] * 10
gapp=1/25
X = create_action_circle(gapp)
Z = create_action_circle(gapp)
N1=X.shape[0]
N2=Z.shape[0]
NN = N1*N2
dmax = d1 * d2
W = np.zeros((NN, dmax))
multiplier=0.01
for i in range(W.shape[0]):
    i1, i2 = np.unravel_index(i, (X.shape[0], Z.shape[0]))
    W[i, :] = np.outer(X[i1, :], Z[i2, :]).ravel()

for exper in range(0,10):
    start=timer()
    cum_regret=np.zeros(time)
    beta=np.sqrt(d)
    V=np.identity(d1*d2)/(d1*d2)
    invV=np.identity(d1*d2)*(d1*d2)
    theta_hat=np.zeros((d1,d2)).ravel()

    uinit,vinit=create_unitvec(d1,d2)
    L=np.outer(uinit,vinit)
    SIGMA=np.outer(uinit,vinit)
    opti=1
    instant_regret=0
    cum_reg=0
    WTy=np.zeros(d*d)
    valid_idx=np.arange(NN)
    s=1
    radius_sq=1
    W_invVt_norm_sq=1
    for t in range(0,time):
        x_t=np.zeros(d)
        z_t=np.zeros(d)
        if (t == 0):
            x_t=X[ra.randint(N1),:]
            z_t=Z[ra.randint(N2),:]
            radius_sq = multiplier * beta
        else:
            A = (X @ theta_hat.reshape(d1, d2) @ Z.T).ravel()
            ucblist=[0]*NN
            for ss in range(0,NN):
                ucblist[ss]=np.dot(np.dot(W[ss],invV),W[ss])
            B = beta* np.sqrt(ucblist)
            obj_func = A + B
            truth = True
            while truth:
                if all(B[valid_idx] < (1 / np.sqrt(time))):
                    truth = False
                    chosen_inner = np.argmax(obj_func[valid_idx])
                    chosen = valid_idx[chosen_inner]
                elif all(B[valid_idx] < 2 ** (-s)):
                    borderline = max(obj_func) - 2 ** (1 - s)
                    valid_idx = np.arange(NN)[obj_func > borderline]
                    s = s + 1
                    # self.Vt = self.lam * np.eye(self.d)
                    # self.invVt = np.eye(self.d) / self.lam
                    # self.W_invVt_norm_sq = np.sum(self.W * self.W, axis=1) / self.lam
                    # self.logdetV = self.d*log(self.lam)
                    # self.theta_hat = np.zeros(self.d)
                    # self.WTy = np.zeros(self.d)
                    # chosen_inner = ra.randint(len(valid_idx))
                    # chosen = valid_idx[chosen_inner]
                else:
                    truth = False
                    chosen_inner = np.argmax(B[valid_idx])
                    chosen = valid_idx[chosen_inner]
            chosenPair = np.unravel_index(chosen, (N1, N2))
            x_t=X[chosenPair[0],:]
            z_t=Z[chosenPair[1],:]

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
            middle=timer()
            print('time: '+str(middle-start))
            print('Iteration: '+str(t))
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
np.savetxt('eFALB_gap_'+str()+'.csv', experi, delimiter=',')

#plt.legend(["eps=8*eps_0", "eps=4*eps_0", "eps=2eps_0", "eps=eps_0"])
#plt.show()

