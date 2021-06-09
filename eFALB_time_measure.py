import numpy as np
import numpy.random as ra
import numpy.linalg as la
from scipy.optimize import fsolve
from matplotlib import pyplot as plt
from scipy.stats import ortho_group
from scipy.optimize import minimize
from timeit import default_timer as timer

from expr01_defs import *
from myutils3_v2 import *
from blbandits_with_efalb_210425 import *
import bleval

class Circle_for_efalb(DataForBilinearBandit):
    def __init__(self, R, r, X, Z, Theta):
        self.R = R
        self.r = r
        self.set_X_Z(X,Z)
        self.set_theta_star(Theta)
        self.S_F=1

    def set_X_Z(self, X, Z):
        self.X = X
        self.Z = Z
        [self.N1, self.d1] = X.shape
        [self.N2, self.d2] = Z.shape
        self.N = self.N1*self.N2
        self.d = self.d1*self.d2

    def set_theta_star(self, Theta):
        self.Th=Theta
        self.expt_reward = (self.X @ self.Th) @ self.Z.T
        self.best_arm_pair = tuple(np.unravel_index(np.argmax(self.expt_reward),
                                              self.expt_reward.shape))
    def get_reward(self, idx_pair):
        x = self.X[idx_pair[0],:]
        z = self.Z[idx_pair[1],:]
        return x @ self.Th @ z + self.R * ra.normal(0,1)

    def get_best_reward(self):
        return self.expt_reward[self.best_arm_pair]

    def get_expected_reward(self, idx_pair):
        """ can also take idx_pair as a list of index pairs (list of tuples)
        """
        return [data.expt_reward[row[0],row[1]] for row in idx_pair]

    def get_expected_regret(self, idx_pair):
        """ can also take idx_pair as a list of index pairs (list of tuples)
        """
        x = self.best_arm_pair[0]
        z = self.best_arm_pair[1]
        return self.expt_reward[x,z] - self.expt_reward[idx_pair[0], idx_pair[1]]
        if type(idx_pair) is list:
            return self.expt_reward[x,z] - self.get_expected_reward(self, idx_pair)

    def __str__(self):
        return str(self.__dict__)

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






d=2
d1=d
d2=d
SIGMA=np.diag([1,0.3])
sigma=0.01
time = 2500

opts = SimpleNamespace()
opts.nTry = int(10)
opts.gSeed = 119
opts.dataSeed = 99
opts.R = 0.01
#--- options for toy
opts.dataopts = SimpleNamespace()

opts.dataopts.d1 = d
opts.dataopts.d2 = d
opts.dataopts.r = 1
opts.dataopts.S_2norm = 0.4
opts.dataopts.R = opts.R

opts.T = 2500
opts.lam = 1.0

resList = []
res = SimpleNamespace()
res.arms = []
res.times = []
res.expected_rewards = []
res.dbgDicts = []

gapp=1/50
X = create_action_circle(gapp)
Z = create_action_circle(gapp)

total_history=np.zeros([5,opts.nTry,time])
paramList=paramGetList1('EFALB')
for paramIdx, algoParam in enumerate(paramList):
    history=[0]*opts.nTry
    for exper in range(0,opts.nTry):
        cum_regret=np.zeros(time)
        opts.dataopts.N1 = X.shape[0]
        opts.dataopts.N2 = Z.shape[0]
        uinit,vinit=create_unitvec(d1,d2)
        SIGMA=np.outer(uinit,vinit)
        #creating datapoints
        data=Circle_for_efalb(opts.dataopts.R, opts.dataopts.r, X, Z, SIGMA)
        print('\n#- paramIdx = %5d' % paramIdx)
        printExpr('algoParam')
        algo = banditFactory(data, 'EFALB', algoParam, opts)
        [rewardAry, armPairAry, dbgDict] = run_bilinear_bandit(algo, data, opts.T)
        ExpectedReward = data.get_expected_reward(armPairAry)
            #[data.get_expected_reward([row[0], row[1]]) for row in armPairAry]
        cum_regret=np.zeros(opts.T)
        cum_regret[0]=1-ExpectedReward[0]
        for i in range(1,opts.T):
            cum_regret[i]=cum_regret[i-1]+1-ExpectedReward[i]
        total_history[paramIdx,exper]=cum_regret
        print(cum_regret[-1])
    total_history=np.array(total_history)
