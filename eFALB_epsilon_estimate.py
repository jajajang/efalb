import numpy as np
import numpy.random as ra
import numpy.linalg as la
from scipy.optimize import fsolve
from matplotlib import pyplot as plt
from scipy.stats import ortho_group
from types import SimpleNamespace
from tqdm import tqdm
import sys

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




d=2
SIGMA=np.diag([1,0.3])
sigma=0.01
time = 2500

opts = SimpleNamespace()
opts.nTry = int(60)
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

# # For debugging
# print("DEBUG"); print("DEBUG"); print("DEBUG")
# opts.nTry = 2


resList = []
res = SimpleNamespace()
res.arms = []
res.times = []
res.expected_rewards = []
res.dbgDicts = []

Nlist=[10,20,40,80]
total_history=np.zeros([len(Nlist),5,opts.nTry,time])
for itera in range(0,len(Nlist)):
    paramList=paramGetList1('EFALB')
    for paramIdx, algoParam in enumerate(paramList):
        history=[0]*opts.nTry
        for exper in range(0,opts.nTry):
            cum_regret=np.zeros(time)
            N=Nlist[itera]
            opts.dataopts.N1 = N
            opts.dataopts.N2 = N
            eps = 1/N
            print(N)
            #creating datapoints
            X=np.zeros((N,d))
            Z=np.zeros((N,d))
            step=ra.uniform(0,1)
            for i in range(0,N):
                X[i,:]=[np.cos(2*np.pi/N*(i+step)), np.sin(2*np.pi*(i+step)/N)]
                Z[i,:] = [np.cos(2*np.pi * (i+step) / N), np.sin(2*np.pi * (i+step) / N)]
            gap=np.min([1-X[0,:]@SIGMA@Z[0,:].T,1-X[N-1,:]@SIGMA@Z[N-1,:].T])
            print(gap)
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
            total_history[itera,paramIdx,exper]=cum_regret
        total_history=np.array(total_history)
        np.savetxt('testy_ortho_central_code' + str(Nlist[itera]) + 'paramIdx' + str(paramIdx) + '.csv', history, delimiter=',')

bestidx=[0]*len(Nlist)
tAry=np.arange(time)
for i in range(0,len(Nlist)):
    best=np.inf
    for j in range(0,5):
        evaluate_by_mean=np.mean(total_history[i,j,:,-1])
        if evaluate_by_mean<best:
            best=evaluate_by_mean
            bestidx[i]=j
    crbest=total_history[i,bestidx[i],:,:]
    me, err = getErrorBarMat(crbest.T)
    plt.plot(tAry, me, alpha=0.7, linewidth=2)
    plt.fill_between(tAry, me-err, me+err, alpha=0.20)

plt.rcParams.update({'font.size': 12})
plt.xlabel('Time', fontsize=12)
plt.ylabel('Regret', fontsize=12)
plt.legend([r"$\epsilon=8\epsilon_0$", r"$\epsilon=4\epsilon_0$", r"$\epsilon=2\epsilon_0$", r"$\epsilon=\epsilon_0$"])
plt.show()
#plt.show()
