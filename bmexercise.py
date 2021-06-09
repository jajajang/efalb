from matrixrecovery import rankone
import numpy as np
import numpy.random as ra
import numpy.linalg as la
import csv
d=8
d1=d
d2=d
c=1/d
N=16
N1=N
N2=N
r=1
R=0.01
delta=0.01
T=5000
nTry=30

beta=2*np.sqrt(d*r*np.log(T/delta))+1
multi=0.015




for mulmul in range(0,4):
    for loo in range(0,nTry):
        X = ra.normal(0, 1, (N, d1))
        norms = la.norm(X, axis=1)
        X /= norms.reshape(-1, 1)


        Z = ra.normal(0, 1, (N, d2))
        norms = la.norm(Z, axis=1)
        Z /= norms.reshape(-1, 1)

        u=ra.normal(0,1,d1)
        u=u/la.norm(u)
        v=ra.normal(0,1,d2)
        v=v/la.norm(v)
        Theta=np.outer(u,v)

        maxi = np.max((X @ Theta @ np.transpose(Z)).ravel())
        W = np.zeros((N1 * N2, d1 * d2))
        for i in range(W.shape[0]):
            i1, i2 = np.unravel_index(i, (N1, N2))
            W[i, :] = np.outer(X[i1, :], Z[i2, :]).ravel()

        Xhis=[]
        Zhis=[]
        cum_regret=[]
        regret=0
        y=[]
        theta_hat=np.zeros((d1,d2))
        V=c*np.identity(d*d)
        for t in range(0,T):
            if(t%100==0): print(t)
            invVt_norm = np.diag(np.matmul(np.matmul(W, la.inv(V)), np.transpose(W)))
            hat_mu = (X @ theta_hat @ np.transpose(Z)).ravel()
            vari = multi*beta*np.sqrt(invVt_norm)
            UCB = hat_mu+vari
            chosen_inner=np.argmax(UCB)
            chosen_pair = np.unravel_index(chosen_inner, (N1,N2))
            if t==0:
                Xhis=[X[chosen_pair[0],:]]
                Zhis=[Z[chosen_pair[1],:]]
            else:
                Xhis= np.concatenate((Xhis, [X[chosen_pair[0],:]]),axis=0)
                Zhis = np.concatenate((Zhis, [Z[chosen_pair[1], :]]), axis=0)
            result=np.dot(W[chosen_inner],Theta.ravel())+ra.normal(0,R)
            y.append(result)
            UU,VV, out_nIter, stat = rankone(np.array(Xhis),np.array(Zhis),y,r,R)
            theta_hat=np.outer(UU,VV)
            V+= np.outer(W[chosen_inner], W[chosen_inner])
            regret+=maxi-np.dot(W[chosen_inner],Theta.ravel())
            cum_regret.append(regret)
        if loo==0:
            data=[cum_regret]
        else:
            data=np.concatenate((data, [cum_regret]), axis=0)
        if loo%2==1:
            namae="bmoracle_multi"+str(multi)+"delta"+str(delta)+"half.csv"
            np.savetxt(namae, data, delimiter=",")
    multi-=0.001
'''
yy=[]
theta_hat_oful=np.zeros((d1,d2))
sum_resulvec=np.zeros(d1*d2)
V=c*np.identity(d*d)
beta=np.sqrt(d*d*np.log(T/delta))
cum_regret_2=[]
regret=0

for t in range(0,T):
    if(t%100==0): print(t)
    beta = np.sqrt(d * d * np.log((t+1) / delta))
    invVt_norm = np.diag(np.matmul(np.matmul(W, la.inv(V)), np.transpose(W)))
    hat_mu = (X @ theta_hat_oful @ np.transpose(Z)).ravel()
    vari = multi*beta*np.sqrt(invVt_norm)
    UCB = hat_mu+vari
    chosen_inner=np.argmax(UCB)
    chosen_pair = np.unravel_index(chosen_inner, (N1,N2))
    result=np.dot(W[chosen_inner],Theta.ravel())+ra.normal(0,R)
    sum_resulvec+=W[chosen_inner]*result
    theta_hat_oful=(np.matmul(la.inv(V),sum_resulvec)).reshape(d1,d2)
    V+= np.outer(W[chosen_inner], W[chosen_inner])
    regret+=maxi-np.dot(W[chosen_inner],Theta.ravel())
    cum_regret_2.append(regret)
'''

'''
X_test_his=[]
Z_test_his=[]
y_his=[]
theta_test=0
for t in range(0,T):
    if(t%100==0): print(t)
    x=ra.normal(0,1,d)
    z=ra.normal(0,1,d)
    if t==0:
        X_test_his=[x]
        Z_test_his=[z]
    else:
        X_test_his= np.concatenate((X_test_his, [x]),axis=0)
        Z_test_his = np.concatenate((Z_test_his, [z]), axis=0)
    result=np.matmul(np.matmul(x,Theta),z)+ra.normal(0,R)
    y_his.append(result)
    UU,VV, out_nIter, stat = rankone(np.array(X_test_his),np.array(Z_test_his),y_his,r,R)
    theta_test=np.outer(UU,VV)
    print(la.norm(theta_test-Theta))

print(theta_test)
'''