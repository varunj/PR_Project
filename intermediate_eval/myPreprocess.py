import numpy as np
import tsfresh
import pandas as pd
import tsfresh.feature_extraction.feature_calculators as features

def mad(a):
    med = np.median(a)
    return np.median(np.absolute(a - med))

#Absolute Energy,Kurtosis, Sample Entropy,Longest Strike Above Mean,Longest Strike below Mean,Skewness,cid_ce,Mean Absolute Change,mean_second_derivative_central
func=[features.abs_energy,features.kurtosis,features.sample_entropy,features.longest_strike_above_mean,features.longest_strike_below_mean,features.skewness,features.cid_ce,features.mean_abs_change,features.mean_second_derivative_central]

extra_para={features.cid_ce:'True'}
overlap=75

for j in range(1,23):
    data = np.genfromtxt(str(j) + '.csv', delimiter=',')
    length = len(data)
    n_block = length // overlap + 1
    ret = np.zeros(shape=(n_block, 10+len(func)*3))
    curr_index = 0
    i = 0
    while i < length:
        block = data[i:i+100, 1:4]
        X,Y,Z=block[:,0],block[:,1],block[:,2]
        val = np.array([j,X.mean(), Y.mean(), Z.mean(),X.std(),Y.std(),Z.std(),mad(X), mad(Y), mad(Z)])
        for f in func:
            for s in [X,Y,Z]:
                if f in extra_para.keys():
                    dt=f(pd.Series(X),extra_para[f])
                    val=np.append(val,dt)
                else:
                    dt=f(pd.Series(X))
                    val=np.append(val,dt)
        ret[curr_index] = val
        curr_index += 1
        i += overlap
    np.savetxt("./Feat/"+str(j) + '_ret_o_75.csv', ret, delimiter=',')
    print(j)
with open("./Feat/total_o_75.csv", "a") as file:
    for i in range(1, 23):
        with open("./Feat/"+str(i)+'_ret_o_75.csv', 'r') as f:
            file.write(f.read())
