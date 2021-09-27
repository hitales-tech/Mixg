import random
import numpy as np
from sklearn.linear_model import LogisticRegression,Lasso
from xgboost.sklearn import XGBClassifier
import pickle
import sys
import time
import os

print('START GA OPT')
score_path = 'C:/tests/out2/scores.pkl'
output_dir = 'c:/tests/out2/opt/'
max_run = 20
step_size = 1

args = sys.argv
if(len(args)>=4):
    score_path = args[1]
    output_dir = args[2]
    max_run = int(args[3])

file = open(score_path, 'rb+')
score_infos = pickle.load(file)
all_tags = score_infos['all_tags']
true_tags = score_infos['true_tags']
tag_index = {}
scores = score_infos['scores']
file.close()
for i in range(0,len(all_tags)):
    tag_index[all_tags[i]] = i
# print(true_tags[0])
# print(all_tags,tag_index)

def func(elem):
    return elem[1]

def getMAP(scores,trs,tags):
    scores2 = []
    all_map = 0
    for i in range(0,len(scores)):
        scores2.append([])
        for j in range(0,len(scores[i])):
            v = scores[i][j]
            v2 = v/trs[j]*0.5
            if(v>trs[j]):
                v2 = 0.5+(v-trs[j])/(1-trs[j])*0.5
            scores2[i].append([j,v2])
        scores2[i].sort(key=func, reverse=True)
        match = 0
        map = 0
        ai_found = 0
        ai_tags = []
        for j in range(0,len(scores2[i])):
            if(scores2[i][j][1]<0.5):
                break
            ai_found = ai_found + 1
            ai_tags.append(all_tags[scores2[i][j][0]])
            if(true_tags[i].__contains__(all_tags[scores2[i][j][0]])):
                match = match+1
                map = map+match/(j+1)
        # print(map,ai_found,true_tags[i],ai_tags)
        if(ai_found>0 and len(true_tags[i])>0):
            all_map = all_map+map/min(ai_found,len(true_tags[i]))
    return all_map/len(scores)


print(len(scores))
from sko.GA import GA

trs = []
ubs = []
lbs = []
for key in all_tags:
    trs.append(0.5)
    ubs.append(0.99)
    lbs.append(0.01)

def op_fun(x):
    return -getMAP(scores,x,true_tags)

# times = 0
# for i in range(0,100):
#     time1 = time.time()
#     getMAP(scores, trs, tags)
#     time2 = time.time()
#     times = times+time2-time1
#     print(i,times/(i+1))
# sys.exit(0)
# import torch
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ga = GA(func=op_fun, n_dim=len(all_tags), size_pop=10, max_iter=max_run, lb=lbs, ub=ubs,
        precision=1e-3)

# ga.to(device=device)

run_times = 0
while(run_times<max_run):
    run_times = run_times+step_size
    best_x, best_y = ga.run(step_size)
    print('Times:',run_times,'\tBest:',-best_y[0])
    best_x2 = {}
    for i in range(0,len(best_x)):
        best_x2[all_tags[i]] = best_x[i]
    print(best_x2)
    file = open(output_dir + 'opt_'+str(run_times), 'wb+')
    info = {'trs':best_x2,'score':-best_y[0],'time':run_times}
    pickle.dump(info,file)
    file.close()

print('Done')