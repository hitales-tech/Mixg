import random
import numpy as np
from sklearn.linear_model import LogisticRegression,Lasso
from xgboost.sklearn import XGBClassifier
import os

import pickle
import sys
import time
import pandas as pd

output_dir = 'C:/tests/out02/'
input_dir = 'C:/tests/out02/'
input_dir2 = 'c:/tests/out02/'
mtype = 'xgb'

global_fs_th = 0.7
pos_fs_th = 0.7
select_fs_size = 90
min_fs_th = 0.25
min_diag_size = 500

args = sys.argv
if(len(args)>=4):
    input_dir = args[1]
    input_dir2 = args[2]
    output_dir = args[3]
    mtype = args[4]
print('Input DIR:',input_dir,input_dir2)
print('Output DIR:',output_dir)
print('Model Type:',mtype)

df = '%Y%m%d%H%M'
version = mtype+time.strftime(df,time.localtime(time.time()))
print(version)
file = open(output_dir+'version.txt','w+')
file.write(version)
file.close()
max_size = 9999999
print('Start Train')

f = open(input_dir2+'test2tag.txt', encoding='utf-8')
line = f.readline()
knowledges = {}
while line:
    s = line[:-1]
    idx = s.index("=")
    tags = s[idx+1:].split(";")
    hy = s[:idx]
    for tag in tags:
        if (not knowledges.__contains__(tag)):
            knowledges[tag] = []
        if(not knowledges[tag].__contains__(hy)):
            knowledges[tag].append(hy)
    line = f.readline()
f.close()
# print(knowledges)
# file = open(output_dir + '~测试', 'wb+')
# pickle.dump('test', file);
# file.close()

file = open(input_dir+'infos.pkl', 'rb+')
infos = pickle.load(file)
diag_indexes = infos['diag_indexes']
inverted_diag_indexes = infos['inverted_diag_indexes']
test_indexes = infos['test_indexes']
inverted_test_indexes = infos['inverted_test_indexes']
file.close()

file = open(input_dir+'datas_train.pkl', 'rb+')
datas = pickle.load(file)
file.close()

ds = datas['datas']
tags = datas['tags']
ignores = datas['ignores']
all_test_counter = {}

print('DATA SIZE',len(ds))

for key in test_indexes:
    all_test_counter[key] = 0

for i in range(0,len(ds)):
    for t in ds[i]:
        all_test_counter[inverted_test_indexes[t]] = all_test_counter[inverted_test_indexes[t]]+1

fixed_tests = []
def sort1(elem):
    return elem['score']
scores = []
for key in test_indexes:
    if(all_test_counter[key]*1.0/len(ds)>=global_fs_th):
        # fixed_tests.append(key)
        scores.append({'name':key,'score':all_test_counter[key]})
scores.sort(key=sort1,reverse=True)
fixed_tests.append('性别')
fixed_tests.append('年龄')
# for j in range(0,int(select_fs_size/4)):
#     if(j>=len(scores)):
#         break
#     fixed_tests.append(scores[j]['name'])
print('Fixed test:',len(fixed_tests),fixed_tests)

isExists=os.path.exists(output_dir+"models")
if not isExists:
        os.makedirs(output_dir+"models")

diag_indexes2 = {}
inverted_diag_indexes2 = {}


for i in range(0,len(diag_indexes)):
    # if(not inverted_diag_indexes[i].__contains__('~肿瘤')):
    #     continue
    diag = inverted_diag_indexes[i]
    print('Processing', str(i) + "/" + str(len(diag_indexes)), diag)
    pos_count = 0
    test_counter = {}
    for t in test_indexes:
        test_counter[t] = 0
    for j in range(0, len(ds)):
        if tags[j].__contains__(diag):
            pos_count = pos_count+1
            for t in ds[j]:
                test_counter[inverted_test_indexes[t]] = test_counter[inverted_test_indexes[t]]+1
    ktests = []
    if(knowledges.__contains__(diag)):
        ktests = knowledges[diag]
    # print(pos_count)
    # print(ktests)
    if(pos_count<min_diag_size):
        continue
    diag_indexes2[inverted_diag_indexes[i]] = len(diag_indexes2)
    inverted_diag_indexes2[len(inverted_diag_indexes2)] = inverted_diag_indexes[i]
    selected = []
    for t in fixed_tests:
        selected.append(t)
    scores = []
    for t in test_counter:
        if(test_counter[t]/pos_count>=pos_fs_th):
            if(not selected.__contains__(t)):
                scores.append({'name':t,'score':test_counter[t]/pos_count-all_test_counter[t]/len(ds)})
    scores.sort(key=sort1, reverse=True)
    for j in range(0, len(scores)):
        t = scores[j]['name']
        # print(scores[j],test_counter[t]/pos_count)
        if(scores[j]['score']<0.1):
            break
        if(len(selected)>=select_fs_size/3):
            break
        if(test_counter[t]/pos_count<=min_fs_th):
            continue
        selected.append(scores[j]['name'])


    for t in ktests:
        if (not selected.__contains__(t) and test_indexes.__contains__(t) and test_counter[t]/pos_count>=min_fs_th):
            # print('2',t,test_counter[t]*1.0/pos_count,diag)
            selected.append(t)
    scores = []
    for t in test_indexes:
        if(test_counter[t]/pos_count<min_fs_th):
            continue
        a1 = []
        a2 = []
        for j in range(0,len(ds)):
            if((diag in tags[j]) or (not diag in ignores[j])):
               if(ds[j].__contains__(test_indexes[t])):
                    if(diag in tags[j]):
                        a1.append(1.0)
                    else:
                        a1.append(0.0)
                    a2.append(float(ds[j][test_indexes[t]]))
        if(len(a1)>=min_diag_size):
            p1 = pd.Series(a1)
            p2 = pd.Series(a2)
            s0 = p1.corr(p2,method='pearson')
            s = abs(s0)
            if(not np.isnan(s)):
                # print(t,s0)
                scores.append({'name':t,'score':s})
                # print(t,s)
    scores.sort(key=sort1,reverse=True)
    for j in range(0,int(select_fs_size*2/3)):
        if(j>=len(scores)):
            break
        # print(3, scores[j]['name'], diag)
        if(not selected.__contains__(scores[j]['name'])):
            selected.append(scores[j]['name'])
            # print(3, scores[j]['name'], diag)
    selected2 = []
    for t in selected:
        if(t.endswith('_P')):
            dp = 0
            if ((t[0:-2]+"_H") in selected):
                dp = dp+1
            if ((t[0:-2]+"_L") in selected):
                dp = dp + 1
            if(not dp==1):
                selected2.append(t)
            # if(dp==2):
            #     print("!!!!!",t,diag,selected)
        else:
            selected2.append(t)
    selected = selected2
    print(diag,len(selected),selected)
    select_indexes = {}
    inverted_select_indexes = {}
    for t in selected:
        select_indexes[t] = len(select_indexes)
        inverted_select_indexes[len(inverted_select_indexes)] = t
    # print(select_indexes)
    # print(inverted_select_indexes)
    # for t in selected
    train_datas = []
    train_tags = []
    for j in range(0,len(ds)):
        if ((diag in tags[j]) or (not diag in ignores[j])):
            d = []
            tg = 0
            if(diag in tags[j]):
                tg = 1
            for k in range(0, len(select_indexes)):
                if (inverted_select_indexes[k] == '血_游离/总前列腺特异性抗原_V'):
                    # print('here!!!!')
                    d.append(random.uniform(0.25, 1.0))
                else:
                    if (inverted_select_indexes[k].endswith('_V')):
                        d.append(random.uniform(0.0, 1.0))
                    else:
                        d.append(0.0)
            for t in ds[j]:
                if(select_indexes.__contains__(inverted_test_indexes[t])):
                    d[select_indexes[inverted_test_indexes[t]]] = float(ds[j][t])
            train_datas.append(d)
            train_tags.append(tg)
            # print(tg,d)
    model = XGBClassifier(n_estimators=50,max_depth=2)
    if (mtype == 'lr'):
        model = LogisticRegression()
    model.fit(np.array(train_datas), np.array(train_tags))
    file = open(output_dir + 'models/'+diag+'.pkl', 'wb+')
    pickle.dump({'model': model, 'select_indexes':select_indexes,'inverted_select_indexes':inverted_select_indexes}, file);
    file.close()
    # from sklearn.metrics import auc, roc_curve,recall_score
    # prs = model.predict_proba(np.array(train_datas))
    # ps = model.predict(np.array(train_datas))
    # fpr, tpr, thresholds = roc_curve(train_tags, prs[:,1])
    # ascore = auc(fpr, tpr)
    # rscore = recall_score(train_tags,ps)
    # print(diag,ascore,pos_count,rscore)

print(diag_indexes2,inverted_diag_indexes2)

print('Done')


