import random
import numpy as np
from sklearn.linear_model import LogisticRegression,Lasso
from xgboost.sklearn import XGBClassifier
import pickle
import sys
import time
import os
import sys

output_path = 'C:/tests/out2/scores_valid.pkl'
input_dir = 'C:/tests/out2/'  #python目录
input_dir2 = 'c:/tests/out0/' #java目录
source = "valid"

args = sys.argv
args = sys.argv
if(len(args)>=4):
    input_dir = args[1]
    input_dir2 = args[2]
    output_path = args[3]
    source = args[4]

print('Start Check')
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
print('Loading Knowledges Done')

file = open(input_dir+'infos.pkl', 'rb+')
infos = pickle.load(file)
test_indexes = infos['test_indexes']
inverted_test_indexes = infos['inverted_test_indexes']
file.close()
print('Loading Infos Done')

file = open(input_dir+'datas_'+source+'.pkl', 'rb+')
datas = pickle.load(file)
file.close()
print('Loading Dataset '+source+' Done')
print(datas.keys())
datas2 = []
files = os.listdir(input_dir+'models/')
all_tags = []
true_tags = []
scores = []

for f in files:
    tag = os.path.basename(f)
    if(tag.endswith('.pkl') and not tag=='~测试.pkl' and not tag=='测试.pkl'):
        file = open(input_dir+'models/'+tag, 'rb+')
        mpack = pickle.load(file)
        tag = tag[:-4]
        all_tags.append(tag)
        file.close()

for i in range(0,len(datas['tags'])):
    tags = []
    for tag in datas['tags'][i]:
        if(all_tags.__contains__(tag)):
            tags.append(tag)
    if(len(tags)>0):
        true_tags.append(tags)
        scores.append([])
        datas2.append(datas['datas'][i])

print(len(datas['datas']),len(datas2))

for i in range(0,len(all_tags)):
    print('Predicting ',all_tags[i],i+1,'/',len(all_tags))
    tag = all_tags[i]
    file = open(input_dir + 'models/' + tag+'.pkl', 'rb+')
    mpack = pickle.load(file)
    datas3 = []
    valids = []
    for j in range(0,len(datas2)):
        data = []
        valid = False
        for k in range(0,len(mpack['inverted_select_indexes'])):
            if (mpack['inverted_select_indexes'][k].endswith('_V')):
                data.append(0.5)
            else:
                data.append(0)
        for index in datas2[j]:
            test = inverted_test_indexes[index]
            if(knowledges.__contains__(tag) and knowledges[tag].__contains__(test)):
                valid = True
            if(mpack['select_indexes'].__contains__(test)):
                data[mpack['select_indexes'][test]] = float(datas2[j][index])
        valids.append(valid)
        datas3.append(data)
    datas3 = np.array(datas3)
    scores2 = mpack['model'].predict_proba(datas3)
    for j in range(0,len(scores2)):
        if(valids[j]):
            scores[j].append(scores2[j][1])
        else:
            scores[j].append(0)
    # print(len(scores2),len(scores))
    # print(tag,scores)
    file.close()


print(len(scores),len(true_tags),len(scores[0]),len(scores[len(scores)-1]),len(all_tags))
save_data = {'scores':scores,'all_tags':all_tags,'true_tags':true_tags}
file = open(output_path, 'wb+')
pickle.dump(save_data, file);
file.close()

print('Done')