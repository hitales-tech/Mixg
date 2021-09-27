import shap
shap.initjs()
#
#
# pred = model.predict(data.values, output_margin=True)
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(data.values)
#
# shap.summary_plot(shap_values, data)
# # shap.dependence_plot(0, shap_values, data)
# shap.dependence_plot("age", shap_values, data)


import random
import numpy as np
from sklearn.linear_model import LogisticRegression,Lasso
from xgboost.sklearn import XGBClassifier
import pickle
import sys
import time
import os
import sys
import pandas as pd


output_path = 'C:/tests/pics3/'
input_dir = 'C:/tests/out01/'  #python目录
source = 'train'

args = sys.argv
args = sys.argv
if(len(args)>=3):
    input_dir = args[1]
    output_path = args[2]

print('Start Check')

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

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.figure(figsize=(10,10))

for i in range(0,len(all_tags)):
    # if(not all_tags[i]=='~大肠癌'):
    #     continue
    print('Predicting ',all_tags[i],i+1,'/',len(all_tags))
    tag = all_tags[i]
    file = open(input_dir + 'models/' + tag+'.pkl', 'rb+')
    mpack = pickle.load(file)
    datas3 = []
    datas4 = {}
    my_tags = []
    for j in range(0, len(mpack['inverted_select_indexes'])):
        datas4[mpack['inverted_select_indexes'][j]] = []
    for j in range(0,len(datas2)):
        data = []
        if(tag in true_tags[j]):
            my_tags.append(1)
        else:
            my_tags.append(0)
        for k in range(0,len(mpack['inverted_select_indexes'])):
            data.append(0)
        for index in datas2[j]:
            test = inverted_test_indexes[index]
            if(mpack['select_indexes'].__contains__(test)):
                data[mpack['select_indexes'][test]] = float(datas2[j][index])
        for k in range(0, len(mpack['inverted_select_indexes'])):
            datas4[mpack['inverted_select_indexes'][k]].append(data[k])
        datas3.append(data)
    skip = {}
    for key in datas4:
        if(key.endswith('_V')):
            pass
        else:
            if(key.endswith('_P')):
                if (datas4.__contains__(key[:-2]+'_V')):
                    skip[key] = 1
            else:
                if (datas4.__contains__(key[:-2]+'_P') or datas4.__contains__(key[:-2]+'_V')):
                    skip[key] = 1
    datas42 = {}
    for key in datas4:
        if(not skip.__contains__(key)):
            if(len(key)>2):
                datas42[key[0:-2]] = datas4[key]
            else:
                datas42[key] = datas4[key]
    datas3 = np.array(datas3)
    datas5 = pd.DataFrame(datas42)
    # scores2 = mpack['model'].predict_proba(datas5)
    # pred = mpack['model'].predict(datas5, output_margin=True)
    model = XGBClassifier(n_estimators=50,max_depth=2)
    model.fit(datas5.values,np.array(my_tags))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(datas5.values)

    # print(mpack.keys())
    fig = plt.figure(figsize=(12, 6), dpi=150)
    plt.title(tag)
    shap.summary_plot(shap_values, datas5,show=False)
    fig.tight_layout()
    plt.savefig(output_path+tag+".png",dpi=150)
    plt.clf()
    plt.close()
    # shap.dependence_plot(0, shap_values, data)
    # shap.dependence_plot("age", shap_values, data)
print('Done')