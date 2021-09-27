import random
import numpy as np
from sklearn.linear_model import LogisticRegression,Lasso
from xgboost.sklearn import XGBClassifier
import pickle
import sys
import time
import os
from sklearn.metrics import precision_score,recall_score,f1_score,auc,roc_curve
import xlwt

print('Start Report Generator')

args = sys.argv

score_path = 'C:/tests/out01/scores_test2.pkl'
if(len(args)>=2):
    score_path = args[1]
report_path = 'c:/tests/report_test_0.xls'
if(len(args)>=3):
    report_path = args[2]
pic_path = 'c:/tests/pics/'
if(len(args)>=4):
    pic_path = args[3]
opt_path = None
if(len(args)>=5):
    if(args[4]=='None' or args[4]=='none'):
        opt_path = None
    else:
        opt_path = args[4]
# opt_path = 'c:/tests/out2/opt/opt_10'

topn = 10
print('Score path',score_path)
print('Report path',report_path)
print('Pic path',pic_path)
print('Opt path',opt_path)

file = open(score_path, 'rb+')
score_infos = pickle.load(file)
print(score_infos.keys())
all_tags = score_infos['all_tags']
true_tags = score_infos['true_tags']
scores = score_infos['scores']
file.close()

print(len(scores),len(scores[0]),len(all_tags),len(true_tags))

trs = {}
if(opt_path==None):
    for i in range(0,len(all_tags)):
        trs[all_tags[i]]=0.5
else:
    file = open(opt_path, 'rb+')
    opt = pickle.load(file)
    trs = opt['trs']
    file.close()

# for i in range(0,len(all_tags)):
#     sarr = []
#     for j in range(0, len(scores)):
#         score = scores[j][i]
#         if(all_tags[i] in true_tags[j]):
#             sarr.append(score)
#     sarr.sort()
#     trs[all_tags[i]] = float(sarr[int(len(sarr)*0.15)])
#     print(all_tags[i],trs[all_tags[i]])
#     if(trs[all_tags[i]]>0.5):
#         trs[all_tags[i]] = 0.5
# print(trs)
import xlwt
row = 1
workbook = xlwt.Workbook()
sheet = workbook.add_sheet("单模型")
sheet.write(0, 0, '诊断/标签')
sheet.write(0, 1, '阈值')
sheet.write(0, 2, '样本数')
sheet.write(0,3,'阳性样本数')
sheet.write(0,4,'AUC')
sheet.write(0,5,'敏感性(召回率)')
sheet.write(0,6,'特异性')
sheet.write(0,7,'精准率')
sheet.write(0,8,'真阴性')
sheet.write(0,9,'假阴性')
sheet.write(0,10,'假阳性')
sheet.write(0,11,'真阳性')
sheet.write(0,12,'F1')

all_tp = 0
all_fp = 0
all_tn = 0
all_fn = 0

import pylab as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']

for i in range(0,len(all_tags)):

    p = []
    pr = []
    t = []
    for j in range(0, len(scores)):
        score = scores[j][i]
        if(true_tags[j].__contains__(all_tags[i])):
            t.append(1)
        else:
            t.append(0)
        pr.append(score)
        if(score>=trs[all_tags[i]]):
            p.append(1)
        else:
            p.append(0)
    fpr, tpr, thresholds = roc_curve(t, pr)
    ascore = auc(fpr, tpr)

    plt.figure(figsize=(8, 8))
    plt.title(all_tags[i][1:]+' ROC')
    plt.plot(fpr, tpr, 'b', label='Val AUC = %0.3f' % ascore)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(pic_path+all_tags[i]+'.png')
    plt.close()
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for j in range(0,len(t)):
        if(t[j]==1):
            if(p[j]==1):
                tp = tp+1
            else:
                fn = fn+1
        else:
            if(p[j]==1):
                fp = fp+1
            else:
                tn = tn+1
    prec = 0
    rec = 0
    ty = 0
    f1 = 0
    pcount = tp+fn
    if(tp+fp>0):
        prec = tp/(tp+fp)
    if(tp+fn>0):
        rec = tp/(tp+fn)
    if(tn+fp>0):
        ty = tn/(tn+fp)
    if(prec>0 and rec>0):
        f1 = 2*prec*rec/(prec+rec)
    ascore = round(ascore,4)
    prec = round(prec,4)
    rec = round(rec,4)
    ty = round(ty,4)
    f1 = round(f1,4)
    if(np.isnan(ascore)):
        ascore = 0
    sheet.write(row, 0, all_tags[i])
    sheet.write(row, 1, float(trs[all_tags[i]]))
    sheet.write(row, 2, len(t))
    sheet.write(row, 3, pcount)
    sheet.write(row, 4, ascore)
    sheet.write(row, 5, rec)
    sheet.write(row, 6, ty)
    sheet.write(row, 7, prec)
    sheet.write(row, 8, tn)
    sheet.write(row, 9, fn)
    sheet.write(row, 10, fp)
    sheet.write(row, 11, tp)
    sheet.write(row, 12, f1)
    row = row+1
    all_fp = all_fp+fp
    all_tp = all_tp+tp
    all_fn = all_fn+fn
    all_tn = all_tn+tn
def func(elem):
    return elem[1]

stat = []

for i in range(0,topn):
    stat.append({'tp':0,'tn':0,'fp':0,'fn':0,'map':0})

def getMAP(scores,trs,tags):
    scores2 = []
    all_map = 0
    for i in range(0,len(scores)):
        maps = []
        founds = []
        for j in range(0, topn):
            maps.append(0)
            founds.append(0)
        scores2.append([])
        for j in range(0,len(scores[i])):
            v = scores[i][j]
            v2 = v/trs[all_tags[j]]*0.5
            if(v>trs[all_tags[j]]):
                v2 = 0.5+(v-trs[all_tags[j]])/(1-trs[all_tags[j]])*0.5
            scores2[i].append([j,v2])
        scores2[i].sort(key=func, reverse=True)
        found = 0
        map = 0
        ntag = len(tags[i])
        found1 = 0
        for j in range(0,len(scores2[i])):
            if(scores2[i][j][1]>=0.5):
                found1 = found1+1
                if(all_tags[scores2[i][j][0]] in tags[i]):
                    found = found+1
                    map = map+found/(j+1)
            if(j<topn):
                if (all_tags[scores2[i][j][0]] in tags[i]):
                    for k in range(j, topn):
                        founds[k] = founds[k]+1
                        maps[k] = maps[k]+founds[k]/(j+1)
                        stat[k]['tp'] = stat[k]['tp']+1
                else:
                    for k in range(j, topn):
                        stat[k]['fp'] = stat[k]['fp']+1
        if(found1>0 and min(ntag,found1)>0):
            all_map = all_map+map/min(ntag,found1)
        for j in range(0,topn):
            if(founds[j]>0):
                if(found>0):
                    stat[j]['map'] = stat[j]['map']+maps[j]/min(j+1,ntag)
    return all_map/len(scores)

aprec = all_tp/(all_tp+all_fp)
arec = all_tp/(all_tp+all_fn)
aty = all_tn/(all_tn+all_fp)
af1 = 2*aprec*arec/(aprec+arec)
map = getMAP(scores,trs,true_tags)
line = '整体：\tMAP:'+str(map)+"\t精准率："+str(aprec)+"\t召回率(敏感性):"+str(arec)\
    +"\t特异性："+str(aty)+"\tF1："+str(af1)
print(line)
sheet2 = workbook.add_sheet("整体性能")
sheet2.write(0,0,'TopN')
sheet2.write(0, 1, 'MAP')
sheet2.write(0, 2, '敏感性(召回率)')
sheet2.write(0, 3, '特异性')
sheet2.write(0, 4, '精准率')
sheet2.write(0, 5, 'F1')

sheet2.write(1,0,'ALL')
sheet2.write(1, 1, round(map,4))
sheet2.write(1, 2, round(arec,4))
sheet2.write(1, 3, round(aty,4))
sheet2.write(1, 4, round(aprec,4))
sheet2.write(1, 5, round(af1,4))

for i in range(0,topn):
    sheet2.write(i+2,0,'Top'+str(i+1))
    sheet2.write(i + 2, 1, round(stat[i]['map']/len(scores),4))
    rec1 = 0
    prec1 = 0
    if(stat[i]['tp']>0):
        prec1 = stat[i]['tp']/(stat[i]['tp']+stat[i]['fp'])
        rec1 = stat[i]['tp']/(all_tp+all_fn)
    ty1 = 0
    tn1 = len(all_tags)*len(scores)-stat[i]['fp']-stat[i]['tp']
    print(stat[i]['tp'], stat[i]['tp']+stat[i]['fp'],tn1)
    if(tn1>0):
        ty1 = tn1/(tn1+stat[i]['fp'])
    sheet2.write(i + 2, 2, round(rec1,4))
    sheet2.write(i + 2, 3, round(ty1, 4))
    sheet2.write(i + 2, 4, round(prec1, 4))
    f11 = 0
    if(prec1>0):
        f11 = prec1*rec1*2/(prec1+rec1)
    sheet2.write(i + 2, 5, round(f11, 4))
workbook.save(report_path)

print('Done')