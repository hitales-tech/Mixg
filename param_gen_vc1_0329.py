from xgboost.sklearn import XGBClassifier
import pickle
from xgboost import plot_importance
import matplotlib.pyplot as plt
import os
import sys
kb_file = 'c:/tests/out0/test2tag.txt'
input_dir = 'C:/tests/out01/'
# model_dir = 'C:/test/zd2/model1029/models/'
save_path = 'c:/tests/XGB模型系数.xls'
mtype = 'xgb' #lr or xgb

args = sys.argv
if(len(args)>=2):
    kb_file = args[1]
if(len(args)>=3):
    input_dir = args[2]
if(len(args)>=4):
    save_path = args[3]
if(len(args)>=5):
    mtype = args[4]

print('Knowledge Path:',kb_file)
print('Input Dir:',input_dir)
print('Save Path:',save_path)
print('Model Type:',mtype)

print('Start Print')

file = open(input_dir+'infos.pkl', 'rb+')
infos = pickle.load(file)
test_indexes = infos['test_indexes']
inverted_test_indexes = infos['inverted_test_indexes']
file.close()

file = open(kb_file,'r',encoding="utf-8")
lines = file.readlines()
knowledges = {}
models = {}
for line in lines:
    line = line.replace("\n","")
    arr = line.split("=")
    test = arr[0]
    idx = test.rfind('_')
    if(idx>0):
        test = test[:idx]
    icds = arr[1].split(";")
    if(not knowledges.__contains__(test)):
        knowledges[test] = []
    for tag in icds:
        if(not knowledges[test].__contains__(tag)):
            knowledges[test].append(tag)
file.close()

import xlwt
workbook = xlwt.Workbook()
sheet = workbook.add_sheet("模型参数")
sheet.write(0, 0, '诊断/标签')
sheet.write(0, 1, '特征')
sheet.write(0, 2, '原始权重')
sheet.write(0,3,'加权权重')
sheet.write(0,4,'是否在知识体系内')
out_row = 1
model_dir = input_dir+"models/"
files = os.listdir(model_dir)

def get_coefs(model):
    if(mtype=='lr'):
        return model.coef_[0]
    if(mtype=='xgb'):
        importance = model.get_booster().get_score(importance_type='gain')
        res = []
        for i in range(0,len(test_indexes)):
            res.append(0)
        for key in importance:
            idx = int(key[1:])
            res[idx] = importance[key]
        return res
    return None

for f in files:
    name = str(f)
    if(name.endswith(".pkl") and (not name=='~测试')):
        file = open(model_dir + name, 'rb+')
        if(name.endswith('.pkl')):
            name = name[0:-4]
        minfo = pickle.load(file)
        model = minfo['model']
        tests = minfo['inverted_select_indexes']
        vall = 0
        vs = get_coefs(model)
        # vs = model.get_booster().get_score(importance_type='gain')
        for v in vs:
            vall = vall+v*v
        for i in range(0,len(tests)):
            sheet.write(out_row,0,name)
            sheet.write(out_row,1,tests[i])
            sheet.write(out_row,2,vs[i])
            sheet.write(out_row,3,vs[i]*vs[i]/vall)
            idx = tests[i].rfind('_')
            tname = tests[i]
            if(idx>0):
                tname = tname[:idx]
            ikb = False
            if(knowledges.__contains__(tname) and knowledges[tname].__contains__(name)):
                ikb = True
            sheet.write(out_row,4,ikb)
            out_row = out_row+1
workbook.save(save_path)
print('Done')
