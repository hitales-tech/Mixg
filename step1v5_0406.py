import random
import numpy as np
from xgboost.sklearn import XGBClassifier
import pickle
import sys
import time

output = "c:/tests/out02/"
input = "C:/tests/out02/"
args = sys.argv
if(len(args)>=3):
    input = args[1]
    output = args[2]
print('Input DIR:',input)
print('Output DIR:',output)

diag_indexes = {}
inverted_diag_indexes = {}
test_indexes = {}
inverted_test_indexes = {}

datas_train = []
tags_train = []
ignores_train = []
key_train = []

datas_valid = []
tags_valid = []
ignores_valid = []
key_valid = []

datas_test = []
tags_test = []
ignores_test = []
key_test = []



count0 = 0
i0 = 0

stat_all = 0
stat_test = 0

fi = open(input+'datas_train.csv', 'r+', encoding='utf-8')
while 1:
    lines = fi.readlines(20000)
    if not lines:
        break
    for line in lines:
        i0 = i0 + 1
        if(i0%10000==0):
            print('Processing train',i0)
        line = line.replace("\n","")
        arr = line.split(",")
        if(i0>1):
            key = arr[0]+"_"+arr[1]
            tags = set()
            a = arr[2].split(";")
            for tag in a:
                tags.add(tag)
                if(not diag_indexes.__contains__(tag)):
                    diag_indexes[tag] = len(diag_indexes)
                    inverted_diag_indexes[len(inverted_diag_indexes)] = tag
            ignores = {}
            a = arr[3].split(";")
            for tag in a:
                idx = tag.find('=')
                tag0 = tag[0:idx]
                tcount = int(tag[idx+1])
                ignores[tag] = tcount
            if(len(tags)>0):
                d = {}
                for j in range(4,len(arr)):
                    if(len(arr[j])>0):
                        d[j-4] = arr[j]
                datas_train.append(d)
                tags_train.append(tags)
                key_train.append(key)
                ignores_train.append(ignores)
        else:
            for j in range(4,len(arr)):
                test_indexes[arr[j]] = len(test_indexes)
                inverted_test_indexes[len(inverted_test_indexes)] = arr[j]
fi.close()

fi = open(input+'datas_valid.csv', 'r+', encoding='utf-8')
i0 = 0
while 1:
    lines = fi.readlines(20000)
    if not lines:
        break
    for line in lines:
        i0 = i0 + 1
        if(i0%10000==0):
            print('Processing valid',i0)
        line = line.replace("\n","")
        arr = line.split(",")
        if(i0>1):
            key = arr[0]+"_"+arr[1]
            tags = set()
            a = arr[2].split(";")
            for tag in a:
                if(diag_indexes.__contains__(tag)):
                    tags.add(tag)
            ignores = {}
            a = arr[3].split(";")
            for tag in a:
                idx = tag.find('=')
                tag0 = tag[0:idx]
                tcount = int(tag[idx+1])
                ignores[tag] = tcount
            if(len(tags)>0):
                d = {}
                for j in range(4,len(arr)):
                    if(len(arr[j])>0):
                        d[j-4] = arr[j]
                datas_valid.append(d)
                tags_valid.append(tags)
                key_valid.append(key)
                ignores_valid.append(ignores)
fi.close()

fi = open(input+'datas_test.csv', 'r+', encoding='utf-8')
i0 = 0
while 1:
    lines = fi.readlines(20000)
    if not lines:
        break
    for line in lines:
        i0 = i0 + 1
        if(i0%10000==0):
            print('Processing test',i0)
        line = line.replace("\n","")
        arr = line.split(",")
        if(i0>1):
            key = arr[0]+"_"+arr[1]
            tags = set()
            a = arr[2].split(";")
            for tag in a:
                if(diag_indexes.__contains__(tag)):
                    tags.add(tag)
            ignores = {}
            a = arr[3].split(";")
            for tag in a:
                idx = tag.find('=')
                tag0 = tag[0:idx]
                tcount = int(tag[idx+1])
                ignores[tag] = tcount
            if(len(tags)>0):
                d = {}
                for j in range(4,len(arr)):
                    if(len(arr[j])>0):
                        d[j-4] = arr[j]
                datas_test.append(d)
                tags_test.append(tags)
                key_test.append(key)
                ignores_test.append(ignores)
fi.close()

print(len(test_indexes),test_indexes.keys())
print(len(diag_indexes),diag_indexes.keys())
print(len(datas_train),len(datas_valid),len(datas_test))
print(len(tags_train),len(tags_valid),len(tags_test))
print(len(ignores_train),len(ignores_valid),len(ignores_test))
print(len(key_train),len(key_valid),len(key_test))

infos = {'test_indexes':test_indexes,'inverted_test_indexes':inverted_test_indexes,'diag_indexes':diag_indexes,'inverted_diag_indexes':inverted_diag_indexes}
file = open(output+'infos.pkl', 'wb+');
pickle.dump(infos, file);
file.close()

datas_train ={'datas':datas_train,'tags':tags_train,'key':key_train,'ignores':ignores_train}
datas_valid ={'datas':datas_valid,'tags':tags_valid,'key':key_valid,'ignores':ignores_valid}
datas_test = {'datas':datas_test,'tags':tags_test,'key':key_test,'ignores':ignores_test}
file = open(output+'datas_train.pkl', 'wb+');
pickle.dump(datas_train, file);
file.close()
file = open(output+'datas_valid.pkl', 'wb+');
pickle.dump(datas_valid, file);
file.close()
file = open(output+'datas_test.pkl', 'wb+');
pickle.dump(datas_test, file);
file.close()
print("Done!")
