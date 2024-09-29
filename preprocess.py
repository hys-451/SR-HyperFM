# -*- coding: cp936 -*-
train_path = 'train.txt'
f1 = open(train_path,'r')
dic= {}
# generate three fold.
# train_x: value
# train_i: index
# train_y: label
f_train_value = open('train_x.txt','w')
f_train_index = open('train_i.txt','w')
f_train_label = open('train_y.txt','w')

for i in range(14):
    dic[i] = {}# create dic[0]={}, dic[1]={}, dic[2]={}, ... ,dic[38]={},

cnt_train = 0

#for debug
#limits = 10000
index = [1] * 8 #[1, 1, 1, ..., 1]total number of each category label (except those appear times <=10)
for line in f1:
    cnt_train +=1
    if cnt_train % 1000 ==0:
        print('now train cnt : %d\n' % cnt_train)

    split = line.strip('\n').split('\t')
    #print(split)
    # 0-label, 1-13  1-6 numerical, 14-39 7-14 category 
    for i in range(6,14):
        if split[i+1] not in dic[i]:
        # [1, 0] 1 is the index for those whose appear times <=3   0 indicates the appear times
            dic[i][split[i+1]] = [1,0]  
        dic[i][split[i+1]][1] += 1
        if dic[i][split[i+1]][0] == 1 and dic[i][split[i+1]][1] > 2:
            index[i-6] += 1
            dic[i][split[i+1]][0] = index[i-6]
f1.close()
print('total entries :%d\n' % (cnt_train - 1))

# calculate number of category features of every dimension
kinds = [6]#13 numerical features + ? categorial features
for i in range(6,14):
    kinds.append(index[i-6])
print('number of dimensions : %d' % (len(kinds)-1))
print(kinds)

for i in range(1,len(kinds)):
    kinds[i] += kinds[i-1]
print(kinds)

# make new data

f1 = open(train_path,'r')
cnt_train = 0
print('remake training data...\n')
for line in f1:
    cnt_train +=1
    if cnt_train % 1000 ==0:
        print('now train cnt : %d\n' % cnt_train)
    #if cnt_train > limits:
    #	break
    entry = ['0'] * 14
    index = [None] * 14
    split = line.strip('\n').split('\t')
    label = str(split[0])
    for i in range(6):
        if split[i+1] != '':
            entry[i] = (split[i+1])
        index[i] = (i+1)
    for i in range(6,14):
        if split[i+1] != '':
            entry[i] = '1'#for category feature, if it is not null, entry value = 1 else 0
        index[i] = (dic[i][split[i+1]][0])# 在该dimension下的该特征出现的索引，1表示出现<=10次
    for j in range(8):
        index[6+j] += kinds[j]
    index = [str(item) for item in index]
    f_train_value.write(' '.join(entry)+'\n')
    f_train_index.write(' '.join(index)+'\n')
    f_train_label.write(label+'\n')
f1.close()


f_train_value.close()
f_train_index.close()
f_train_label.close()


