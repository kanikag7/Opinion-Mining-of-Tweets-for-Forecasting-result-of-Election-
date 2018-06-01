import numpy as np
import random
import operator
def voting(predicted):
    new_predicted=[None for i in range(len(predicted[0]))]
    for i in range(len(predicted[0])):
        count_label={}
        for j in range(len(predicted)):
            try:
                if count_label.has_key(predicted[j][i]):
                    count_label[predicted[j][i]]+=1
                else:
                    count_label[predicted[j][i]]=0
            except IndexError:
                print "i:",i,"j:",j,"\n"
        new_predicted[i]=max(count_label.iteritems(), key=operator.itemgetter(1))[0]
    return new_predicted
        
    
def balance_data(train_data,labels):
    new_train_data=[[] for i in range(0,5)]
    new_train_label=[[] for i in range(0,5)]
    label_hash={}
    for i in range(0,len(labels)):
        key=labels[i]
        if key in label_hash:
            label_hash[key].append(i)
        else:
            label_hash[key]=[i]
    n=len(train_data)
    div_limit=n/5
    
    for i in range(0,5):
           
        for key in label_hash:
            data=[]
            try:
                data.append(random.sample(label_hash[key], div_limit))
            except ValueError:
                data.append(random.sample(label_hash[key],len(label_hash[key])))
            #new_train_data[i].append(data)
            for d in data:
                for element in d:
                    new_train_data[i].append(train_data[element])
                    new_train_label[i].append(labels[element])
            #print "i",i,"\n",len(new_train_data[i])
    return new_train_data,new_train_label
            
    
            #print len(label_hash[key]),"\n"
def find_accuracy(test_labels,predicted):
    count=0
    for t,p in zip(test_labels,predicted):
        if t==p:
            count=count+1
    return count
            
             
    