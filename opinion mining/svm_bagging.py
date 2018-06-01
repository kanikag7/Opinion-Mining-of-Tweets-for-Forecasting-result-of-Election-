from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from preprocess import  preprocess
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit
from balance_train_data import balance_data
from balance_train_data import voting
from balance_train_data import find_accuracy
from sklearn import metrics
from sklearn import cross_validation
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
p=preprocess()


feature_file=open("obama_words.txt","r")
class_file=open("obama_labels.txt","r")
features=feature_file.readlines()
labels=class_file.readlines()
i=0
for str in labels:
    labels[i]=str.replace("\n", "")
    i=i+1
kf = cross_validation.StratifiedKFold(labels, n_folds=10,shuffle=False)
accuracy=0
accuracy1=0
total_count=0
obama_cnt=0
romney_count=0
precision=[0 for i in range(3)]
recall=[0 for i in range(3)]
fscore=[0 for i in range(3)]
clf = svm.LinearSVC()
for train_index, test_index in kf:
    #print("TRAIN:", (train_index), "TEST:", (test_index))
    #Fetching training features
    train_features=list( features[i] for i in train_index )
    #Fetching train labels
    train_labels=list( labels[i] for i in train_index )
    [new_train_data,new_train_label]=balance_data(train_features,train_labels)
    count_vect = CountVectorizer()
    bag_classifier=[]
    #Fetching test features
    test_features=list( features[i] for i in test_index )
    #Fetching test labels
    test_labels=list( labels[i] for i in test_index )
    #count_vect = CountVectorizer(ngram_range=(1, 3)
    predicted=[]
    for batch,batch_label in zip(new_train_data,new_train_label):
        #count word_freq
        X_train_counts = count_vect.fit_transform(batch)
        #TF IDF object instantiation
        tfidf_transformer = TfidfTransformer()
        #TF IDF object fitting to data
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        #Train Naive Bayes

        clf.fit(X_train_tfidf,batch_label)
        X_new_counts = count_vect.transform(test_features)
        X_new_tfidf = tfidf_transformer.transform(X_new_counts)
        predicted.append (clf.predict(X_new_tfidf))

    new_predicted=voting(predicted)

    for k in new_predicted:
        #print type(k)

        total_count+=1
        if(k==np.string_(1.0)):
            obama_cnt+=1
        elif (k==np.string_(-1.0)):
            romney_count+=1
    accuracy+= metrics.accuracy_score(test_labels, new_predicted, normalize=True, sample_weight=None)
    metric= precision_recall_fscore_support(test_labels, new_predicted)
    precision+=metric[0]
    recall+=metric[1]
    fscore+=metric[2]
precision=precision/10
recall=recall/10
fscore=fscore/10
print precision,"\n", recall,"\n",fscore,"\n"
print accuracy/10
romney_win_percentage = float(float(romney_count)/float(total_count))*100
print("The chances that romney will win is: ")
print(romney_win_percentage)
obama_win_percentage  = 100.0 - romney_win_percentage
print("The chances that Obama will win is: ")
print(obama_win_percentage)

#pie chart creation
labels = 'Romney', 'Obama'
sizes = [romney_win_percentage, obama_win_percentage ]
colors = ['gold', 'yellowgreen']
explode = (0.1, 0.1)  # explode 1st slice

# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')
plt.show()
