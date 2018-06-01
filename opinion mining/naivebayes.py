from sklearn.feature_extraction.text import CountVectorizer
from preprocess import  preprocess
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn import cross_validation
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit
from balance_train_data import balance_data
from balance_train_data import voting
import matplotlib.pyplot as plt

precision=[0 for i in range(3)]
recall=[0 for i in range(3)]
fscore=[0 for i in range(3)]
fm=open("misclassified.txt","w")
#import count
def count(labels):
    count=[0 for i in range(3)]
    for i in labels:
        i=i.replace("\n", "")
        i=float(i)
        i=int(i)
        count[i+1]+=1
    #print count
def print_top10(vectorizer, clf, class_labels):
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    print feature_names[-10:]
    for i, class_label in enumerate(class_labels):
        feature_names = np.asarray(vectorizer.get_feature_names())
        top10 = np.argsort(clf.coef_[i])[-10:]
        print("%s: %s" % (class_label, " ".join(feature_names[top10])))
p=preprocess()
feature_file=open("obama_words.txt","r")
class_file=open("obama_labels.txt","r")
features=feature_file.readlines()
labels=class_file.readlines()

i=0
#Stripping \n from all labels
for str in labels:
    labels[i]=str.replace("\n", "")
    i=i+1
#Cross validation-Generating indices
count(labels)
kf = cross_validation.KFold(len(labels), n_folds=10)
sss = StratifiedShuffleSplit(labels, 10, test_size=0.1, random_state=0)
accuracy=0
clf = MultinomialNB()
total_count=0
obama_cnt=0
romney_count=0
#Repeating 10 times(10 fold cross validation)
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
        clf.fit(X_train_counts,batch_label)
        X_new_counts = count_vect.transform(test_features)
        X_new_tfidf = tfidf_transformer.transform(X_new_counts)
        predicted.append (clf.predict(X_new_counts))



    count(train_labels)
    #Initializing count vector for counting word frequency

    #clf = MultinomialNB().fit(X_train_counts,train_labels)
    #Test feature transformation

    #predicted = clf.predict(X_new_counts)
    #Add accuracy
    #for i in range(5):
    new_predicted=voting(predicted)
    #print "====================================="
    #print new_predicted
    #print "====================================="

    for k in new_predicted:
        #print type(k)

        total_count+=1
        if(k==np.string_(1.0)):
            obama_cnt+=1
        elif (k==np.string_(-1.0)):
            romney_count+=1


    accuracy+= metrics.accuracy_score(test_labels, new_predicted, normalize=True, sample_weight=None)
    #print precision_recall_fscore_support(test_labels, predicted[i], average='macro')
    metric= precision_recall_fscore_support(test_labels, new_predicted)
    precision+=metric[0]
    recall+=metric[1]
    fscore+=metric[2]
    print(metrics.classification_report(test_labels, new_predicted,target_names=['-1','0','1']))
    #Uncomment this to see Fscore for all tests
    for doc, category,actual in zip(test_features, new_predicted,test_labels):
        if category!=actual:
            fm.write(doc+"\t"+category+"\t"+actual+"\n")
        #print('%r => predicted: %s   actual:%s ' % (doc, category,actual))

print accuracy/10
precision=precision/10
recall=recall/10
fscore=fscore/10
print precision,"\n", recall,"\n",fscore,"\n"
print accuracy/10

#print obama_cnt
#print romney_count
#print total_count
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
#print_top10(count_vect, clf, ['-1.0','0.0','1.0'])
