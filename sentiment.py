import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.tokenize import word_tokenize
from nltk.classify import ClassifierI
from statistics import mode
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.corpus import stopwords


docs = []

pos_re_view = open("re_views/pos_.txt","r").read()
neg_re_view = open("re_views/neg_.txt","r").read()



for t in neg_re_view.split('\n'):
    docs.append( (t, "neg") )

for t in pos_re_view.split('\n'):
    docs.append( (t, "pos") )


whole_word_set = []

neg_re_view_words = word_tokenize(neg_re_view)
neg_re_view_words = [word for word in neg_re_view_words if word not in stopwords.words('english')]
pos_re_view_words = word_tokenize(pos_re_view)
pos_re_view_words = [word for word in pos_re_view_words if word not in stopwords.words('english')]


for wrd in neg_re_view_words:
    whole_word_set.append(wrd.lower())
for wrd in pos_re_view_words:
    whole_word_set.append(wrd.lower())


whole_word_set = nltk.FreqDist(whole_word_set)

word_features = list(whole_word_set.keys())[:5000]


def find_features_set(document):
    words = word_tokenize(document)
    features = {}
    for wrd in word_features:
        features[wrd] = (wrd in words)

    return features

featuresets = [(find_features_set(rev), category) for (rev, category) in docs]
random.shuffle(featuresets)

training_set = featuresets[:10]
testing_set =  featuresets[10:]
training_set = featuresets[:50]
testing_set =  featuresets[50:]
training_set = featuresets[:100]
testing_set =  featuresets[100:]
training_set = featuresets[:500]
testing_set =  featuresets[500:]
training_set = featuresets[:1000]
testing_set =  featuresets[1000:]
training_set = featuresets[:5000]
testing_set =  featuresets[5000:]
training_set = featuresets[:10000]
testing_set =  featuresets[10000:]


NB_classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(NB_classifier, testing_set))*100)
NB_classifier.show_most_informative_features(20)


#SVC_classifier = SklearnClassifier(SVC())
#SVC_classifier.train(training_set)
#print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

#BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
#BernoulliNB_classifier.train(training_set)
#print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

#LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
#LogisticRegression_classifier.train(training_set)
#print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

#LinearSVC_classifier = SklearnClassifier(LinearSVC())
#LinearSVC_classifier.train(training_set)
#print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

#MNB_classifier = SklearnClassifier(MultinomialNB())
#MNB_classifier.train(training_set)
#print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

#SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
#SGDClassifier_classifier.train(training_set)
#print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)



#NuSVC_classifier = SklearnClassifier(NuSVC())
#NuSVC_classifier.train(training_set)
#print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)
