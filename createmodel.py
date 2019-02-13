from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite



def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag,
        'postag[:2]=' + postag[:2],
    ]
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:postag=' + postag1,
            '-1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('BOS')
        
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:postag=' + postag1,
            '+1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('EOS')
                
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label_1 for token, postag, label_1 , label_2  in sent]

def sent2labelsv2(sent):
    return [label_2 for token, postag, label_1 , label_2 in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]

def file2IOB(filename):
    FILE = open(filename,"r", encoding='utf8', errors='ignore')
    listsent = []
    listword = []
    for line in FILE:
        if line :
            if line == '\n':
                listsent.append(listword)
                listword = []  
            else:
                
                linesplited = line.rstrip().split(' ')
                listword.append(tuple(linesplited))
    FILE.close()
    return listsent

# print(sklearn.__version__)
# nltk.download('abc')
# print(nltk.corpus.abc.fileids())


# print(%%time)

train_sents = file2IOB('./data/eng.train')
# train_sents = list(nltk.corpus.conll2000.iob_sents('train.txt'))
# test_sents = list(nltk.corpus.conll2000.iob_sents('test.txt'))

# print(train_sents)

X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]
z_train = [sent2labelsv2(s) for s in train_sents]

# X_test = [sent2features(s) for s in test_sents]
# y_test = [sent2labels(s) for s in test_sents]
# print(sent2features(train_sents[0])[0])
# print(train_sents)
# print(test_sents)

trainerNLP = pycrfsuite.Trainer(verbose=False)
for xseq, yseq in zip(X_train, y_train):
    trainerNLP.append(xseq, yseq)
trainerNER = pycrfsuite.Trainer(verbose=False)
for xseq, zseq in zip(X_train, z_train):
    trainerNER.append(xseq, zseq)

trainerNLP.set_params({
    'c1': 1.0,   # coefficient for L1 penalty
    'c2': 1e-3,  # coefficient for L2 penalty
    'max_iterations': 50,  # stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
})

trainerNER.set_params({
    'c1': 1.0,   # coefficient for L1 penalty
    'c2': 1e-3,  # coefficient for L2 penalty
    'max_iterations': 50,  # stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
})

# print(trainer.params())

trainerNER.train('./model/engNER.model')
trainerNLP.train('./model/engNLP.model')

# print(trainer.logparser.last_iteration)

# tagger = pycrfsuite.Tagger()
# tagger.open('engNER.model')

# example_sent = test_sents[1]
# print(example_sent)
# print(' '.join(sent2tokens(example_sent)), end='\n\n')

# print("Predicted:", ' '.join(tagger.tag(sent2features(example_sent))))
# print("Correct:  ", ' '.join(sent2labels(example_sent)))