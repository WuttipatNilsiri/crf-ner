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
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

class chucker():
    
    def word2IOB(self,word):
        text = nltk.word_tokenize(word)
        x = nltk.pos_tag(text)
        modified = [ele + ('X',) for ele in x]
        return modified
        

        
c = chucker()

result = c.word2IOB("Wuttipat plan to study at Tokyo Tech University in Japan")
# print(result)

tagger = pycrfsuite.Tagger()
tagger.open('./model/engNER.model')


#############################



def merge(result,predicted):
    listres = []
    for i in range(len(result)):
        listx = list(result[i])
        listx[2] = predicted[i]
        listres.append(tuple(listx))
    return listres






#############################

example_sent = result
# print(example_sent)
print(' '.join(sent2tokens(example_sent)), end='\n\n')

predicted = tagger.tag(sent2features(example_sent))


# print(predicted)

# listres = []
# for i in range(len(result)):
#     listx = list(result[i])
#     listx[2] = predicted[i]
#     listres.append(tuple(listx))
learned = merge(result,predicted)
# print(merge(result,predicted))






print("Predicted:", ' '.join( predicted ))

nltk.chunk.conlltags2tree(learned).draw()
# print("Correct:  ", ' '.join(sent2labels(learned)))
# x = nltk.pos_tag(text)
# print(x)