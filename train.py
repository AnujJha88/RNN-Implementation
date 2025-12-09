from vocab import make_vocab
from data import train_data

vocab=make_vocab()

def preprocess():
    training_data=[]
    for sentence in train_data.keys():
        words=sentence.split()
        indexing=[vocab[word] for word in words]
        training_data.append((indexing,int(train_data[sentence])))
    return training_data

