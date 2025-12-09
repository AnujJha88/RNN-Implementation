from data import  *
def make_vocab():
    vocab = set()

    for sentence in train_data.keys():
        for word in sentence.split():
            vocab.add(word)

    vocab=list(vocab)
    out={}
    for index,item in enumerate(vocab):
        out[item]=index
    with open('vocab.py','a') as f:
        f.write('vocab='+str(out))
    return out
if __name__=="__main__":
    make_vocab()