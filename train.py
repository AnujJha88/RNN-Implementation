from vocab import make_vocab
from data import train_data
from lstm import LSTM
from gru import GRU
from rnn import RNN
from embedding import EmbedLayer
from loss import CrossEntropyLoss,MSELoss,BCELoss
import numpy as np

vocab=make_vocab()
loss=BCELoss()
def preprocess():
    training_data=[]
    for sentence in train_data.keys():
        words=sentence.split()
        indexing=[vocab[word] for word in words]
        training_data.append((indexing,int(train_data[sentence])))
    return training_data

def train_sentence(indices,model,embed_layer,target):
    model.zero_grad()
    embed_layer.zero_grad()
    h=np.zeros((1,model.hidden_size))
    c=np.zeros((1,model.hidden_size))
    caches=[]
    
    for idx in indices:
        x=embed_layer.forward(np.array([idx]))
        h_prev=h
        c_prev=c

        h,c,y,gates=model.forward(x,h_prev,c_prev)

        caches.append((x,h_prev,c_prev,h,c,y,gates))
    
    dy=loss.backward(y,np.array([target]))
    dh_next=np.zeros_like(h)
    dc_next=np.zeros_like(c)
    for idx,cache in zip(reversed(indices),reversed(caches)):
        x,h_prev,c_prev,h,c,y,gates=cache
        f,i,o,c_tilde=gates
        dx,dh_prev,dc_prev=model.backward(dh_next,dy,h_prev,c_prev,x,h,c,dc_next,o,i,c_tilde,f)
        embed_layer.backward(dx,[idx])
        dy=np.array([[0]])
        dh_next=dh_prev
        dc_next=dc_prev
    return h,c,y,caches


def training(epochs):
    training_data=preprocess()
    model1=LSTM(input_size=50,hidden_size=100,output_size=1)
    model2=GRU(input_size=50,hidden_size=100,output_size=1)
    model3=RNN(input_size=50,hidden_size=100,output_size=1)
    embed_layer=EmbedLayer(vocab_size=len(vocab),embed_dim=50)
    for indices,label in training_data:
        train_sentence(indices,model1,embed_layer,label)
        model1.step(learning_rate=0.01)
        embed_layer.step(learning_rate=0.01)
        
     
def main():
    training(epochs=100000)

if __name__=="__main__":
    main()