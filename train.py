from vocab import make_vocab,get_idx
from data import train_data
from lstm import LSTM
from gru import GRU
from rnn import RNN
from embedding import EmbedLayer
from loss import CrossEntropyLoss,MSELoss,BCELoss
import numpy as np
import time
from activation import Sigmoid

vocab=make_vocab()
loss=BCELoss()
def preprocess():
    training_data=[]
    for sentence in train_data.keys():
        words=sentence.split()
        indexing=[get_idx(vocab,word) for word in words]
        training_data.append((indexing,int(train_data[sentence])))
    split_idx = int(len(training_data) * 0.8)
    train_split = training_data[:split_idx]
    val_split = training_data[split_idx:]
    
    return train_split, val_split
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

        h,c,y_logit,gates=model.forward(x,h_prev,c_prev)

        caches.append((x,h_prev,c_prev,h,c,y_logit,gates))
    
    y_prob=Sigmoid.forward(y_logit)
    dy=loss.backward(np.array([target]),y_prob)
    dh_next=np.zeros_like(h)
    dc_next = np.zeros_like(c) if c is not None else None
    for idx, cache in zip(reversed(indices), reversed(caches)):
        x, h_prev, c_prev, h, c, y_stored, gates = cache
        
        # Handle unpacking safely for different models
        if isinstance(model, LSTM):
            f, i, o, c_tilde = gates
            dx, dh_prev, dc_prev = model.backward(dh_next, dy, h_prev, c_prev, x, h, c, dc_next, o, i, c_tilde, f)
        elif isinstance(model, GRU):
            # GRU returns (z, r, h_tilde)
            z, r, h_tilde = gates
            dx, dh_prev = model.backward(dh_next, dy, h, x, h_prev, z, r, h_tilde)
            dc_prev = None
        else: # RNN
            # RNN returns empty gates
            dx, dh_prev, _ = model.backward(dh_next, dy, h_prev, None, x, h, None, None)
            dc_prev = None

        embed_layer.backward(dx, [idx])
        dy=np.array([[0]])
        dh_next=dh_prev
        dc_next=dc_prev
    return h,c,y_prob,caches


def training(epochs):
    training_data, validation_data = preprocess()
    model1=LSTM(input_size=50,hidden_size=100,output_size=1)
    model2=GRU(input_size=50,hidden_size=100,output_size=1)
    model3=RNN(input_size=50,hidden_size=100,output_size=1)
    embed_layer=EmbedLayer(vocab_size=len(vocab),embed_dim=50)
    for epoch in range(epochs):
        start_time = time.time()
        
        epoch_loss = 0
        correct_predictions = 0
        total_examples = len(training_data)

        for indices, label in training_data:
            # 1. Train on the sentence
            h, c, y, caches = train_sentence(indices, model1, embed_layer, label)
            
            # 2. Update weights
            model1.step(learning_rate=0.1)
            embed_layer.step(learning_rate=0.1)

            # 3. Calculate Stats for this sentence
            # Calculate loss (scalar)
            current_loss = loss.forward( np.array([label]),y)
            epoch_loss += current_loss

            # Calculate accuracy (threshold at 0.5)
            prediction = 1 if y >= 0.5 else 0
            if prediction == label:
                correct_predictions += 1

        # 4. End of Epoch Stats
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        avg_loss = epoch_loss / total_examples
        accuracy = (correct_predictions / total_examples) * 100

        if (epoch + 1) % 10 == 0:
                print(f'Epoch: {epoch+1:02} | Time: {int(epoch_mins)}m {int(epoch_secs)}s')
                print(f'\tTrain Loss: {avg_loss:.4f} | Train Acc: {accuracy:.2f}%')
                time.sleep(0.1)
        val_loss = 0
        val_correct = 0
        for indices, label in validation_data:
            # Forward pass ONLY (No backward, no step)
            h, c, y, _ = train_sentence(indices, model1, embed_layer, label)
            
            val_loss += loss.forward(np.array([label]), y)
            if (1 if y >= 0.5 else 0) == label:
                val_correct += 1
                
        # Calculate Val Stats
        avg_val_loss = val_loss / len(validation_data)
        val_acc = (val_correct / len(validation_data)) * 100
        
        print(f"Epoch {epoch}: Train Loss {avg_loss:.4f} | Val Loss {avg_val_loss:.4f} | Val Acc {val_acc:.2f}%")
     
def main():
    training(epochs=100)

if __name__=="__main__":
    main()