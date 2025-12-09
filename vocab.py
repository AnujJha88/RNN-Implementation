from collections import Counter
from data import train_data

def make_vocab(max_size=2000):
    print("Building Vocabulary...")
    word_counts = Counter()
    
    for sentence in train_data.keys():
        for word in sentence.split():
            word_counts[word] += 1
            
    # Keep only most common words to keep matrix size manageable
    most_common = word_counts.most_common(max_size)
    vocab_list = [word for word, count in most_common]
    
    # Add <UNK> token for unknown words
    vocab_list.append("<UNK>")
    
    vocab = {word: i for i, word in enumerate(vocab_list)}
    
    print(f"Vocab built with {len(vocab)} words.")
    return vocab

# Helper to safely get index
def get_idx(vocab, word):
    return vocab.get(word, vocab["<UNK>"])