# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py
import numpy as np
import random
import torch
from datasets import load_dataset, load_from_disk 

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets - use HuggingFace directly
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    
    # Filter out empty texts
    train_texts = [text for text in traindata['text'] if text.strip()]
    
    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    
    for _ in range(nsamples):
        # Keep trying until we find a text long enough
        attempts = 0
        max_attempts = len(train_texts)
        
        while attempts < max_attempts:
            # Pick a random text
            text_idx = random.randint(0, len(train_texts) - 1)
            text = train_texts[text_idx]
            
            # Tokenize this specific text
            trainenc = tokenizer(text, return_tensors='pt', truncation=False)
            
            # Check if this text is long enough for our sequence length
            if trainenc.input_ids.shape[1] > seqlen:
                # Extract a random subsequence of length seqlen
                max_start = trainenc.input_ids.shape[1] - seqlen - 1
                i = random.randint(0, max_start)
                j = i + seqlen
                inp = trainenc.input_ids[:, i:j]
                tar = inp.clone()
                tar[:, :-1] = -100
                trainloader.append((inp, tar))
                break
            
            attempts += 1
        
        # If we couldn't find a long enough text, create a shorter one and pad
        if attempts >= max_attempts:
            print(f"Warning: Could not find text longer than {seqlen} tokens, using shorter text")
            # Use the last text we tried and pad if necessary
            text = train_texts[0]  # Use first non-empty text
            trainenc = tokenizer(text, return_tensors='pt', truncation=False)
            
            if trainenc.input_ids.shape[1] < seqlen:
                # Pad to required length
                padding_length = seqlen - trainenc.input_ids.shape[1]
                padding = torch.zeros(1, padding_length, dtype=trainenc.input_ids.dtype, device=trainenc.input_ids.device)
                inp = torch.cat([trainenc.input_ids, padding], dim=1)
            else:
                inp = trainenc.input_ids[:, :seqlen]
            
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
    
    # For test data, create a reasonable-sized test set
    test_texts = [text for text in testdata['text'] if text.strip()]
    # Take first few texts and limit total length
    limited_test_texts = test_texts[:100]  # Limit to first 100 non-empty texts
    testenc = tokenizer("\n\n".join(limited_test_texts), return_tensors='pt', max_length=seqlen*8, truncation=True)
    
    return trainloader, testenc

# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name or 'wikitext' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    else:
        raise ValueError(f"Unknown dataset name: {name}")
