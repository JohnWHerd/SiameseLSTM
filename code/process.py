import re
import nltk
import pickle
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from ast import literal_eval

def clean_sent(sent):
    sent = sent.lower()
    sent = re.sub(u'[_"\-;%()|+&=*%.,!?:#$@\[\]/]',' ',sent)
    sent = re.sub('¡',' ',sent)
    sent = re.sub('¿',' ',sent)
    sent = re.sub('Á','á',sent)
    sent = re.sub('Ó','ó',sent)
    sent = re.sub('Ú','ú',sent)
    sent = re.sub('É','é',sent)
    sent = re.sub('Í','í',sent)
    return sent

def data_process(train_number, test_number):

    df_sent_pairs = pd.read_csv('questions.csv')

    df_sent_pairs.drop(columns=['id', 'qid1', 'qid2'])

    for p in df_sent_pairs.index:
        if type(df_sent_pairs['question1'][p]) == str:
            df_sent_pairs.at[p, 'question1'] = clean_sent(df_sent_pairs['question1'][p])
        else:
            df_sent_pairs.drop([p])
        if type(df_sent_pairs['question2'][p]) == str:    
            df_sent_pairs.at[p, 'question2'] = clean_sent(df_sent_pairs['question2'][p])
        else:
            df_sent_pairs.drop([p])
            
    df_sent_pairs.dropna()
    
    df = df_sent_pairs[['question1', 'question2', 'is_duplicate']]
    
    training_set = df.sample(frac=0.8)
    test_set = df.drop(training_set.index)
    
    if train_number < training_set.shape[0]:
        ### Smaller set size for testing
        tr_set = training_set[:train_number]
        t_set = test_set[:test_number]
    else: 
        tr_set = training_set[:30000]
        t_set = test_set[:3000]
        
        
    training_set.to_csv("cleaned_train.csv", index=False)
    test_set.to_csv("cleaned_test.csv", index=False)
    
    tr_set.to_csv("ctr.csv", index=False)
    t_set.to_csv("ct.csv", index=False)


def save_embed(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    print ('Embedding saved')

def load_embed(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_embedding(word_dict, embedding_path, embedding_dim=300):
    # find existing word embeddings
    word_vec = {}
    with open(embedding_path) as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word_dict:
                word_vec[word] = np.array(list(map(float, vec.split())))
    print('Found {0}/{1} words with embedding vectors'.format(
        len(word_vec), len(word_dict)))
    missing_word_num = len(word_dict) - len(word_vec)
    missing_ratio = round(float(missing_word_num) / len(word_dict), 4) * 100
    print('Missing Ratio: {}%'.format(missing_ratio))

    # handling unknown embeddings
    for word in word_dict:
        if word not in word_vec:
            # If word not in word_vec, create a random embedding for it
            new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
            word_vec[word] = new_embedding
    print ("Filled missing words' embeddings.")
    print ("Embedding Matrix Size: ", len(word_vec))
    
    return word_vec