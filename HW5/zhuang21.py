import nltk, re, string
from sklearn.preprocessing import normalize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
# numpy is the package for matrix cacluation
import numpy as np  
import csv 
from scipy.spatial import distance

# Step 1. get tokens of each document as list
def get_doc_tokens(doc, lemmatized=False):
    stop_words = stopwords.words('english')
    tokens=[token.strip() \
            for token in nltk.word_tokenize(doc.lower()) \
            if token.strip() not in stop_words and\
               token.strip() not in string.punctuation]
    
    #print tokens
    # you can add bigrams, collocations, or lemmatization here
    if(lemmatized==True):
        wordnet_lemmatizer = WordNetLemmatizer()
        tagged_tokens= nltk.pos_tag(tokens)
        le_words=[wordnet_lemmatizer.lemmatize(word, get_wordnet_pos(tag)) \
            # tagged_tokens is a list of tuples (word, tag)
            for (word, tag) in tagged_tokens \
            # remove stop words
            if word not in stop_words and \
            # remove punctuations
            word not in string.punctuation]
        tokens = le_words
    return tokens

def tfidf(docs):
    # step 2. process all documents to get list of token list
    docs_tokens=[get_doc_tokens(doc, False) for doc in docs]
    voc=list(set([token for tokens in docs_tokens \
              for token in tokens]))

    # step 3. get document-term matrix
    dtm=np.zeros((len(docs), len(voc)))

    for row_index,tokens in enumerate(docs_tokens):
        for token in tokens:
            col_index=voc.index(token)
            dtm[row_index, col_index]+=1
            
    # step 4. get normalized term frequency (tf) matrix        
    doc_len=dtm.sum(axis=1, keepdims=True)
    tf=np.divide(dtm, doc_len)
    
    # step 5. get idf
    doc_freq=np.copy(dtm)
    doc_freq[np.where(doc_freq>0)]=1

    smoothed_idf=np.log(np.divide(len(docs)+1, np.sum(doc_freq, axis=0)+1))+1

    
    # step 6. get tf-idf
    smoothed_tf_idf=normalize(tf*smoothed_idf)
    
    return smoothed_tf_idf


def top10_similarity(tf_idf):
    similarity=1-distance.squareform(distance.pdist(tf_idf, 'cosine'))
    doc_1_sim=similarity[0].tolist()
    return sorted(enumerate(doc_1_sim), key=lambda item:-item[1])[1:11]


def get_wordnet_pos(pos_tag):
    
    # if pos tag starts with 'J'
    if pos_tag.startswith('J'):
        # return wordnet tag "ADJ"
        return wordnet.ADJ
    
    # if pos tag starts with 'V'
    elif pos_tag.startswith('V'):
        # return wordnet tag "VERB"
        return wordnet.VERB
    
    # if pos tag starts with 'N'
    elif pos_tag.startswith('N'):
        # return wordnet tag "NOUN"
        return wordnet.NOUN
    
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        # be default, return wordnet tag "NOUN"
        return wordnet.NOUN

if __name__ == "__main__":
    
    with open("amazon_review_300.csv", "r") as f:
            reader=csv.reader(f, delimiter=',') 
            docs=[row[2] for row in reader]

    smoothed_tf_idf = tfidf(docs);
    
    print("\nSmoothed TF-IDF Matrix")
    print smoothed_tf_idf
    print("\ntop10 similarity of the first review")
    print top10_similarity(smoothed_tf_idf)


        
        

        

