from string import punctuation
from gensim.models.doc2vec import TaggedDocument
import json
import gensim
import nltk,string
from random import shuffle
from gensim.models import doc2vec
from sklearn.preprocessing import MultiLabelBinarizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.layers import Embedding, Dense, Conv1D, MaxPooling1D, \
Dropout, Activation, Input, Flatten, Concatenate
from keras.regularizers import l2
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from nltk import tokenize
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
import os
import matplotlib  
matplotlib.use('Agg') 
from matplotlib.pyplot import plot,savefig 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn import metrics
from keras.models import load_model

DOCVECTOR_MODEL="docvector_model"
BEST_MODEL_FILEPATH="best_model"
BEST_LABEL_WEIGHT_FILEPATH="best_label_weight"
BEST_SENT_MODEL_FILEPATH="best_sent_model"
BEST_SENT_WEIGHT_FILEPATH="best_sent_weight"
QUALITY_MODEL="quality_model"
MAX_NB_WORDS=1000
MAX_DOC_LEN=200
EMBEDDING_DIM=200
FILTER_SIZES=[2,3,4]
BTACH_SIZE = 64
NUM_EPOCHES = 20

class ReviewAnalyser(object):
    
    # review's ann model
    ann_model = None
    # label's cnn model
    label_model = None
    # label's classification: ['amenities' 'environment' 'food' 'location' 'null' 'price' 'service' 'transport']
    label_mlb = None
    # labels input padding sequence
    label_padding_sequence = None
    # labels actual classification
    label_act = None
    # labels test set feature
    label_X_test = None
    # labels test set labels
    label_Y_set = None
    # labels validation set feature
    label_X_train = None
    # labels validation set labels
    label_Y_train = None
    # sentiment's cnn model
    sent_model = None
    # sentiment's classification: ['0', '1'] 0: neutral, 1: positive/negative
    sent_mlb = None
    # sentiment input padding sequence
    sent_padding_sequence = None
    # sentiment actual classification
    sent_act = None
    # labels test set
    sent_test_set = None
    # sentiment's validation set
    sent_validation_set = None
    # sentitment's test set feature
    sent_X_test = None
    # sentitment's test set labels
    sent_Y_set = None
    # sentitment's validation set feature
    sent_X_train = None
    # sentitment's validation set labels
    sent_Y_train = None
    # doc2vector's cnn model
    wv_model = None
    
    def __init__(self, data): 
        self.data = data;
        
    @staticmethod
    def ann_model():
        lam=0.01
        model = Sequential()
        model.add(Dense(12, input_dim=8, activation='relu', \
                        kernel_regularizer=l2(lam), name='L2') )
        model.add(Dense(8, activation='relu', \
                        kernel_regularizer=l2(lam),name='L3') )
        model.add(Dense(1, activation='sigmoid', name='Output'))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        return model
        
    @staticmethod    
    def cnn_model(FILTER_SIZES, \
        # filter sizes as a list
        MAX_NB_WORDS, \
        # total number of words
        MAX_DOC_LEN, \
        # max words in a doc
        NUM_OUTPUT_UNITS=1, \
        # number of output units
        EMBEDDING_DIM=200, \
        # word vector dimension
        NUM_FILTERS=64, \
        # number of filters for all size
        DROP_OUT=0.5, \
        # dropout rate
        PRETRAINED_WORD_VECTOR=None,\
        # Whether to use pretrained word vectors
        LAM=0.01,\
        ACTIVATION='sigmoid'):            
        # regularization coefficient
    
        main_input = Input(shape=(MAX_DOC_LEN,), \
                           dtype='int32', name='main_input')

        if PRETRAINED_WORD_VECTOR is not None:
            embed_1 = Embedding(input_dim=MAX_NB_WORDS+1, \
                            output_dim=EMBEDDING_DIM, \
                            input_length=MAX_DOC_LEN, \
                            weights=[PRETRAINED_WORD_VECTOR],\
                            trainable=False,\
                            name='embedding')(main_input)
        else:
            embed_1 = Embedding(input_dim=MAX_NB_WORDS+1, \
                            output_dim=EMBEDDING_DIM, \
                            input_length=MAX_DOC_LEN, \
                            name='embedding')(main_input)

        conv_blocks = []
        for f in FILTER_SIZES:
            conv = Conv1D(filters=NUM_FILTERS, kernel_size=f, \
                          activation='relu', name='conv_'+str(f))(embed_1)
            conv = MaxPooling1D(MAX_DOC_LEN-f+1, name='max_'+str(f))(conv)
            conv = Flatten(name='flat_'+str(f))(conv)
            conv_blocks.append(conv)

        z=Concatenate(name='concate')(conv_blocks)
        drop=Dropout(rate=DROP_OUT, name='dropout')(z)

        dense = Dense(192, activation='relu',\
                        kernel_regularizer=l2(LAM),name='dense')(drop)
        preds = Dense(NUM_OUTPUT_UNITS, activation=ACTIVATION, name='output')(dense)
        model = Model(inputs=main_input, outputs=preds)

        model.compile(loss="binary_crossentropy", \
                  optimizer="adam", metrics=["accuracy"]) 

        return model

    # training to change document into vector using gensim
    def pretrain(self, RETRAIN=0):
        with open("word_sample.json", 'r') as f:
            reviews=[]
            for line in f: 
                review = json.loads(line) 
                try:
                    review["text"].strip().lower().encode('ascII')
                except:
                    # do nothing
                    a = 1
                else:
                    reviews.append(review["text"])

        sentences=[ [token.strip(string.punctuation).strip() \
                     for token in nltk.word_tokenize(doc.lower()) \
                         if token not in string.punctuation and \
                         len(token.strip(string.punctuation).strip())>=2]\
                     for doc in reviews]


        docs=[TaggedDocument(sentences[i], [str(i)]) for i in range(len(sentences)) ]
        
        if RETRAIN==0 and os.path.exists(DOCVECTOR_MODEL):
            self.wv_model = doc2vec.Doc2Vec.load(DOCVECTOR_MODEL)
        else:
            self.wv_model = doc2vec.Doc2Vec(dm=1, min_count=5, window=5, size=200, workers=4)
            self.wv_model.build_vocab(docs)
            for epoch in range(30):
                # shuffle the documents in each epoch
                shuffle(docs)
                # in each epoch, all samples are used
                self.wv_model.train(docs, total_examples=len(docs), epochs=1)
                
            self.wv_model.save(DOCVECTOR_MODEL)

#         print("Top 5 words similar to word 'price'")
#         print self.wv_model.wv.most_similar('price', topn=5)

#         print("Top 5 words similar to word 'price' but not relevant to 'bathroom'")
#         print self.wv_model.wv.most_similar(positive=['price','money'], negative=['bathroom'], topn=5)

#         print("Similarity between 'price' and 'bathroom':")
#         print self.wv_model.wv.similarity('price','bathroom') 

#         print("Similarity between 'price' and 'charge':")
#         print self.wv_model.wv.similarity('price','charge') 

#         print self.wv_model.wv

    # training labels CNN
    def trainLebels(self, RETRAIN=0):
        labels = []
        # fetch labels for each sentence        
        for subdata in self.data[2][0:500]:
            label = []
            for d in subdata.split(","):
                label.append(d.strip())
            labels.append(label)
            
        mlb = MultiLabelBinarizer()
        Y=mlb.fit_transform(labels)
        self.label_act = Y
        self.label_mlb = mlb
        np.sum(Y, axis=0)

        tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        tokenizer.fit_on_texts(self.data[1][0:500])
        NUM_WORDS = min(MAX_NB_WORDS, len(tokenizer.word_index))
        embedding_matrix = np.zeros((NUM_WORDS+1, EMBEDDING_DIM))

        for word, i in tokenizer.word_index.items():
            if i >= NUM_WORDS:
                continue
            if word in self.wv_model.wv:
                embedding_matrix[i]=self.wv_model.wv[word]

        voc=tokenizer.word_index
        sequences = tokenizer.texts_to_sequences(self.data[1][0:500])
        padded_sequences = pad_sequences(sequences, \
                                         maxlen=MAX_DOC_LEN, \
                                         padding='post', truncating='post')
        self.label_padding_sequence = padded_sequences
        
        
        NUM_OUTPUT_UNITS=len(mlb.classes_)

        X_train, X_test, Y_train, Y_test = train_test_split(\
                        padded_sequences[0:500], Y[0:500], test_size=0.3, random_state=0)
        
        self.label_X_train = X_train
        self.label_Y_train = Y_train
        self.label_X_test = X_test
        self.label_Y_test = Y_test
        
        if(RETRAIN == 0 and os.path.exists(BEST_MODEL_FILEPATH)):
#                 self.label_model.load_weights(BEST_MODEL_FILEPATH)
                self.label_model = load_model(BEST_MODEL_FILEPATH)
#                 pred=self.label_model.predict(padded_sequences[0:500])
                return
        
        self.label_model=ReviewAnalyser.cnn_model(FILTER_SIZES, MAX_NB_WORDS, \
                        MAX_DOC_LEN, NUM_OUTPUT_UNITS, \
                        PRETRAINED_WORD_VECTOR=embedding_matrix)

        earlyStopping=EarlyStopping(monitor='val_loss', patience=0, verbose=2, mode='min')
        checkpoint = ModelCheckpoint(BEST_LABEL_WEIGHT_FILEPATH, monitor='val_acc', \
                                     verbose=2, save_best_only=True, mode='max')
        
        training=self.label_model.fit(X_train, Y_train, \
                  batch_size=BTACH_SIZE, epochs=NUM_EPOCHES, \
                  callbacks=[earlyStopping, checkpoint],\
                  validation_data=[X_test, Y_test], verbose=2)
        
        self.label_model.save(BEST_MODEL_FILEPATH)
        
        return
        
    # training sentiment CNN        
    def trainSentiment(self, RETRAIN=0):
        labels = []
        for i,subdata in enumerate(self.data[3][0:500]):
            if subdata == 1:
                labels.append(['1'])
            else:
                labels.append(['0'])

        Y_labels = np.copy(labels)
        mlb = LabelBinarizer()
        Y = mlb.fit_transform(Y_labels)
        self.sent_act = Y
        self.sent_mlb = mlb
        
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        tokenizer.fit_on_texts(self.data[1][0:500])
        NUM_WORDS = min(MAX_NB_WORDS, len(tokenizer.word_index))
        embedding_matrix = np.zeros((NUM_WORDS+1, EMBEDDING_DIM))

        for word, i in tokenizer.word_index.items():
            if i >= NUM_WORDS:
                continue
            if word in self.wv_model.wv:
                embedding_matrix[i]=self.wv_model.wv[word]

        voc=tokenizer.word_index
        sequences = tokenizer.texts_to_sequences(self.data[1][0:500])
        padded_sequences = pad_sequences(sequences, \
                                         maxlen=MAX_DOC_LEN, \
                                         padding='post', truncating='post')
        self.sent_padding_sequence = padded_sequences

        NUM_OUTPUT_UNITS=len(mlb.classes_)

        X_train, X_test, Y_train, Y_test = train_test_split(padded_sequences[0:500], Y[0:500], test_size=0.3, random_state=0)
        self.sent_X_train = X_train
        self.sent_X_test = X_test
        self.sent_Y_train = Y_train
        self.sent_Y_test = Y_test
        
        if(RETRAIN == 0 and os.path.exists(BEST_SENT_MODEL_FILEPATH)):
#                 self.sent_model.load_weights("best_sent_model")
                self.sent_model = load_model(BEST_SENT_MODEL_FILEPATH)
                pred=self.sent_model.predict(padded_sequences[0:500])
                return
        
        
        self.sent_model=ReviewAnalyser.cnn_model(FILTER_SIZES, MAX_NB_WORDS, \
                    MAX_DOC_LEN, \
                    PRETRAINED_WORD_VECTOR=embedding_matrix)

        earlyStopping=EarlyStopping(monitor='val_loss', patience=0, verbose=2, mode='min')
        checkpoint = ModelCheckpoint(BEST_SENT_WEIGHT_FILEPATH, monitor='val_acc', \
                                     verbose=2, save_best_only=True, mode='max')

        training=self.sent_model.fit(X_train, Y_train, \
                  batch_size=BTACH_SIZE, epochs=NUM_EPOCHES, \
                  callbacks=[earlyStopping, checkpoint],\
                  validation_data=[X_test, Y_test], verbose=2) 
        
        self.sent_model.save(BEST_SENT_MODEL_FILEPATH)
        
        return
    
    # training review quality ANN
    def trainQuality(self, RETRAIN=0, PERFORMANCE=0):
        rows = {}
        for subdata in self.data[0:192].values.tolist():
            if rows.has_key(subdata[0]):
                labels = subdata[2].split(',')
                for label in labels:
                    rows[subdata[0]][label.strip()] = rows[subdata[0]][label.strip()]+1.0
                rows[subdata[0]]["sentiment"] = rows[subdata[0]]["sentiment"] + subdata[3]
                rows[subdata[0]]["quality"] = subdata[4]
                rows[subdata[0]]["items"] = rows[subdata[0]]["items"] + 1
            else:
                rows[subdata[0]] = {
                    'items' : 0.0,
                    'amenities' : 0.0,
                    'environment' : 0.0,
                    'food' : 0.0,
                    'location' : 0.0,
                    'null' : 0.0,
                    'price': 0.0,
                    'service': 0.0,
                    'sentiment': 0.0,
                    'quality': 0.0
                }
        data = []
        for key in rows:
            subdata=[]
            subdata.append(rows[key]["amenities"]/rows[key]["items"])
            subdata.append(rows[key]["environment"]/rows[key]["items"])
            subdata.append(rows[key]["food"]/rows[key]["items"])
            subdata.append(rows[key]["location"]/rows[key]["items"])
            subdata.append(rows[key]["null"]/rows[key]["items"])
            subdata.append(rows[key]["price"]/rows[key]["items"])
            subdata.append(rows[key]["service"]/rows[key]["items"])
            subdata.append(rows[key]["sentiment"]/rows[key]["items"])
            subdata.append(rows[key]["quality"])
            data.append(subdata)

        df=pd.DataFrame(data, columns=["amenities","environment","food","location","null","price","service","sentiment","quality"])
        X=df.values[:,0:8]
        Y=df.values[:,8]

        if RETRAIN == 0 and os.path.exists(QUALITY_MODEL):
            self.ann_model = load_model(QUALITY_MODEL)
        else:
            self.ann_model = ReviewAnalyser.ann_model()
            training=self.ann_model.fit(X, Y, validation_split=0.3, shuffle=True, epochs=150, batch_size=32, verbose=2)
            self.ann_model.save(QUALITY_MODEL)
        
        if PERFORMANCE==1:
            scores = self.ann_model.evaluate(X, Y)
            print("\n%s: %.2f%%" % (self.ann_model.metrics_names[1], scores[1]*100))

            predicted=self.ann_model.predict(X)
            predicted=np.reshape(predicted, -1)
            predicted = np.round(predicted, decimals=1)
#             predicted=np.where(predicted>0.5, 1, 0)
            print("mean_squared_error:")
            print(mean_squared_error(Y, predicted))
#             print(metrics.classification_report(Y, predicted, labels=[0,1]))
        
        
        return 
    

    @staticmethod
    def checkPerform(model, mlb, data_tobe_predicted, Y_actual):
        pred=model.predict(data_tobe_predicted)
        Y_pred=np.copy(pred)
        Y_pred=np.where(Y_pred>0.5,1,0)
        print(classification_report(Y_actual, Y_pred, \
                                    target_names=mlb.classes_))
        return classification_report(Y_actual, Y_pred, \
                                    target_names=mlb.classes_)
    
    def checkLabelPerform(self):
        pred=self.label_model.predict(self.label_X_test)
        Y_pred=np.copy(pred)
        Y_pred=np.where(Y_pred>0.5,1,0)
        Y_actual = self.label_Y_test
        print(classification_report(Y_actual, Y_pred, \
                                    target_names=self.label_mlb.classes_))
        return classification_report(Y_actual, Y_pred, \
                                    target_names=self.label_mlb.classes_)
        

    def checkSentimentPerform(self):
        pred=self.sent_model.predict(self.sent_X_test)
        Y_pred=np.copy(pred)
        Y_pred=np.where(Y_pred>0.5,1,0)
        Y_actual = self.sent_Y_test
        print(classification_report(Y_actual, Y_pred, \
                                    target_names=self.sent_mlb.classes_))
        return classification_report(Y_actual, Y_pred, \
                                    target_names=self.sent_mlb.classes_)
        
       
    # check document information to determine the value of hyper-parameter
    def checkDocInform(self):  
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        tokenizer.fit_on_texts(self.data[1][0:500])
        total_nb_words=len(tokenizer.word_counts)
        sequences = tokenizer.texts_to_sequences(self.data[1][0:500])
        print "\n############## document information ##############\n"
        print "total_nb_words:"
        print(total_nb_words)

        word_counts=pd.DataFrame(\
                    tokenizer.word_counts.items(), \
                    columns=['word','count'])
        df=word_counts['count'].value_counts().reset_index()
        df['percent']=df['count']/len(tokenizer.word_counts)
        df['cumsum']=df['percent'].cumsum()

        plt.bar(df["index"].iloc[0:50], df["percent"].iloc[0:50])
        plt.plot(df["index"].iloc[0:50], df['cumsum'].iloc[0:50], c='green')

        plt.xlabel('Word Frequency')
        plt.ylabel('Percentage')
        savefig('1.jpg')
        plt.show()
        
        sen_len=pd.Series([len(item) for item in sequences])

        df=sen_len.value_counts().reset_index().sort_values(by='index')
        df.columns=['index','counts']

        df=df.sort_values(by='index')
        df['percent']=df['counts']/len(sen_len)
        df['cumsum']=df['percent'].cumsum()
        
        plt.plot(df["index"], df['cumsum'], c='green')

        plt.xlabel('Sentence Length')
        plt.ylabel('Percentage')
        savefig('2.jpg')
        plt.show()
        
    # predict labels for text, need to execute trainLabels first
    def predictLabels(self, text_arr=[]):
        if len(text_arr)==0:
            return
        rtn = {}
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        tokenizer.fit_on_texts(self.data[1][0:500])
        sub_sequences = tokenizer.texts_to_sequences(text_arr)
        padded_sub_sequences = pad_sequences(sub_sequences, \
                                 maxlen=MAX_DOC_LEN, \
                                 padding='post', truncating='post')
        sub_pred = self.label_model.predict(padded_sub_sequences)
        for i, key in enumerate(text_arr):
            dict1 = {}
            pred_list = sub_pred[i].tolist()
            for i, sub_pred_list in enumerate(pred_list):
                dict1[self.label_mlb.classes_[i]] = pred_list[i]
            rtn[key] = dict1
        return rtn
        
    # predict sentiments for text, need to execute trainSentiment first    
    def predictSentiment(self, text_arr=[]):
        if len(text_arr)==0:
            return
        rtn = {}
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        tokenizer.fit_on_texts(self.data[1][0:500])
        sub_sequences = tokenizer.texts_to_sequences(text_arr)
        padded_sub_sequences = pad_sequences(sub_sequences, \
                                 maxlen=MAX_DOC_LEN, \
                                 padding='post', truncating='post')
        sub_pred = self.sent_model.predict(padded_sub_sequences)
        for i, key in enumerate(text_arr):
            rtn[key] = sub_pred[i].tolist()[0]
        return rtn
    
    # predict quality for reviews, need to execute trainLabels,trainSentiment and trainQuality first    
    def predictQuality(self, review_arr=[]):
        text_arr=[]
        sentence_review_mapping = []
        data = []
        rows = []
        if len(review_arr)==0:
            return
        for i, rev in enumerate(review_arr):
            rows.append({
                'items' : 0.0,
                'amenities' : 0.0,
                'environment' : 0.0,
                'food' : 0.0,
                'location' : 0.0,
                'null' : 0.0,
                'price': 0.0,
                'service': 0.0,
                'sentiment': 0.0
            })
            rev_sent = tokenize.sent_tokenize(rev)
            for sent in rev_sent:
                text_arr.append(sent)
                sentence_review_mapping.append((i,sent))
            
        label_predict = self.predictLabels(text_arr)
        sentiment_predict = self.predictSentiment(text_arr)
#         print sentence_review_mapping
       
        for mapping in sentence_review_mapping:
            rows[mapping[0]]["items"] = rows[mapping[0]]["items"] + 1
            rows[mapping[0]]["amenities"] = rows[mapping[0]]["amenities"]+label_predict[mapping[1]]["amenities"] 
            rows[mapping[0]]["environment"] = rows[mapping[0]]["environment"]+label_predict[mapping[1]]["environment"] 
            rows[mapping[0]]["food"] = rows[mapping[0]]["food"]+label_predict[mapping[1]]["food"]
            rows[mapping[0]]["location"] = rows[mapping[0]]["location"]+label_predict[mapping[1]]["location"]
            rows[mapping[0]]["null"] = rows[mapping[0]]["null"]+label_predict[mapping[1]]["null"]
            rows[mapping[0]]["price"] = rows[mapping[0]]["price"]+label_predict[mapping[1]]["price"]
            rows[mapping[0]]["service"] = rows[mapping[0]]["service"]+label_predict[mapping[1]]["service"]
            rows[mapping[0]]["sentiment"] = rows[mapping[0]]["sentiment"]+sentiment_predict[mapping[1]]
            
        data = []
        for row in rows:
            subdata=[]
            subdata.append(row["amenities"]/row["items"])
            subdata.append(row["environment"]/row["items"])
            subdata.append(row["food"]/row["items"])
            subdata.append(row["location"]/row["items"])
            subdata.append(row["null"]/row["items"])
            subdata.append(row["price"]/row["items"])
            subdata.append(row["service"]/row["items"])
            subdata.append(row["sentiment"]/row["items"])
            data.append(subdata)
        df=pd.DataFrame(data, columns=["amenities","environment","food","location","null","price","service","sentiment"])
        X = df.values[:,0:8]
        
        predicted=self.ann_model.predict(X)
        predicted=np.reshape(predicted, -1)
#         print(predicted)
        #predicted=np.where(predicted>0.5, 1, 0)
        rtn = {
            "label_predict": label_predict,
            "sentiment_predict": sentiment_predict,
            "review_predict": predicted.tolist()
        }
        return rtn
