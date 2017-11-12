import nltk
import csv 
from nltk.corpus import stopwords

def tokenize(text):
    pattern=r'[a-zA-Z]+[a-zA-Z\-\.]*' 
    tokens=nltk.regexp_tokenize(text, pattern)
    return tokens

def sentiment_analysis(text):
    tokens = tokenize(text)
    stop_words = stopwords.words('english')
    filtered_tokens={word.lower() for word in tokens if word not in stop_words}
    with open("positive-words.txt",'r') as f:
        positive_words=[line.strip() for line in f]
    with open("negative-words.txt",'r') as f:
        negative_words=[line.strip() for line in f]
    positive_tokens=[pos_token for pos_token in filtered_tokens if pos_token in positive_words]
    count_positive=len(positive_tokens)
    negative_tokens=[neg_token for neg_token in filtered_tokens if neg_token in negative_words]
    count_negative=len(negative_tokens)
    #print count_positive,count_negative
    if count_positive > count_negative:
        return 'positive'
    elif count_positive < count_negative:
        return 'negative'
    else:
        return 'neutral'

    
def evaluate_accuracy(input_file):
    correct = 0
    with open(input_file, "r") as f:
        reader=csv.reader(f, delimiter=',') 
        rows=[(row[0], row[1]) for row in reader]      
    for row in rows:
        #print row[1],sentiment_analysis(row[0])
        if row[1] == sentiment_analysis(row[0]):
            correct = correct + 1

    return 1.000 * correct/len(rows) 
        
#with open('movie_reivew.txt', 'r') as f:
#    lines=f.readlines()
    
#text=" ".join(lines)
#print sentiment_analysis(text)
#input_file = "finding_dory_reivew.csv"
#evaluate_accuracy(input_file)

if __name__ == "__main__":
    # question 2
    with open('movie_reivew.txt', 'r') as f:
       lines=f.readlines()
    text=" ".join(lines)
    print sentiment_analysis(text)
    # question 3 (bonus)
    input_file = "finding_dory_reivew.csv"
    print evaluate_accuracy(input_file)
