from string import punctuation
import json
import matplotlib.pyplot as plt

def count_token(text):
    list1 = text.split()
    token_count = {}
    s = ""
    for idx,x in enumerate(list1):
        s = x.strip().lower().strip(punctuation)
        if(s != ''):
            if(s in token_count):
                token_count[s] += 1
            else:
                token_count[s] = 1
    return token_count

def tweets_analysis(input_file="python.txt"):
    tweets=[]
    text=""
    with open(input_file, 'r') as f:
        for line in f: 
            tweet = json.loads(line) 
            tweets.append(tweet)
        count_per_topic={}
        for t in tweets:
            #concatenate each row of the tweets
            text += t["text"].strip()
            if "entities" in t and "hashtags" in t["entities"]:
                topics=set([hashtag["text"].lower() for hashtag in t["entities"]["hashtags"]])
                for topic in topics:
                    topic=topic.lower()
                    if topic in count_per_topic:
                        count_per_topic[topic]+=1
                    else:
                        count_per_topic[topic]=1
        
        topic_count_list=count_per_topic.items()
        #print topic_count_list
        topics, counts=zip(*topic_count_list)
        
        #draw the diagram
        fig_size = plt.rcParams["figure.figsize"]
        fig_size[0] = 40
        fig_size[1] = 6
        plt.rcParams["figure.figsize"] = fig_size

        x_pos = range(len(topics))
        plt.bar(x_pos, counts)
        plt.xticks(x_pos, topics)
        plt.ylabel('Count of Tweets')
        plt.title('Count of Tweets per Topic')
        plt.xticks(rotation=90) 
        plt.show()
        
        words_count_list = count_token(text)
        words_count_list = words_count_list.items()
        sorted_words=sorted(words_count_list, key=lambda item:-item[1])
        #print(sorted_words)

        # get top 20 topics
        top_50_words=sorted_words[0:50]

        # split the list of tuples into two tuples
        words, counts=zip(*top_50_words)
        
        x_pos = range(len(words))
        plt.bar(x_pos, counts)
        plt.xticks(x_pos, words)
        plt.ylabel('the frequency of tokens')
        plt.title('Count of frequency of tokens')
        plt.xticks(rotation=90) 
        plt.show()
        

if __name__ == "__main__":

    input_file='python.txt'

    tweets_analysis(input_file)



