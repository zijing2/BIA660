import csv 
from string import punctuation

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

class Text_Analyzer(object):
    
    def __init__(self, input_file, output_file):
        # add code to initialize an instance
        self.input_file = input_file
        self.output_file = output_file
          
    def analyze(self):
        # add your code
        f = open(self.input_file, "r") 
        text = ""
        dict1 = {}
        for line in f:
            text += line
        f.close()
        dict1 = count_token(text)
       
        with open("foo.csv", "w") as f:  
            rows = []
            for x in dict1:
                rows.append((x,dict1[x]))
            rows = sorted(rows, key=lambda row: row[1], reverse=True)
            #print(rows)
            writer=csv.writer(f, delimiter=',')          
            writer.writerows(rows)

analyzer=Text_Analyzer("foo.txt", "foo.csv")
vocabulary=analyzer.analyze()
