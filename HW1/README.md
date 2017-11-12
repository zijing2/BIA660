# Python Basics

## Assignment 1

### Part (1). Analyze the frequency of words in a string. A sample string is given below

### Part (2). Analyze token frequency in a text file

## Usage

import zhuang21

if __name__ == "__main__":  
    
    text='''Hello world!
        This is a hello world example !'''   
    print(count_token(text))
    
    analyzer=Text_Analyzer("foo.txt", "foo.csv")
    vocabulary=analyzer.analyze()

## Entry point

./zhuang21.py

## Input file

./foo.txt

## output file
This program will generate a foo.csv, the content is as follow:

a	2
great	2
is	2
world	2
hello	2
language	1
this	1
it's	1
yeah	1
example	1
python	1
