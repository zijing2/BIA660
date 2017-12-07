# ReviewAnalyser(using deep learning)

To choose the premium reviews with high quality. The details can be found in the presentation directory.

## deploying
download or check out the project, make sure the port 8887 and 3001 in the local are not in use.

## Usage

1. python ReviewAnalyser.py(this step can be skip if you dont want to train the model again)
2. python RestfulAPI.py(to test the restful api: use http://localhost:8887/)
3. cd admin; npm install; npm start;(access http://127.0.0.1:3001) account:admin pwd:admin

## Entry point

./ReviewAnalyser.py
./RestfulAPI.py
./admin/app.js

## Input file

./data_sample2.csv (use to train the data)
./word_sample.json (which is the subset of yelp_data for the doc_vector training)

# dependencies
python:
python 2.7
numpy
panda
gensim
nltk
sklearn
keras
matplot
etc..

nodejs:
express
uuid
request

# in the future
1. Add more trainning same (sample quantity)
2. Let more people to do the sample aggrement (sample quality) 
3. As long as the data sample incresing, the CNN training time processing increase as well. So we will put it into a distributed system
4. change the sentiment and review quality classification to regression. We can even change the labels classification to regression as well.


