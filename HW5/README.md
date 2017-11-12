# TF-IDF

## Assignment 5

## Usage

python zhuang21.py

## Entry point

./zhuang21.py

## Notice

what you need to include in the same directory is as following:

./amazon_review_300.csv

if you want to turn off the lemmatization, please modify line 36 get_doc_tokens(doc, True) into get_doc_tokens(doc, False)

## Output would be as follows (For the similarity part, the first review is the seleted one)

Smoothed TF-IDF Matrix
[[ 0.  0.  0. ...,  0.  0.  0.]
 [ 0.  0.  0. ...,  0.  0.  0.]
 [ 0.  0.  0. ...,  0.  0.  0.]
 ..., 
 [ 0.  0.  0. ...,  0.  0.  0.]
 [ 0.  0.  0. ...,  0.  0.  0.]
 [ 0.  0.  0. ...,  0.  0.  0.]]

top10 similarity of the first review
[(69, 0.2141006635204249), (4, 0.1856263304750434), (195, 0.18053052570390327), (2, 0.17950120095468614), (212, 0.1742544269124907), (216, 0.17099243959703625), (210, 0.16810560204703884), (220, 0.1495523013617911), (71, 0.14619436627508675), (70, 0.139819177088212)]



