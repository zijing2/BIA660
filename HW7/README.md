# Arrays; Text Clustering; Reading

## Assignment 7

## Usage

python zhuang21.py

## Entry point

./zhuang21.py

## Notice

what you need to include in the same directory is as following:

./4NewsGroup.pkl

## Output would be as follows:

------ task 1 ------
input:
[[9 1 9]
 [6 7 7]
 [9 9 2]
 [4 9 1]]
shape of the input:
(4, 3)
shape of the output:
(4, 3)
ouput:
[[ 1.     0.     1.   ]
 [ 0.     1.     1.   ]
 [ 1.     1.     0.   ]
 [ 0.375  1.     0.   ]]
------ end of task 1 ------

------ task 2 ------
Cluster 0: edu; mac; apple; com; lines; subject; organization; university; thanks; posting; drive; nntp; host; know; does; se; problem; ca; monitor; quadra 
Cluster 1: car; com; edu; cars; article; writes; oil; engine; just; subject; organization; lines; don; like; good; university; dealer; new; usa; hp 
Cluster 2: edu; baseball; year; team; game; players; games; writes; article; cs; com; runs; jewish; organization; subject; win; lines; season; university; braves 
Cluster 3: space; nasa; henry; edu; access; gov; alaska; digex; moon; toronto; pat; com; article; launch; zoo; writes; just; orbit; jpl; earth 
             precision    recall  f1-score   support

          0       0.68      0.98      0.80       578
          1       0.96      0.90      0.93       594
          2       0.98      0.90      0.94       597
          3       0.99      0.70      0.82       593

avg / total       0.90      0.87      0.87      2362

name of each cluster base on the weight of TfIdf:
0: engineer
1: astronomy and geography
2: sport
3: mathematics

------ end of task 2 ------






