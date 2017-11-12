#Assignment 5: Arrays and Text Clustering
#Task 1: Data normalization using array operations and broadcasting

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import numpy as np
import pandas as pd
from nltk.cluster import KMeansClusterer, cosine_distance


def minmax_norm(array):
    
    min_arr = [];
    max_min_arr = [];
    nomalized_arr = [];
    min_arr = np.amin(array, axis=1)
    max_arr = np.amax(array, axis=1)
    max_min_arr = max_arr-min_arr;

    print array
    print min_arr
    print max_min_arr
    nomalized_arr = ((0.0+array.T-min_arr)/max_min_arr).T
    
    print "------ task 1 ------"
    print "input:"
    print array
    print "shape of the input:"
    print array.shape
    print "shape of the output:"
    print nomalized_arr.shape
    print "ouput:"
    print nomalized_arr
    print "------ end of task 1 ------\n"
    
# MAIN BLOCK
if __name__ == "__main__": 
    array = np.random.randint(0,10,(4,3))
    minmax_norm(array);

#Task 2: Text Clustering
if __name__ == "__main__":
    import pickle
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn import metrics

    print "------ task 2 ------"
    data=pickle.load(open("4NewsGroup.pkl","r"))

    text,target=zip(*data)
    text=list(text)
    target=list(target)

    tfidf_vect = TfidfVectorizer(stop_words="english", min_df=10, max_df=0.9) 
    dtm= tfidf_vect.fit_transform(text)

    num_clusters = 4

    clusterer = KMeansClusterer(num_clusters, cosine_distance, repeats=5)
    clusters = clusterer.cluster(dtm.toarray(), assign_clusters=True)

    order_centroids = np.array(clusterer.means()).argsort()[:, ::-1] 
    voc=tfidf_vect.vocabulary_
    voc_lookup={tfidf_vect.vocabulary_[word]:word \
                for word in tfidf_vect.vocabulary_}

    for i in range(num_clusters):
        top_words=[voc_lookup[word_index] \
                   for word_index in order_centroids[i, :20]]
        print("Cluster %d: %s " % (i, "; ".join(top_words)))


    cluster_dict={0:0, 1:2, 2:1, 3:3}
    predicted_clusters=[cluster_dict[i] for i in clusters]
 
    #print target
    print(metrics.classification_report(target, predicted_clusters))
    
    # get confusion matrix
    df=pd.DataFrame(zip(target, predicted_clusters), columns=['actual','predict'])
    df.groupby(['predict','actual']).size()
    print df

    # according to confusion matrix, we can get cluster_dict as follows:
    # {0:0, 1:2, 2:1, 3:3}
    # also we can get this conclusion by scanning and labeling(adding meaningful labels) the sample in the excel first, 
    # then mapping with the name of the clusters if they looks similar.
    
    ## Write down the name of each cluster as comments in your code.
    # cluster 0: edu.math (label 0 in excel)
    # cluster 1: edu.sport (label 2 in excel)
    # cluster 2: edu.engineering (label 1 in excel)
    # cluster 3: edu.astronomy.geography (label 3 in excel)

    print "name of each cluster base on the weight of TfIdf:"
    print "cluster 0: edu.math (label 0 in excel)\ncluster 1: edu.sport (label 2 in excel)\ncluster 2: edu.engineering (label 1 in excel)\ncluster 3: edu.astronomy.geography (label 3 in excel)\n"
    print "------ end of task 2 ------"