# Scrape movie reviews

## Assignment 2

## Usage

from zhuang21 import getReviews


# best practice to test your class
# if your script is exported as a module,
# the following part is ignored
# this is equivalent to main() in Java

if __name__ == "__main__":  
    
    movie_id='finding_dory'
    reviews=getReviews(movie_id)
    #print(len(reviews))
    print(reviews)

## Entry point

./zhuang21.py

Notice: It takes a few seconds to show the result, so please be patient ^_^
