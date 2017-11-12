import requests    
from bs4 import BeautifulSoup  
def getReviews(movie_id):
    
    reviews=[]  # variable to hold all reviews
    
    page_url="https://www.rottentomatoes.com/m/"+movie_id+"/reviews/"

    while page_url!=None:
        page = requests.get(page_url) 
    
        if page.status_code!=200:    # a status code !=200 indicates a failure, exit the loop 
            page_url=None
        else:                       # status_code 200 indicates success.
            # insert your code to process page content
            soup = BeautifulSoup(page.content, 'html.parser')
            
            divs=soup.select("div.review_table div.review_table_row")
            for idx, div in enumerate(divs):
                date=None
                desc=None
                score=None
                date = div.select("div.review_date")[0].get_text()
                desc = div.select("div.the_review")[0].get_text()
                p_score = div.select("div.review_desc div.small.subtle")[0].get_text().split(":")
                if len(p_score) > 1:
                    score = p_score[1].strip()
                reviews.append((date, desc, score))
            
            # GET URL OF NEXT PAGE IF EXISTS, AND START A NEW ROUND OF LOOP
            # first set page_url is None. Update this value if you find next page
            page_url = None
            if soup.select("a.btn.btn-xs.btn-primary-rt")[1]["href"]!='#':
                page_url = "https://www.rottentomatoes.com/" + soup.select("a.btn.btn-xs.btn-primary-rt")[1]["href"]
            # second, look for next page. The URL is specified at (4) in the Figure above
            # third, if next page exists, update page_url using the "href" attribute in <a> tag
    return reviews

if __name__ == "__main__":    

    movie_id='finding_dory'

    reviews=getReviews(movie_id)

    print(len(reviews))
