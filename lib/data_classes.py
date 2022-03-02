import pandas as pd
from pymed import PubMed
import traceback
# %%
class PubMedData:
    def __init__(self,search_term,max_results):
        self.search_term = search_term
        self.max_results = max_results
        self.df = None
        
        try:
            self._pull_data_from_pubmed()
        except Exception as e:
            print("Error in dataframe formation")
            traceback.print_exc()
            
        
    def _pull_data_from_pubmed(self):
        print("***Note: Avoid making concurrent and identical requests from PubMed.***\n ***Create this object only once per search term wherever possible***")
        #^note: for later we can set up some type of SQL database and build up a repository of tested search terms
        pubmed = PubMed(tool="curiosity", email="jdavbraun@gmail.com") #not required but they ask for details
        results = pubmed.query(self.search_term,self.max_results) #this return a special single-use iterator object.

        results_list = []
        for article in results:
            results_list.append(article.toDict())
            
        #converting collected and converted results into a dataframe
        self.df = pd.DataFrame.from_records(results_list)
        if self.df.size > 0: 
            print("Success")       