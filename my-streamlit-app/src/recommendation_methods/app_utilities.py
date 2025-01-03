import pandas as pd
from surprise import KNNBasic
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from collections import defaultdict


def get_unique_customer_ids(user_input:str , config):

    data_path = config['data_path']
    search_rows_limit = config['search_rows_limit']
    search_cust_limit = config['search_cust_limit']
    data = pd.read_pickle(data_path+ 'prepared_data.pkl')
    
    # Load the data
    data_subset = data.head(search_rows_limit)  # Adjust the number of rows as needed
    
    unique_cust = data_subset['cust_id'].unique()
    # Simulate combining different algorithms
    if user_input and user_input.isdigit() :
        n_unique_cust = unique_cust[:int(user_input)]
    else:
        n_unique_cust = []
    print(n_unique_cust)
    return n_unique_cust