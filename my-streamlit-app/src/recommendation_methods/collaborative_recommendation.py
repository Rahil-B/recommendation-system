import pandas as pd
from surprise import KNNBasic
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
from collections import defaultdict


# data_path = 'Recommendation/data/'
data_path = '/Users/rahil/Documents/Online_Retail_Recommendation_Project/'

# Collaborative Filtering with Cosine Similarity
def collaborative_filtering(data):
    # Create Surprise Dataset
    reader = Reader(rating_scale=(-5, 5))
    dataset = Dataset.load_from_df(data[['cust_id', 'prod_cat_subcat', 'Qty']], reader)

    # Split dataset into train and test sets
    trainset, testset = train_test_split(dataset, test_size=0.2)

    # Instantiate KNNBasic algorithm with cosine similarity
    algo = KNNBasic(k=40, min_k=1, sim_options={'name': 'cosine', 'user_based': True})

    # Train the model
    algo.fit(trainset)

    # Make predictions on the test set
    predictions = algo.test(testset)

    # Compute RMSE
    rmse = accuracy.rmse(predictions)

    return algo

def get_top_n_recommendations(algo, user_id, data, n=10):
    # Filter the dataset to include only items the user hasn't rated
    rated_items = data[data['cust_id'] == user_id]['prod_cat_subcat'].tolist()
    unrated_items = list(set(data['prod_cat_subcat'].unique()) - set(rated_items))

    # Use algo.predict() to get predictions for the user and unrated items
    pred = [algo.predict(user_id, item_id) for item_id in unrated_items]
    # Sort predictions by estimated rating
    pred.sort(key=lambda x: x.est, reverse=True)
    # Return top N recommended items
    return [p.iid for p in pred[:n]]


def collaborative_recommendation(user_input:str , config):

    data_path = config['data_path']
    search_rows_limit = config['search_rows_limit']
    search_cust_limit = config['search_cust_limit']
    
    data = pd.read_pickle(data_path+ 'prepared_data.pkl')
    
    # Load a subset of the dataset for demonstration
    data_subset = data.head(search_rows_limit) 

    
    # Train collaborative filtering model
    algo = collaborative_filtering(data_subset)

    # Get recommendations for the first 10 customer IDs
    recommendations = {}
    for cust_id in data_subset['cust_id'].unique()[:search_cust_limit]:
        recommendations[cust_id] = get_top_n_recommendations(algo, cust_id, data_subset)
    
    # Create DataFrame from recommendations
    # df_recommendations = pd.DataFrame(recommendations.items(), columns=['cust_id', 'recommended_items'])
    # df_recommendations
    
    # Here we would implement the logic to filter or recommend items based on user_input
    # print("records",len(recommendations), recommendations.keys())
    # print(user_input.isdigit())
    if user_input and user_input.isdigit() and int(user_input) in recommendations.keys():
        # print("user_input",user_input)
        recommendations = recommendations[int(user_input)]
    else:
        recommendations = "Caution : Customer Id not found in the dataset"
    
    return recommendations