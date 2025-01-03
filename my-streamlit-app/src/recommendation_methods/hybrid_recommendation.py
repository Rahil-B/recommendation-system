import pandas as pd
from surprise import KNNBasic
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from collections import defaultdict

# Collaborative Filtering with Surprise
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

    return algo

# Content-Based Filtering
def content_based_filtering(data):
    # Create TF-IDF vectorizer
    tfidf = TfidfVectorizer(stop_words='english')

    # Create user profiles based on 'prod_cat_subcat', 'Store_type', and 'Gender'
    user_profiles = data['prod_cat_subcat'] + ' ' + data['Store_type'] +  ' ' + data['Gender']

    # Fit and transform the user profiles
    user_tfidf_matrix = tfidf.fit_transform(user_profiles)

    # Compute cosine similarity between user profiles
    cosine_sim_user = linear_kernel(user_tfidf_matrix, user_tfidf_matrix)

    return cosine_sim_user

def get_hybrid_recommendations(collab_algo, content_similarity_matrix, user_id, data, n=10):
    # Collaborative filtering recommendations
    collab_recs = get_collaborative_recommendations(collab_algo, user_id, data, n)

    # Content-based filtering recommendations
    content_recs = get_content_based_recommendations(content_similarity_matrix, user_id, data, n)

    # Combine recommendations from both methods
    hybrid_recs = list(set(collab_recs + content_recs))[:n]

    return hybrid_recs

def get_collaborative_recommendations(algo, user_id, data, n=10):
    # Filter the dataset to include only items the user hasn't rated
    rated_items = data[data['cust_id'] == user_id]['prod_cat_subcat'].tolist()
    unrated_items = list(set(data['prod_cat_subcat'].unique()) - set(rated_items))

    # Use algo.predict() to get predictions for the user and unrated items
    pred = [algo.predict(user_id, item_id) for item_id in unrated_items]
    # Sort predictions by estimated rating
    pred.sort(key=lambda x: x.est, reverse=True)
    # Return top N recommended items
    return [p.iid for p in pred[:n]]

def get_content_based_recommendations(similarity_matrix, user_id, data, n=10):
    # Find index of the user
    user_index = data[data['cust_id'] == user_id].index[0]

    # Compute similarity scores between the user and all other users
    user_similarity_scores = similarity_matrix[user_index]

    # Get indices of top N similar users (excluding the user itself)
    top_n_similar_users = user_similarity_scores.argsort()[::-1][1:n+1]

    # Get recommendations based on top similar users
    recommendations = defaultdict(int)
    for similar_user_index in top_n_similar_users:
        similar_user_id = data.loc[similar_user_index, 'cust_id']
        similar_user_items = data.loc[data['cust_id'] == similar_user_id, 'prod_cat_subcat']
        for item in similar_user_items:
            recommendations[item] += 1

    # Sort recommendations by frequency and return top N
    top_n_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n]
    return [item for item, _ in top_n_recommendations]


def hybrid_recommendation(user_input:str , config):

    data_path = config['data_path']
    search_rows_limit = config['search_rows_limit']
    search_cust_limit = config['search_cust_limit']
    data = pd.read_pickle(data_path+ 'prepared_data.pkl')
    
    # Load the data
    data_subset = data.head(search_rows_limit)  # Adjust the number of rows as needed
    
    # Train collaborative filtering model
    collab_algo = collaborative_filtering(data_subset)

    # Compute similarity matrix for content-based filtering
    content_similarity_matrix = content_based_filtering(data_subset)
    # print(content_similarity_matrix)
    # Get hybrid recommendations for the first 10 customer IDs
    hybrid_recommendations = {}
    for cust_id in data_subset['cust_id'].unique()[:search_cust_limit]:
        hybrid_recommendations[cust_id] = get_hybrid_recommendations(collab_algo, content_similarity_matrix, cust_id, data_subset)
    
    # Simulate combining different algorithms
    if user_input and user_input.isdigit() and int(user_input) in hybrid_recommendations.keys():
        recommendations = hybrid_recommendations[int(user_input)]
    
    return recommendations