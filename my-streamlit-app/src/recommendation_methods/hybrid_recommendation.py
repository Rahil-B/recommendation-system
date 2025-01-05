import pandas as pd
from surprise import KNNBasic
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from collections import defaultdict

class HybridRecommendationEngine:
    def __init__(self, config):
        self.data_path = config['data_path']
        self.search_rows_limit = config['search_rows_limit']
        self.search_cust_limit = config['search_cust_limit']
        self.data = pd.read_pickle(self.data_path + 'prepared_data.pkl')
        
        # Load the data
        self.data_subset = self.data.head(self.search_rows_limit)  # Adjust the number of rows as needed
        
        # Train collaborative filtering model
        self.collab_algo = self.collaborative_filtering(self.data_subset)

        # Compute similarity matrix for content-based filtering
        self.content_similarity_matrix = self.content_based_filtering(self.data_subset)
        # print(content_similarity_matrix)
        
    
    def collaborative_filtering(self, data):
        reader = Reader(rating_scale=(-5, 5))
        dataset = Dataset.load_from_df(data[['cust_id', 'prod_cat_subcat', 'Qty']], reader)
        trainset, testset = train_test_split(dataset, test_size=0.2)
        algo = KNNBasic(k=40, min_k=1, sim_options={'name': 'cosine', 'user_based': True})
        algo.fit(trainset)
        return algo

    def content_based_filtering(self, data):
        # Create TF-IDF vectorizer
        tfidf = TfidfVectorizer(stop_words='english')

        # Create user profiles based on 'prod_cat_subcat', 'Store_type', and 'Gender'
        user_profiles = data['prod_cat_subcat'] + ' ' + data['Store_type'] +  ' ' + data['Gender']

        # Fit and transform the user profiles
        user_tfidf_matrix = tfidf.fit_transform(user_profiles)

        # Compute cosine similarity between user profiles
        cosine_sim_user = linear_kernel(user_tfidf_matrix, user_tfidf_matrix)

        return cosine_sim_user
    
    def get_hybrid_recommendations(self, collab_algo, content_similarity_matrix, user_id, data, n=10):
        # Collaborative filtering recommendations
        collab_recs = self.get_collaborative_recommendations(collab_algo, user_id, data, n)

        # Content-based filtering recommendations
        content_recs = self.get_content_based_recommendations(content_similarity_matrix, user_id, data, n)

        # Combine recommendations from both methods
        hybrid_recs = list(set(collab_recs + content_recs))[:n]

        return hybrid_recs

    def get_collaborative_recommendations(self, algo, user_id, data, n=10):
        # Filter the dataset to include only items the user hasn't rated
        rated_items = data[data['cust_id'] == user_id]['prod_cat_subcat'].tolist()
        unrated_items = list(set(data['prod_cat_subcat'].unique()) - set(rated_items))

        # Use algo.predict() to get predictions for the user and unrated items
        pred = [algo.predict(user_id, item_id) for item_id in unrated_items]
        # Sort predictions by estimated rating
        pred.sort(key=lambda x: x.est, reverse=True)
        # Return top N recommended items
        return [p.iid for p in pred[:n]]
    
    def get_content_based_recommendations(self,similarity_matrix, user_id, data, n=10):
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
    
    
    def hybrid_recommendation(self, user_input:str , config):

        # Get hybrid recommendations for the first 10 customer IDs
        hybrid_recommendations = {}
        for cust_id in self.data_subset['cust_id'].unique()[:self.search_cust_limit]:
            hybrid_recommendations[cust_id] = self.get_hybrid_recommendations(self.collab_algo, self.content_similarity_matrix, cust_id, self.data_subset)
        
        # Simulate combining different algorithms
        if user_input and user_input.isdigit() and int(user_input) in hybrid_recommendations.keys():
            recommendations = hybrid_recommendations[int(user_input)]
        
        return recommendations
        
