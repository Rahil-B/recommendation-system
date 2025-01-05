import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from collections import defaultdict

class ContentBasedRecommendationEngine:
    def __init__(self, config):
        self.data_path = config['data_path']
        self.search_rows_limit = config['search_rows_limit']
        self.search_cust_limit = config['search_cust_limit']
        
        data = pd.read_pickle(self.data_path+ 'prepared_data.pkl')

        # Load a subset of the dataset for demonstration
        self.data_subset = data.head(self.search_rows_limit)  # Adjust the number of rows as needed
        
        # Compute similarity matrix
        self.cosine_sim_user = self.content_based_filtering(self.data_subset)

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
    
    def get_top_n_recommendations(self, similarity_matrix, user_id, data, n=10):
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
    
    def content_based_recommendation(self, user_input: str):

        # Get recommendations for the first 10 customer IDs
        recommendations = {}
        for cust_id in self.data_subset['cust_id'].unique()[:self.search_cust_limit]:
            recommendations[cust_id] = self.get_top_n_recommendations(self.cosine_sim_user, cust_id, self.data_subset)

        # print(recommendations.keys())
        # print(user_input.isdigit())
        if user_input and user_input.isdigit() and int(user_input) in recommendations.keys():
            recommendations = recommendations[int(user_input)]
        else:
            recommendations = "Caution : Customer Id not found in the dataset"

        return recommendations
        