import pandas as pd
from surprise import KNNBasic
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
from collections import defaultdict

class CollaborativeRecommendationEngine:
    def __init__(self, config):
        self.data_path = config['data_path']
        self.search_rows_limit = config['search_rows_limit']
        self.search_cust_limit = config['search_cust_limit']
        
        data = pd.read_pickle(self.data_path+ 'prepared_data.pkl')
        # Load a subset of the dataset for demonstration
        self.data_subset = data.head(self.search_rows_limit) 
        # Train collaborative filtering model
        self.algo = self.collaborative_filtering(self.data_subset)

    def collaborative_filtering(self, data):
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

    def get_top_n_recommendations(self, algo, user_id, data, n=10):
        # Filter the dataset to include only items the user hasn't rated
        rated_items = data[data['cust_id'] == user_id]['prod_cat_subcat'].tolist()
        unrated_items = list(set(data['prod_cat_subcat'].unique()) - set(rated_items))

        # Use algo.predict() to get predictions for the user and unrated items
        pred = [algo.predict(user_id, item_id) for item_id in unrated_items]
        # Sort predictions by estimated rating
        pred.sort(key=lambda x: x.est, reverse=True)
        # Return top N recommended items
        return [p.iid for p in pred[:n]]


    def collaborative_recommendation(self, user_input:str):
        # Get recommendations for the first 10 customer IDs
        recommendations = {}
        for cust_id in self.data_subset['cust_id'].unique()[:self.search_cust_limit]:
            recommendations[cust_id] = self.get_top_n_recommendations(self.algo, cust_id, self.data_subset)
        
        # Here we would implement the logic to filter or recommend items based on user_input
        
        if user_input and user_input.isdigit() and int(user_input) in recommendations.keys():
            recommendations = recommendations[int(user_input)]
        else:
            recommendations = "Caution : Customer Id not found in the dataset"
        return recommendations   
 