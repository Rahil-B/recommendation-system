import streamlit as st
import yaml
from recommendation_methods.collaborative_recommendation import collaborative_recommendation
from recommendation_methods.content_based_recommendation import content_based_recommendation
from recommendation_methods.hybrid_recommendation import hybrid_recommendation
from recommendation_methods.app_utilities import get_unique_customer_ids


# Load configuration
with open('my-streamlit-app/src/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

def main():
    st.title("Recommendation System")

    st.sidebar.header("User Input (Insert user_id)")
    user_input = st.sidebar.text_input("Enter your preferences:")

    if st.sidebar.button("Collaborative Recommendation"):
        recommendations = collaborative_recommendation(user_input, config)
        st.write("Recommendations from Collaborative method:")
        st.write(recommendations)

    if st.sidebar.button("Content Based Recommendation"):
        recommendations = content_based_recommendation(user_input, config)
        st.write("Recommendations from Content Based:")
        st.write(recommendations)

    if st.sidebar.button("Hybrid Recommendation"):
        recommendations = hybrid_recommendation(user_input, config)
        st.write("Recommendations from Hybrid method:")
        st.write(recommendations)
    
    if st.sidebar.button("Get Customer N unique Id's"):
        recommendations = get_unique_customer_ids(user_input, config)
        st.write("Recommendations from Hybrid method:")
        st.write(recommendations)

if __name__ == "__main__":
    main()