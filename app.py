import streamlit as st
import pandas as pd
import pickle

# --- PAGE CONFIG ---
st.set_page_config(page_title="Netflix Recommender", layout="centered")
st.title("🎬 Netflix Recommendation Engine")

# --- LOAD PRE-PROCESSED DATA ---
@st.cache_resource 
def load_data():
    # Use 'rb' (read binary) for pickle files [cite: 228]
    with open('netflix_model.pkl', 'rb') as f:
        model = pickle.load(f)
    # Ensure encoding matches your original data [cite: 217]
    movies = pd.read_csv('movies_clean.csv', encoding='ISO-8859-1') 
    return model, movies

model, movies = load_data()

# --- USER INTERFACE ---
user_id = st.number_input("Enter your Customer ID:", min_value=1, value=1331154)

if st.button('Recommend'):
    with st.spinner('Calculating your personalized picks...'):
        # Apply the SVD prediction logic from your research [cite: 243, 259]
        movies['Score'] = movies['Movie_Id'].apply(lambda x: model.predict(user_id, x).est)
        
        # Sort to find the highest predicted ratings [cite: 261]
        top_movies = movies.sort_values('Score', ascending=False).head(5)
        
        st.subheader(f"Top 5 Recommendations for User {user_id}")
        for i, row in top_movies.iterrows():
            st.success(f"**{row['Name']}** ({int(row['Year'])})")
            st.write(f"Predicted Match: {round(row['Score'], 2)} / 5.0")
            st.divider() # Adds a clean line between movies