# **🎬 Netflix Recommendation Engine**

## Scaling Collaborative Filtering with SVD and Docker

### 📌 Project Overview:

This project is an end-to-end Machine Learning application that provides personalized movie recommendations. Using the Netflix Prize Dataset, I processed over 24 million user ratings to build a recommendation engine powered by **Singular Value Decomposition (SVD)**.

### 🛠️ The Technical Deep-Dive 

**1. Data Engineering & "De-infusion**

"The raw Netflix dataset was provided in a "column-infused" format where Movie IDs were headers followed by Customer IDs and Ratings.

* Challenge: Parsing 24M+ rows efficiently.
* Solution: Developed a custom Python script to "de-infuse" the data, mapping every rating to its respective Movie ID into a structured CSV.

**2. Strategic Benchmarking (Noise Reduction)**

To ensure the model suggests high-quality, relevant titles, I applied a 60th percentile benchmark:

* Active Users: Filtered out users with low rating frequency.
* Relevant Movies: Dropped niche titles with very few ratings.
* Result: A "Golden Dataset" that maintains 40% of the most significant interactions, significantly improving model accuracy and reducing training time.

**3. SVD Model Training**

Using the scikit-surprise library, I implemented Matrix Factorization.

* Model: Singular Value Decomposition (SVD).
* Mathematics: $R \approx U \Sigma V^T$.
* Inference: The model predicts the rating a specific user would give to every movie in the database and returns the top 5 highest-scored titles.


### 🚀 Deployment & Reproducibility

The app is built to be environment-agnostic using Docker.

* Containerization: Forced a Python 3.10-slim environment to ensure compatibility with scikit-surprise and numpy.
* Port Mapping: Optimized for Hugging Face Spaces on port 7860.
* Storage: Utilized Git LFS to manage the 28MB+ serialized model file (netflix_model.pkl).

### 📈 Future Enhancements 
* [ ] Implement a "Surprise Me" button for exploratory recommendations.
* [ ] Add movie posters via the TMDB API.
* [ ] Transition from SVD to Deep Learning-based Neural Collaborative Filtering (NCF).


##### How to Run Locally

* Clone the repo: git clone https://github.com/Rohit-hub00/Netflix-Recommendation-System.git
* Install dependencies: pip install -r requirements.txt
* streamlit run app.py
