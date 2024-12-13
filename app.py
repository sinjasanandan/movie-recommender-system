import streamlit as st
import pandas as pd
import numpy as np

from model import myIBCF, S_top30, top_movies_with_titles, movies, ratings_matrix


img_base_url = "https://liangfgithub.github.io/MovieImages/"

#title
st.title("Movie Recommendation System")
st.write("Select movies by clicking their images, rate them, and get recommendations!")

#new user creation
new_user = pd.Series(data=np.nan, index=S_top30.columns)

#display initial movies 5x5 grid
st.subheader("Pick from these 25 Classics or search more below!")
rows = 5
columns = 5
movies_to_display = movies.head(25)  
selected_movie_ids = []  

for row in range(rows):
    cols = st.columns(columns) 
    for col, (_, movie) in zip(cols, movies_to_display.iloc[row * columns:(row + 1) * columns].iterrows()):
        with col:
            image_url = f"{img_base_url}{movie['MovieID']}.jpg?raw=true"
            st.image(image_url, caption=movie["Title"], width=120)

            #add checkbox
            if st.checkbox(f"Select {movie['Title']}", key=f"select_{movie['MovieID']}"):
                selected_movie_ids.append(movie["MovieID"])

# Multiselect Search Bar for Other Movies
st.subheader("Step 1: Search and Add More Movies")
selected_additional_movies = st.multiselect(
    "Search for movies:", movies["Title"].tolist()
)

#add searched movies to the selected list
for additional_movie in selected_additional_movies:
    movie_id = movies.loc[movies["Title"] == additional_movie, "MovieID"].values[0]
    if movie_id not in selected_movie_ids:
        selected_movie_ids.append(movie_id)

#Ratings for each movie
if selected_movie_ids:
    st.subheader("Step 2: Rate Your Selected Movies")
    for movie_id in selected_movie_ids:
        movie_title = movies.loc[movies["MovieID"] == movie_id, "Title"].values[0]
        rating = st.slider(f"Rate {movie_title}:", 1, 5, 5)  
        new_user[f"m{movie_id}"] = rating  


if st.button("Step 3: Get Recommendations"):
    if selected_movie_ids:
        recommended_movies = myIBCF(new_user.to_numpy(), S_top30, movies)

        recommended_movie_ids = [int(movie[1:]) for movie in recommended_movies]

        recommended_movies_df = movies[movies["MovieID"].isin(recommended_movie_ids)]

        st.subheader("Recommended Movies:")
        rows = 5
        cols = 5
        for row in range((len(recommended_movies_df) + cols - 1) // cols):  
            row_columns = st.columns(cols)
            for col, (_, movie) in zip(
                row_columns, recommended_movies_df.iloc[row * cols : (row + 1) * cols].iterrows()
            ):
                with col:
                    image_url = f"{img_base_url}{movie['MovieID']}.jpg?raw=true"
                    st.image(image_url, caption=movie["Title"], width=120)

    else:
        st.write("Please select and rate at least one movie to get recommendations.")