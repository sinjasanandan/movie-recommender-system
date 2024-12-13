import pandas as pd
import numpy as np

#import data
ratings_data_url = 'https://liangfgithub.github.io/MovieData/ratings.dat?raw=true'
ratings = pd.read_csv(ratings_data_url, sep='::', engine = 'python', header=None)
ratings.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
ratings['Timestamp'] = pd.to_datetime(ratings['Timestamp'], unit='s')


movie_data_url = 'https://liangfgithub.github.io/MovieData/movies.dat?raw=true'
movies = pd.read_csv(movie_data_url, sep='::', engine = 'python',
                     encoding="ISO-8859-1", header = None)
movies.columns = ['MovieID', 'Title', 'Genres']




# #get top_movies matrix
# ratings_matrix = pd.read_csv('ratings_matrix.csv')

# rating_mtx = ratings_matrix.copy()
# rating_mtx.columns = rating_mtx.columns.str.strip('m').astype(int)

# # From ratings matrix, get average ratings and total number of reviews for each movie
# movie_review_summary = rating_mtx.apply(lambda x: x.dropna().mean(), axis=0).to_frame('Average Rating')
# movie_review_summary['Average Rating'] = movie_review_summary['Average Rating'].round(3)
# movie_review_summary['Total Reviews'] = rating_mtx.notna().sum(axis=0)
# movie_review_summary = movie_review_summary.rename_axis('Movie ID').reset_index()

# filtered_movies = movie_review_summary[movie_review_summary['Total Reviews'] > 1000]
# top_movies = filtered_movies.sort_values(by='Average Rating', ascending=False).head(10)
# top_movies

# # Add corresponding movie titles (from movies.dat)
# movie_ids = top_movies['Movie ID'].tolist()
# movie_titles = movies[movies['MovieID'].isin(movie_ids)]['Title']
# top_movies_with_titles = pd.merge(top_movies, movies[['MovieID','Title']], left_on='Movie ID', right_on='MovieID', how='left')
# top_movies_with_titles = top_movies_with_titles.drop(columns=['MovieID'])



#get S_top30 matrix
sim_mtx_url = 'https://www.dropbox.com/scl/fi/pu6enu2l6e41cum2healh/similarity_matrix.csv?rlkey=xrllq2vzueh1souasoq7oyhcp&st=nj3cdv83&dl=1'
S_matrix = pd.read_csv(sim_mtx_url, index_col=0)



def keep_top_30(row):
    sorted_indices = row.nlargest(30).index

    # Set mask to indexes of top 30 values in row
    mask = row.index.isin(sorted_indices)
    # Set rest of the values to nan
    row[~mask] = np.nan
    return row

S_top30 = S_matrix.apply(keep_top_30, axis=1)
#print(S_top30.index[:5])




def myIBCF(newuser, S, top_movies):

  predictions = np.zeros(len(newuser))

  rated_indices = np.where(~np.isnan(newuser))[0]
  rated_indices = rated_indices.astype(int)
  unrated_indices = np.where(np.isnan(newuser))[0]
  

  columns = S.columns 
  rated_columns = [columns[idx] for idx in rated_indices]
  unrated_columns = [columns[idx] for idx in unrated_indices]  
  

  for i, movie in enumerate(unrated_columns):
 
      numerator = np.sum(S_top30.loc[str(movie), rated_columns] * newuser[rated_indices])
      denominator = np.sum(np.abs(S_top30.loc[str(movie), rated_columns]))

      if denominator > 0:
          #print( i , movie, numerator / denominator)
          predictions[unrated_indices[i]] = numerator / denominator
      else:
          #print("we entered the wrong state")
          predictions[unrated_indices[i]] = 0


  rec_indices = np.argsort(predictions)[::-1][:10]
  recommended_movies = [
      columns[idx] for idx in rec_indices if not np.isnan(predictions[idx])
  ]


  if len(recommended_movies) < 10:
        print("using system 1")
        for _, row in top_movies.iterrows():
            movie_id = row["Movie ID"]
            if f"m{movie_id}" not in recommended_movies and movie_id - 1 not in rated_indices:
                recommended_movies.append(f"m{movie_id}")
            if len(recommended_movies) == 10:
                break

  return recommended_movies


