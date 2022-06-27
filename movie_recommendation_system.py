from ast import literal_eval
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np

# Step 1: Perform Exploratory Data Analysis (EDA) on the data

# load the movie dataset using pandas
credits_df = pd.read_csv("tmdb_5000_credits.csv")
movies_df = pd.read_csv("tmdb_5000_movies.csv")

# We only need the id, title, cast, and crew columns of the credits dataframe. Let’s merge the dataframes into one on the column ‘id’
credits_df.columns = ['id', 'tittle', 'cast', 'crew']
movies_df = movies_df.merge(credits_df, on="id")

# Step 2: Build the Movie Recommender System

# use literal_eval function for cleaning the data
features = ["cast", "crew", "keywords", "genres"]
for feature in features:
    movies_df[feature] = movies_df[feature].apply(literal_eval)

#storing all the movie tittles in a list
movies_df['title'] = movies_df['title'].str.lower()
all_tittle = [i for i in movies_df['title']]


# The get_director() function extracts the name of the director of the movie
def get_director(x):
    for i in x:
        if i["job"] == "Director":
            return i["name"]
    return np.nan

# The get_list() returns the top 3 elements or the entire list whichever is more
def get_list(x):
    if isinstance(x, list):
        names = [i["name"] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names
    return []


# applying both the functions get_director() and get_list() to our dataset
movies_df["director"] = movies_df["crew"].apply(get_director)

features = ["cast", "keywords", "genres"]
for feature in features:
    movies_df[feature] = movies_df[feature].apply(get_list)

# print(movies_df[['title', 'cast', 'director', 'keywords', 'genres']].head())


# convert the above feature instances into lowercase and remove all the spaces between them
def clean_data(row):
    if isinstance(row, list):
        return [str.lower(i.replace(" ", "")) for i in row]
    else:
        if isinstance(row, str):
            return str.lower(row.replace(" ", ""))
        else:
            return ""


features = ['cast', 'keywords', 'director', 'genres']
for feature in features:
    movies_df[feature] = movies_df[feature].apply(clean_data)


# create a “soup” containing all of the metadata information extracted to input into the vectorizer
def create_soup(features):
    return ' '.join(features['keywords']) + ' ' + ' '.join(features['cast']) + ' ' + features['director'] + ' ' + ' '.join(features['genres'])


movies_df["soup"] = movies_df.apply(create_soup, axis=1)

# our text data should be preprocessed and converted into a vectorizer using the CountVectorizer
# CountVectorizer counts the frequency of each word and returns a 2D vector containing frequencies
count_vectorizer = CountVectorizer(stop_words="english")
count_matrix = count_vectorizer.fit_transform(movies_df["soup"])


# we use the cosine similarity score as this is just the dot product of the vector output by the CountVectorizer
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)


# reset the indices of our dataframe
movies_df = movies_df.reset_index()

# Create a reverse mapping of movie titles to indices. By this, we can easily find the title of the movie based on the index
indices = pd.Series(movies_df.index, index=movies_df['title'])

# output:
#index  [tittle1,tittle2,tittle3]
# indices[0,      1,      2...]

# remove duplicates if any
indices = pd.Series(movies_df.index, index=movies_df["title"]).drop_duplicates()


# Step 3: Get recommendations for the movies

# Create a function that takes in the movie title and the cosine similarity score as input and returns the top 10 movies similar to it
def get_recommendations(title, cosine_sim=cosine_sim2):
    # Get the index of the movie using the title.
    idx = indices[title]
    # Get the list of similarity scores of the movies concerning all the movies
    # Enumerate them (create tuples) with the first element being the index and the second element is the cosine similarity score
    similarity_scores = list(enumerate(cosine_sim[idx]))
    # Sort the list of tuples in descending order based on the similarity score
    similarity_scores = sorted(
        similarity_scores, key=lambda x: x[1], reverse=True)
    # Get the list of the indices of the top 10 movies from the above sorted list. Exclude the first element because it is the title itself
    similarity_scores = similarity_scores[1:11]
    # Map those indices to their respective titles and return the movies list
    movies_indices = [ind[0] for ind in similarity_scores]
    movies = movies_df["title"].iloc[movies_indices]
    return [j for j in movies]
    # return movies


moviename ="the pink panther"
print(*get_recommendations(moviename, cosine_sim2),sep="\n")

