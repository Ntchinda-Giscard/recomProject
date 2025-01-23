import os
from sklearn.model_selection import train_test_split
from mlProject.entity.config_entity import DataProcessingConfig
import pandas as pd
from mlProject import logger
from typing import Tuple
import pandas as pd
from mlProject.base import FeatureExtractor
from mlProject.config.configuration import ConfigurationManager
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import numpy as np

class MovieFeatureExtractor(FeatureExtractor):
    def generate_features(self, movies_df: pd.DataFrame, tags_df: pd.DataFrame) -> np.ndarray:
        genres = movies_df['genres'].str.get_dummies('|')
        movies_df['year'] = movies_df['title'].str.extract('(\d{4})', expand=False)
        movies_df['year'] = pd.to_numeric(movies_df['year'], errors='coerce')
        year_normalized = StandardScaler().fit_transform(movies_df[['year']].fillna(movies_df['year'].mean()))
        movie_tags = tags_df.groupby('movieId')['tag'].agg(lambda x: ' '.join(x)).reset_index()
        movie_tags = movies_df[['movieId']].merge(movie_tags, on='movieId', how='left')
        movie_tags['tag'] = movie_tags['tag'].fillna('')
        tfidf = TfidfVectorizer(max_features=50, stop_words='english')
        tag_features = tfidf.fit_transform(movie_tags['tag']).toarray()
        movie_features = np.hstack([genres.values, year_normalized, tag_features])
        return movie_features

class UserFeatureExtractor(FeatureExtractor):
    def calculate_user_genre_ratings(self, ratings_df, movies_df):
        genres = movies_df['genres'].str.get_dummies('|')
        movies_with_genres = pd.concat([movies_df[['movieId']], genres], axis=1)
        ratings_with_genres = ratings_df.merge(movies_with_genres, on='movieId')
        genre_columns = genres.columns
        user_genre_ratings = []
        for user_id in ratings_with_genres['userId'].unique():
            user_ratings = ratings_with_genres[ratings_with_genres['userId'] == user_id]
            genre_avgs = {'userId': user_id}
            for genre in genre_columns:
                genre_movies = user_ratings[user_ratings[genre] == 1]
                genre_avgs[f'{genre}_avg_rating'] = genre_movies['rating'].mean() if len(genre_movies) > 0 else 0
            user_genre_ratings.append(genre_avgs)
        return pd.DataFrame(user_genre_ratings)

    def generate_features(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame, tags_df: pd.DataFrame) -> pd.DataFrame:
        rating_stats = ratings_df.groupby('userId').agg({'rating': ['mean', 'std', 'count']}).fillna(0)
        rating_stats.columns = ['avg_rating', 'std_rating', 'rating_count']
        rating_stats = rating_stats.reset_index()
        tag_stats = tags_df.groupby('userId').agg({'tag': 'count'}).reset_index()
        tag_stats.columns = ['userId', 'tag_count']
        genre_ratings = self.calculate_user_genre_ratings(ratings_df, movies_df)
        user_features = rating_stats.merge(tag_stats, on='userId', how='left')
        user_features = user_features.merge(genre_ratings, on='userId', how='left')
        user_features = user_features.fillna(0)
        user_ids = user_features['userId']
        feature_columns = [col for col in user_features.columns if col != 'userId']
        user_features_normalized = StandardScaler().fit_transform(user_features[feature_columns])
        return user_features_normalized, user_features

class DataPreprocessor:
    def __init__(self, config: DataProcessingConfig) -> None:
        self.movie_feature_extractor = MovieFeatureExtractor()
        self.user_feature_extractor = UserFeatureExtractor()
        self.config = config
        self.movies_df = pd.read_csv(self.config.movies)
        self.ratings_df = pd.read_csv(self.config.ratings)
        self.tags_df = pd.read_csv(self.config.tags)
        self.top_5_users = self.ratings_df['userId'].value_counts().head(5).index
        self.ratings_df = self.ratings_df[self.ratings_df['userId'].isin(self.top_5_users)]
        self.relevant_movies = self.ratings_df['movieId'].unique()
        self.movies_df = self.movies_df[self.movies_df['movieId'].isin(self.relevant_movies)]
        self.tags_df = self.tags_df[self.tags_df['userId'].isin(self.top_5_users)]
    
    def prepare_training_data(self, ratings_df, movie_features, user_features):
        user_encoder = LabelEncoder()
        movie_encoder = LabelEncoder()
        ratings_df['user_encoded'] = user_encoder.fit_transform(self.ratings_df['userId'])
        ratings_df['movie_encoded'] = movie_encoder.fit_transform(self.ratings_df['movieId'])
        X_user = user_features[ratings_df['user_encoded']]
        X_movie = movie_features[ratings_df['movie_encoded']]
        y = ratings_df['rating'].values
        return X_user, X_movie, y, user_encoder, movie_encoder
    
    def process_data(self) -> pd.DataFrame:
        logger.info(f"Extracting \033[36mMovies features...⏳\033[0m")
        movie_features = self.movie_feature_extractor.generate_features(self.movies_df, self.tags_df)
        logger.info(f"Extracting \033[36mMovies features\033[0m \033[34mcompleted\033[0m ✅ ")
        logger.info(f"Extracting \033[36mUser features...⏳\033[0m")
        user_features, user_features_df = self.user_feature_extractor.generate_features(self.ratings_df, self.movies_df, self.tags_df)
        logger.info(f"Extracting \033[36mUser features\033[0m \033[34mcompleted\033[0m ✅")
        logger.info(f"Merging features with \033[36mratings...⏳\033[0m")
        X_user, X_movie, y, _, _ = self.prepare_training_data(self.ratings_df, movie_features, user_features)
        logger.info(f"Merging features with ratings \033[34mcompleted\033[0m ✅")
        return X_user, X_movie, y

    def train_validation_split(self, X_user, X_movie, y) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        train_size = 0.8
        indices = np.arange(len(y))
        logger.info(f"Spliting \033[36mfinal dataset features...⏳\033[0m")
        train_indices, val_indices = train_test_split(indices, train_size=train_size, random_state=42)
        train_user_features = X_user[train_indices]
        train_movie_features = X_movie[train_indices]
        train_ratings = y[train_indices]
        val_user_features = X_user[val_indices]
        val_movie_features = X_movie[val_indices]
        val_ratings = y[val_indices]
        logger.info(f"Training set size: y_train = {train_ratings.shape}, X_train_user = {train_user_features.shape}, X_train_movie = {train_movie_features.shape}")
        logger.info(f"Validation set size: y_val = {val_ratings.shape}, X_val_user = {val_user_features.shape}, X_val_movie = {val_movie_features.shape}")
        logger.info(f"Spliting \033[36mfinal dataset\033[0m \033[34mcompleted\033[0m ✅")
        return train_user_features, train_movie_features, train_ratings, val_user_features, val_movie_features, val_ratings