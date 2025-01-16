import os
from sklearn.model_selection import train_test_split
from mlProject.entity.config_entity import DataProcessingConfig
import pandas as pd
from mlProject import logger
from typing import Tuple
import pandas as pd
from mlProject.base import FeatureExtractor
from mlProject.config.configuration import ConfigurationManager
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from pathlib import Path




class MovieFeatureExtractor(FeatureExtractor):
    """
    Extracts features from movie data.

    Attributes:
        None

    Methods:
        load_dataset(file_path: Path) -> pd.DataFrame:
            Loads a dataset from a given file path and returns it as a pandas DataFrame.

        generate_features(movies_path: Path, tags_path: Path) -> pd.DataFrame:
            Generates features from movie and tag data and returns them as a pandas DataFrame.
    """

    def load_dataset(self, file_path: Path) -> pd.DataFrame:
        """
        Loads a dataset from a given file path and returns it as a pandas DataFrame.

        Args:
            file_path (Path): The path to the dataset file.

        Returns:
            pd.DataFrame: The loaded dataset as a pandas DataFrame.
        """
        return pd.read_csv(file_path)

    def generate_features(self, movies_path: Path, tags_path: Path) -> pd.DataFrame:
        """
        Generates features from movie and tag data and returns them as a pandas DataFrame.

        Args:
            movies_path (Path): The path to the movies dataset file.
            tags_path (Path): The path to the tags dataset file.

        Returns:
            pd.DataFrame: The generated features as a pandas DataFrame.
        """
        movies = self.load_dataset(movies_path)
        tags = self.load_dataset(tags_path)
        movies['genres'] = movies['genres'].str.split('|')
        genre_set = set(genre for genres in movies['genres'] for genre in genres)

        for genre in genre_set:
            movies[genre] = movies['genres'].apply(lambda x: int(genre in x))

        tags_aggregated = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()

        movies_features = movies.merge(tags_aggregated, on='movieId', how='left')
        movies_features['tag'] = movies_features['tag'].fillna('')

        tfidf = TfidfVectorizer(max_features=50)
        tfidf_features = tfidf.fit_transform(movies_features['tag']).toarray()

        tfidf_df = pd.DataFrame(tfidf_features, columns=[f'tag_{i}' for i in range(tfidf_features.shape[1])])
        movies_features = pd.concat([movies_features.reset_index(drop=True), tfidf_df], axis=1)

        movies_features = movies_features.drop(['genres', 'title', 'tag'], axis=1)

        return movies_features


class UserFeatureExtractor(FeatureExtractor):
    """
    Extracts user features from ratings and movies data.

    Attributes:
        None

    Methods:
        generate_features: Generates user features based on ratings and movies data.

    """
    def load_dataset(self, file_path: Path) -> pd.DataFrame:
        return pd.read_csv(file_path)

    def generate_features(self, ratings_path: Path, movies_path: Path) -> pd.DataFrame:
        """
        Generates user features based on ratings and movies data.

        Args:
            ratings (pd.DataFrame): DataFrame containing user ratings data.
            movies (pd.DataFrame): DataFrame containing movies data.

        Returns:
            pd.DataFrame: DataFrame containing user features.

        """
        ratings = self.load_dataset(ratings_path)
        movies = self.load_dataset(movies_path)
        user_ratings = ratings.groupby('userId')['rating'].agg(['mean', 'count']).reset_index()
        user_ratings.rename(columns={'mean': 'avg_rating', 'count': 'rating_count'}, inplace=True)

        user_genres = ratings.merge(movies[['movieId', 'genres']], on='movieId', how='left')

        user_genres['genres'] = user_genres['genres'].fillna('').str.split('|')
        genre_set = set(genre for genres in user_genres['genres'] for genre in genres)
        
        for genre in genre_set:
            user_genres[genre] = user_genres['genres'].apply(lambda x: int(genre in x))

        user_genre_preferences = user_genres.groupby('userId')[list(genre_set)].mean().reset_index()

        user_features = user_ratings.merge(user_genre_preferences, on='userId', how='left')

        return user_genres


class DataPreprocessor:
    def __init__(self, config: DataTransformationConfig ) -> None:
        self.movie_feature_extractor = MovieFeatureExtractor()
        self.user_feature_extractor = UserFeatureExtractor()
        self.config = config
    
    def load_dataset(self, file_path: Path) -> pd.DataFrame:
        return pd.read_csv(file_path)
    
    def process_data(self) -> pd.DataFrame:
        ratings_df = self.load_dataset(self.config.ratings)
        logger.info(f"Extracting Movies features...⏳")
        movies_featues = self.movie_feature_extractor.generate_features(self.config.movies, self.config.tags)
        logger.info(f"Extracting Movies features completed ✅ ")
        logger.info(f"Extracting User features...⏳")
        users_feature = self.user_feature_extractor.generate_features(self.config.ratings, self.config.movies)
        logger.info(f"Extracting User features completed ✅")
        logger.info(f"Merging features with ratings...⏳")
        ratings_with_users = ratings_df.merge(users_feature, on='userId', how='left')
        final_dataset = ratings_with_users.merge(movies_featues, on='userId', how='left')
        logger.info(f"Merging features with ratings completed ✅")
        logger.info(f"Final dataset: {final_dataset.head()}")

        return final_dataset

    def train_test_spliting(self, dataset: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        X = dataset.drop(['rating', 'userId', 'movieId'], axis=1)
        y = dataset['rating']

        X_tarin, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info(f"Training set size: {X_tarin.shape}")
        logger.info(f"Testing set size: {X_test.shape}")

        return X_tarin, X_test, y_train, y_test


