import os
import pandas as pd
from tensorflow.keras import layers, Model
import tensorflow as tf
import numpy as np
from typing import Tuple, List
import joblib
from mlProject.entity.config_entity import ModelTrainerConfig
from mlProject import logger



# model

class RecommenderNet(Model):
    def __init__(self, user_shape: int, movie_shape: int):
        super(RecommenderNet, self).__init__()
        
        # User tower layers
        self.user_input = tf.keras.layers.Input(shape=(user_shape,), name="user_input")
        self.user_NN = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16)
        ])
        
        # Movie tower layers
        self.movie_input = tf.keras.layers.Input(shape=(movie_shape,), name="movie_input")
        self.movie_NN = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16)
        ])
        
        # Combined layers
        self.output_layer = tf.keras.layers.Dot(axes=1)
        
    def call(self, inputs):
        user_input, movie_input = inputs
        
        # User tower
        vu = self.user_NN(user_input)
        vu = tf.linalg.l2_normalize(vu, axis=1)
        
        # Movie tower
        vm = self.movie_NN(movie_input)
        vm = tf.linalg.l2_normalize(vm, axis=1)
        
        return self.output_layer([vu, vm])
    
    def build_graph(self):
        """Create model graph for visualization"""
        model = Model(
            inputs=[self.user_input, self.movie_input],
            outputs=self.call([self.user_input, self.movie_input])
        )
        logger.info(f"Model summary: {model.summary()}")
        return model


class RecommenderTrainer:
    def __init__(
        self,
        config: ModelTrainerConfig,
        user_shape: int,
        movie_shape: int,
        # learning_rate: float = 0.001,
        # batch_size: int = 32,
        # epochs: int = 10
    ):
        self.model = RecommenderNet(user_shape, movie_shape)
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.history = None
        self.cost_fn = tf.keras.losses.MeanSquaredError()
        self.validation_split = config.validation_split
        
    def compile_model(self):
        """Compile the model with specified parameters"""
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss= self.cost_fn,
            metrics=['mae', 'mse']
        )
        
    def train(
        self,
        X_user: np.ndarray,
        X_movie: np.ndarray,
        y: np.ndarray,
        callbacks: List = None
    ):
        """Train the model"""
        validation_split = self.validation_split
        logger.info("Compiling recommender model...⏳")
        self.compile_model()
        logger.info("Compiling model \033[34mcompleted\033[0m✅")
        logger.info("Begin training has begun...⏳")
        self.history = self.model.fit(
            [X_user, X_movie],
            y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.validation_split,
            callbacks=callbacks
        )
        logger.info("Model training has \033[34mcompleted\033[0m✅")
        return self.model, self.history
    
    def predict(self, X_user: np.ndarray, X_movie: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict([X_user, X_movie])
    
    def evaluate(
        self,
        X_user: np.ndarray,
        X_movie: np.ndarray,
        y: np.ndarray
    ) -> Tuple[float, float]:
        """Evaluate the model"""
        return self.model.evaluate([X_user, X_movie], y)
    
    def save_model(self, path: str):
        """Save the model"""
        self.model.save(path)
    
    @staticmethod
    def load_model(path: str):
        """Load a saved model"""
        return tf.keras.models.load_model(path)
