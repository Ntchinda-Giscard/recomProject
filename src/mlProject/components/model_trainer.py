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