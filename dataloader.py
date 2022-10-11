from email.mime import image
import numpy as np
import os
import pickle
import keras_cv
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
import matplotlib.pyplot as plt
from absl import logging
import re

from vit_keras import vit
np.set_printoptions(precision=4)

class Transfer_Model(tf.keras.Model):
    def __init__(self, vit_image_shape, num_classes):
        super(Transfer_Model, self).__init__()
        self.vit_image_shape = vit_image_shape
        self.num_classes = num_classes
        
        self.model = vit.vit_b16(image_size=self.vit_image_shape, activation='sigmoid', pretrained=True, include_top=False, pretrained_top=False)
        
    def freeze_model(self, model, limit=  None):
    # handle negative indices
        if limit != None and limit < -1:
            limit = limit + len(model.layers) 
        # loop for all valid indices and mark the corresponding layer
        for index, layer in enumerate(model.layers):
            if limit != None and index > limit:
                break
            layer.trainable = False

    def unfreeze_model(self, model):
        for index, layer in enumerate(model.layers):
            layer.trainable = True
    
    def new_model(self):
        self.freeze_model(self.model)
        x = self.model.output                         # shape should be (bs=None, 7, 7, 2048)
        x = Dropout(0.3)(x)     # shape should be (bs=None, 2048)
        x = Dense(512, activation='swish')(x) # shape should be (bs=None, 1024)
        x = BatchNormalization()(x)
        y = Dense(self.num_classes, activation='softmax')(x) # shape should be (bs=None, 37)
        model2 = Model(inputs=self.model.input, outputs=y)
        return model2
    
    def fine_model(self):
        self.unfreeze_model(self.model)
        
    def test(self, x, top_k = 5):
        """_summary_

        Args:
            x (_type_): ndarray 3d shape cannel last
        """
        out = self.model(x)
        
        # index, out_prob = np.sort(out.numpy())[:top_k]
        # lables_description = np.array(['a','b','c'])
        
        # return lables_description[index], out_prob
        return out
        
        
labels = [
    'abyssinian', 'american_bulldog', 'american_pit_bull_terrier', 'basset_hound',
    'beagle', 'bengal', 'birman', 'bombay', 'boxer', 'british_shorthair', 'chihuahua',
    'egyptian_mau', 'english_cocker_spaniel', 'english_setter', 'german_shorthaired',
    'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond', 'leonberger', 'maine_coon',
    'miniature_pinscher', 'newfoundland', 'persian', 'pomeranian', 'pug', 'ragdoll',
    'russian_blue', 'saint_bernard', 'samoyed', 'scottish_terrier', 'shiba_inu', 'siamese',
    'sphynx', 'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier',
]

def load_OxfordPet(batch_size, shuffle_buffer_size, input_shape):
    TRAIN, TEST = tfds.load(name="oxford_iiit_pet", split=['train', 'test'], as_supervised=True)
    
    def preprocessDataset(image, label):
        image = tf.cast(image, tf.float32)/255.0
        image = tf.image.resize(image, input_shape)
        # image = tf.image.random_crop(image, size=(224, 224, 3))
        return image, label
    
    def preprocessDataset1(image, label):
        # label = tf.cast(label, tf.float32)
        image = tf.cast(image, tf.float32)/255.0
        image = tf.image.resize(image, input_shape)
        # image = tf.image.central_crop(image, 224/384)
        return image, label

    train_batches = TRAIN.map(preprocessDataset).shuffle(shuffle_buffer_size).batch(batch_size)
    test_batches = TEST.map(preprocessDataset1).shuffle(shuffle_buffer_size).batch(batch_size)
    
    return train_batches, test_batches