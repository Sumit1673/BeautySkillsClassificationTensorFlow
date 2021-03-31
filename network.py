import tensorflow as tf
from config import *
from dataset_creator import *


class BeautyModel:
    def __init__(self, generator):
        self.base_model = tf.keras.applications.VGG16(input_shape=(224, 224, 3),
                                                    include_top=False,
                                                    weights='imagenet')
        preprocess_input = tf.keras.applications.vgg16.preprocess_input
        image_batch, label_batch = next(iter(generator))
        feature_batch = self.base_model(image_batch)
        print(feature_batch.shape)
        self.base_model.trainable = True
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        feature_batch_average = global_average_layer(feature_batch)
        prediction_layer = tf.keras.layers.Dense(19, activation='softmax')
        prediction_batch = prediction_layer(feature_batch_average)

        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = preprocess_input(inputs)
        x = self.base_model(x, training=True)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = global_average_layer(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = prediction_layer(x)
        model = tf.keras.Model(inputs, outputs)

