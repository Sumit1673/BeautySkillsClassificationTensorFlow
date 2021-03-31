import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from config import *
import tensorflow as tf

config, _ = get_config()


class DatasetCreator:
    def __init__(self):
        self.dataset_df = pd.read_csv(config.dataset_csv)
        self.dataset_df.sample(frac=1).reset_index(drop=True, inplace=True)
        self.labels = self.dataset_df.skill.values
        print(f'SKILLS: {self.labels},\nTotal: {len(self.labels)}')

        self._datagen = ImageDataGenerator(
            rescale=1. / 255.,
            validation_split=0.25,
            # featurewise_center=True,
            # featurewise_std_normalization=True,
            rotation_range=60,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=False
        )
        # self.test_train_dataset_split()

    def test_train_dataset_split(self):
        splits = StratifiedShuffleSplit(n_splits=1, random_state=42, test_size=0.2)

        for train_index, test_index in splits.split(self.dataset_df['file_path'], self.dataset_df['skill']):
            self.X_train, self.X_test = self.dataset_df.loc[train_index], self.dataset_df.loc[test_index]

        kf_splits = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        self.kf_dataset = {}
        for kf_id, (train_index, val_index) in enumerate(kf_splits.split(self.X_train['file_path'], self.X_train['skill'])):
            training_data = self.X_train.iloc[train_index]
            validation_data = self.X_train.iloc[val_index]
            if kf_id not in self.kf_dataset:
                self.kf_dataset[kf_id] = [training_data, validation_data]
            del training_data, validation_data
        # temp_df = pd.DataFrame(self.kf_dataset)
        # temp_df.to_csv('inp_data_files/indexes.csv')
        self.X_test.to_csv('inp_data_files/test_tf2.csv')

    def train_generator(self, data_df, subset):
        train_gen = self._datagen.flow_from_dataframe(
            dataframe=data_df,
            x_col="file_path",
            y_col='skill',
            subset=subset,
            batch_size=config.batch_size,
            seed=42,
            shuffle=True,

            # save_to_dir='augmented_imgs',
            class_mode="categorical",
            target_size=(config.image_dim, config.image_dim)
        )
        print('Train generator created')
        return train_gen

    def test_generator(self, test_df):
        _test_datagen = ImageDataGenerator(rescale=1. / 255.)
        test_gen = _test_datagen.flow_from_dataframe(
            dataframe=test_df,
            x_col="file_path",
            y_col='skill',
            class_mode="categorical",
            batch_size=1,
            seed=42,
            shuffle=False,
            target_size=(config.image_dim, config.image_dim)
        )
        print('Test generator created')
        return test_gen


