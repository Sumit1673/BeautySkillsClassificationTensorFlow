import tensorflow as tf
from config import *
from network import BeautyModel
from dataset_creator import *
from trainer import Trainer


physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
   tf.config.experimental.set_memory_growth(physical_devices[0], True)


if __name__ == '__main__':
    config, _ = get_config()
    trainer = Trainer()

    if config.is_train:
        trainer.construct_model()
        trainer.train()
    else:
        test_df = pd.read_csv('inp_data_files/test_tf.csv')
        trainer.test(test_df, predict=False)
