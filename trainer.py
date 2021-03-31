from dataset_creator import *
from network import *
from config import *
import json
import matplotlib.pyplot as plt


def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1 ** (epoch / s)

    return exponential_decay_fn


class Trainer:
    def __init__(self):
        self.config, _ = get_config()
        self.dataset_creator_obj = DatasetCreator()

    def construct_model(self):
        self.dataset_creator_obj.test_train_dataset_split()
        one_traindataset = self.dataset_creator_obj.train_generator(
            self.dataset_creator_obj.kf_dataset[0][0], subset='training')
        self.dataset = self.dataset_creator_obj.kf_dataset

        # create model
        model_obj = BeautyModel(one_traindataset)
        self.model = model_obj.model
        print(self.model.summary())

    def train(self):
        # callbacks
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.2,
                                                         patience=5, min_lr=0.0000)
        exponential_decay_fn = exponential_decay(0.01, 20)

        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)
        early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10,
                                                             restore_best_weights=True)
        # compile
        self.model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                      optimizer=tf.keras.optimizers.Adadelta(lr=5e-5),
                      metrics=['accuracy'])

        results = []
        for i_fold, data in self.dataset.items():
            train_data, valid_data = data[0], data[1]

            train_steps_epoch = len(train_data) // self.config.batch_size
            valid_steps_epoch = len(valid_data) // self.config.batch_size

            tr_generator = self.dataset_creator_obj.train_generator(data_df=train_data, subset='training')

            val_generator = self.dataset_creator_obj.train_generator(data_df=valid_data, subset='validation')

            chkpt_path = os.path.join(self.config.model_path, 'model_' + str(i_fold) + '.h5')

            model_chkpt = tf.keras.callbacks.ModelCheckpoint(filepath=chkpt_path, monitor='val_accuracy',
                                                             save_best_only=True, verbose=1)
            callback_list = [lr_scheduler, early_stopping_cb, model_chkpt]

            history = self.model.fit(
                tr_generator,
                batch_size=config.batch_size,
                epochs=self.config.epochs,
                # steps_per_epoch=train_steps_epoch,
                validation_data=val_generator,
                # validation_steps=valid_steps_epoch,
                callbacks=callback_list,
                verbose=1)

            np.save('evaluation/29mar_history' + str(i_fold) + '.npy', history.history)
            results.append(history)

    def test(self, test_df, predict=True, verbose=False):
        models = [os.path.join(self.config.model_path, x) for x in os.listdir(self.config.model_path)]

        test_gen = self.dataset_creator_obj.test_generator(test_df)
        for id, i_model in enumerate(models):
            # Recreate the exact same model, including its weights and the optimizer
            new_model = tf.keras.models.load_model(i_model)

            # Show the model architecture
            if verbose:
                new_model.summary()
            if predict:
                for img, label in test_gen:
                    plt.imshow(img[0, :, :, :])
                    result = new_model.predict(img)
                    conf = np.amax(result)
                    label = np.unique(self.dataset_creator_obj.labels)[result.argmax(-1)[0]]
                    print(conf, label)
            else:
                loss, acc = new_model.evaluate_generator(test_gen, verbose=2)
                print('Restored model {}, accuracy: {:5.2f}%'.format(id, 100 * acc))

    def plot_history(self):
        models = [os.path.join(self.config.model_path, x) for x in os.listdir(self.config.model_path)]

        for id, i_model in enumerate(models):
            tf.keras.utils.plot_model(
                i_model,
                to_file="evaluation/model" +str(id) + ".png",
                show_shapes=False,
                show_dtype=False,
                show_layer_names=True,
                rankdir="TB",
                expand_nested=False,
                dpi=96,
            )
