from keras.callbacks import Callback


class TrainingCallback(Callback):
    def on_train_begin(self, logs=None):
        print('Starting training....')

    def on_epoch_begin(self, epoch, logs=None):
        print('Starting epoch {}'.format(epoch))

    def on_train_batch_begin(self, batch, logs=None):
        print('Training: Starting batch {}'.format(batch))

    def on_train_batch_end(self, batch, logs=None):
        print('Training: Finished batch {}'.format(batch))

    def on_epoch_end(self, epoch, logs=None):
        print('Finished epoch {}'.format(epoch))

    def on_train_end(self, logs=None):
        print('Finished training!')


class TestingCallback(Callback):
    def on_test_begin(self, logs=None):
        print('Starting testing....')

    def on_test_batch_begin(self, batch, logs=None):
        print('Testing: Starting batch {}'.format(batch))

    def on_test_batch_end(self, batch, logs=None):
        print('Testing: Finished batch {}'.format(batch))

    def on_test_end(self, logs=None):
        print('Finished testing!')


class PredictionCallback(Callback):
    def on_predict_begin(self, logs=None):
        print('Prediction testing....')

    def on_predict_batch_begin(self, batch, logs=None):
        print('Prediction: Starting batch {}'.format(batch))

    def on_predict_batch_end(self, batch, logs=None):
        print('Prediction: Finished batch {}'.format(batch))

    def on_predict_end(self, logs=None):
        print('Finished prediction!')