import numpy as np
np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
np.set_printoptions(precision=4)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import ShuffleSplit
import argparse
import os
from utils import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers

look_back = 2

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=self.projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches,
            output_dim=self.projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'num_patches': self.num_patches,
            'projection_dim': self.projection_dim
        })
        return config

def FFN(x, hidden_units):
    #Section 3.3 in "Attention in All You Need"
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
    return x

def Transformer(input_shape, projection_dim, num_heads, n_classes):

    inputs = layers.Input(shape=input_shape)
    encoded_patches = PatchEncoder(input_shape[0], projection_dim)(inputs)

    num_transformer_blocks = len(num_heads)
    for i in range(num_transformer_blocks):

        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(num_heads=num_heads[i], key_dim=projection_dim, dropout=0.0)(x1, x1)

        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)

        # Section 3.3 in "Attention in All You Need"
        #x3 = FFN(x3, hidden_units=transformer_units)
        x3 = layers.Dense(projection_dim * 2, activation=tf.nn.gelu)(x3)
        x3 = layers.Dense(projection_dim, activation=tf.nn.gelu)(x3)

        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])


    encoded_patches = layers.Flatten()(encoded_patches)
    outputs = layers.Dense(n_classes)(encoded_patches)

    return keras.Model(inputs, outputs)

def testing(model, wells, test_idx):
    mse = np.zeros((len(test_idx)))
    r2 = np.zeros(len(test_idx))
    rms = np.zeros(len(test_idx))
    for i, idx in enumerate(test_idx):
        well_name = wells[idx]['name']
        logs = wells[idx]['logs']
        ajus = wells[idx]['ajus']
        sound = wells[idx]['sound']

        # Prepares a single sample
        X_test = np.column_stack((sound[:, 1], logs[:, 1]))
        y_test = ajus[:, 1]

        unormalized_X = np.copy(X_test)

        X_test = (X_test - x_train_mean) / x_train_std

        padding = np.tile(X_test[-1], (look_back + 1, 1))
        X_test = np.vstack([X_test, padding])

        X_test = time_step_data(X_test, y=None, look_back=look_back)

        y_pred = model.predict(X_test)
        y_pred = (y_pred * y_train_std) + y_train_mean
        mse[i] = mean_squared_error(y_test, y_pred)
        r2[i] = r2_score(y_test, y_pred)
        rms[i] = mean_squared_error(y_test, y_pred, squared=False)
        #print('Well [{}] RMSE [{:.2f}] R2 [{:.2f}] MSE [{:.2f}] '.format(well_name, rms[i], r2[i], mse[i]))
        #visualization(y_pred, y_test, file_name=well_name, save_graph=well_name, depth=sound[:, 0])

    print('Mean RMSE [{:.2f}] R2 [{:.4f}] MSE [{:.2f}] '.format(np.average(rms), np.average(r2), np.average(mse)))

def hold_out(wells):
    same = []
    for i, well in enumerate(wells):
        _, ajus, sond = well['logs'], well['ajus'], well['sound']
        if np.isclose(ajus[:, 1], sond[:, 1]).all():
            same.append(i)
    same = np.array(same)
    all_idx = np.arange(len(wells))

    not_same = np.delete(all_idx, same)
    np.random.shuffle(not_same)

    train_idx = np.random.choice(not_same, int(0.7 * len(not_same)), replace=False)
    test_idx = np.array([i for i in not_same if i not in train_idx])

    #Now we get the samples where sond and ajus are the same
    tmp = np.random.choice(same, int(0.7 * len(same)), replace=False)
    train_idx = np.concatenate((train_idx, tmp))
    tmp = np.array([i for i in same if i not in tmp])
    test_idx = np.concatenate((test_idx, tmp))

    for i in train_idx:
        for j in test_idx:
            if i == j:
                print('Train test with equal samples')
                exit(-1)

    return train_idx, test_idx


def get_flops(model):
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function([tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)
    return flops.total_float_ops

if __name__ == '__main__':
    np.random.seed(12227)
    tf.random.set_seed(12227)

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=str, default='GRU') #Options: 'RNN', 'LSTM', 'GRU', 'TRAN'

    args = parser.parse_args()
    method = args.c
    bs = 1024
    ep = 500
    if method == 'TRAN':
        look_back = 4
        #bs = 32
        #ep = 100

    wells = np.load('data.npy', allow_pickle=True)
    wells = np.delete(wells, -1)#Removes the well number 100 (with bugs in ground-truth)

    for i, well in enumerate(wells):
        logs, ajus, sound = well['logs'], well['ajus'], well['sound']

        x = logs[:, 0]
        y = logs[:, 1]
        f = interp1d(x, y, fill_value='extrapolate')

        interpolated_logs = np.zeros((sound.shape))
        interpolated_logs[:, 0] = sound[:, 0]
        interpolated_logs[:, 1] = f(sound[:, 0])
        #Update the wells logs
        wells[i]['logs'] = interpolated_logs

    #Split train and test
    train_idx, test_idx = hold_out(wells)

    #We execute the normalization before runing time_step_data
    x_train_mean, x_train_std, y_train_mean, y_train_std = ZScore(wells[train_idx])

    X_train, y_train = [], []

    for idx in train_idx:
        logs, ajus, sound = wells[idx]['logs'], wells[idx]['ajus'], wells[idx]['sound']

        #Prepares a single sample
        X = np.column_stack((sound[:, 1], logs[:, 1]))
        y = np.expand_dims(ajus[:, 1], axis=1)

        X = (X-x_train_mean)/x_train_std
        y = (y-y_train_mean)/y_train_std

        X, y = time_step_data(X, y, look_back)

        X_train.append(X)
        y_train.append(y)

    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    if method == 'TRAN':
        model = Transformer(X_train.shape[1:], projection_dim=8, num_heads=[16, 16], n_classes=1)
        y_train = y_train[:, 0, :]  # 0 means the current

    else:
        model = Sequential()

        if method == 'RNN':
            model.add(SimpleRNN(16, input_shape=(look_back, 2)))
        if method == 'GRU':
            model.add(GRU(64, input_shape=(look_back, 2), return_sequences=False))
        if method == 'LSTM':
            model.add(LSTM(32, input_shape=(look_back, 2), return_sequences=False))  # True when using multiple layers

        model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='Adam')
    print("Parameters [{}] The FLOPs is:{}".format(model.count_params(), get_flops(model)), flush=True)
    #Uncomment the next three lines for training the models by yourself
    #model.fit(X_train, y_train, batch_size=bs, verbose=2, epochs=ep)
    #model.load_weights('models/Transformer.h5')
    #save_deep_learning_model(method, model)

    model = load_deep_learning_model('weights/'+method, 'weights/'+method)
    testing(model, wells, test_idx)