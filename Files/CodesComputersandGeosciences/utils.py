import numpy as np
import os
import platform
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.keras import layers

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

def ZScore(wells_train):
    X_train, y_train = None, None

    for well in wells_train:

        #_, logs, ajus, sound = well  # name, interpolated_logs, ajus, sound
        logs = well['logs']
        ajus = well['ajus']
        sound = well['sound']

        # Prepares a single sample
        X = np.column_stack((sound[:, 1], logs[:, 1]))
        y = ajus[:, 1]
        y = np.expand_dims(y, axis=1)

        if X_train is None:
            X_train, y_train = X, y
        else:
            X_train, y_train = np.vstack((X_train, X)), np.vstack((y_train, y))

    x_train_mean = np.mean(X_train, axis=0)
    x_train_std = np.std(X_train, axis=0)

    y_train_mean = np.mean(y_train, axis=0)
    y_train_std = np.std(y_train, axis=0)
    return x_train_mean, x_train_std, y_train_mean, y_train_std

def continuous_data(X, Y):
    m = 0
    for i, input_i in enumerate(X):
        m = m + input_i.shape[0]

    X_tmp = np.zeros((m, 2))
    y_tmp = np.zeros((m, 1))

    init = 0
    for x, y in zip(X, Y):  # i, input_i in enumerate(X):

        X_tmp[init:init + x.shape[0]] = x
        y_tmp[init:init + x.shape[0]] = np.array(y).reshape(-1, 1)
        init = init + x.shape[0]

    return X_tmp, y_tmp

def time_step_data(X,y=None, look_back=3):
    X_tmp, y_tmp = [], []
    if y is not None:
        for i in range(len(X)-look_back-1):
            a = X[i:(i+look_back)]
            X_tmp.append(a)
            y_tmp.append(y[i:(i+look_back)])
        return np.array(X_tmp), np.array(y_tmp)
    else:
        for i in range(len(X)-look_back-1):
            a = X[i:(i+look_back)]
            X_tmp.append(a)
        return np.array(X_tmp)

def visualization(y_pred, y_test, depth, unormalized_X=None, file_name='', save_graph=''):
    import matplotlib.pyplot as plt
    plt.rcParams["figure.figsize"] = (20, 5)
    plt.rcParams['figure.dpi'] = 300# or 300

    mse = mean_squared_error(y_test, y_pred)
    rms = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    plt.plot(depth, y_pred, 'k--', label='Ajuste Predito')
    plt.plot(depth, y_test, 'k-', label='Ajuste Geologo')

    if unormalized_X is not None:
        plt.plot(depth, unormalized_X[:, 0], 'b-', label='Sond (CG)')
        plt.plot(depth, unormalized_X[:, 1], 'r-', label='Log (GR)')

    plt.title('Poço [{}] RMSE = {:.2f} R2 = {:.2f} MSE = {:.2f} '.format(file_name, rms, r2, mse))
    plt.xlabel('Profundidade')
    plt.ylabel('API')
    plt.legend()

    if save_graph != '':

        if not os.path.isdir(save_graph):
            os.makedirs(save_graph)

        plt.savefig('{}/{}.png'.format(save_graph, file_name), dpi=600)
    plt.show()
    plt.clf()

def save_deep_learning_model(file_name='', model=None):
    import tensorflow.keras as keras
    model.save_weights(file_name + '.h5')
    with open(file_name + '.json', 'w') as f:
        f.write(model.to_json())

def load_deep_learning_model(architecture_file='', weights_file=''):
    import tensorflow.keras as keras
    from tensorflow.keras import layers
    from tensorflow.keras import backend as K

    def _hard_swish(x):
        return x * K.relu(x + 3.0, max_value=6.0) / 6.0

    def _relu6(x):
        return K.relu(x, max_value=6)

    if '.json' not in architecture_file:
        architecture_file = architecture_file+'.json'

    with open(architecture_file, 'r') as f:
        with keras.utils.CustomObjectScope({'relu6': _relu6,
                                'DepthwiseConv2D': layers.DepthwiseConv2D,
                               '_hard_swish': _hard_swish},
                              {'PatchEncoder': PatchEncoder}):
            model = keras.models.model_from_json(f.read())

    if '.h5' not in weights_file:
        weights_file = weights_file + '.h5'
        model.load_weights(weights_file)

    return model

def load_xls(file_name):
    output = []
    data = pd.ExcelFile(file_name)

    well_names = data.parse(data.sheet_names.index('Logs')).columns[1:]
    well_data = []

    for name in well_names:
        logs = data.parse(data.sheet_names.index('Logs'))[['DEPT', name]].dropna().drop_duplicates(subset='DEPT',
                                                                                                   keep='first')

        sound = data.parse(data.sheet_names.index('CG_sond'))[[name, name.replace('P', 'CG')]].dropna().drop_duplicates(
            subset=name,
            keep='first')

        ajus = data.parse(data.sheet_names.index('CG_ajus'))[[name, name.replace('P', 'CG')]].dropna().drop_duplicates(
            subset=name,
            keep='first')

        # astype(float) avoids bugs in np.sqrt and linear interpolation
        if int(platform.python_version().split('.')[1]) <= 6:
            logs = logs.get_values().astype(float)
            ajus = ajus.get_values().astype(float)
            sound = sound.get_values().astype(float)
        else:#<>.get_values do not work on Python 3.7 or higher
            logs = logs.to_numpy().astype(float)
            ajus = ajus.to_numpy().astype(float)
            sound = sound.to_numpy().astype(float)

        well_data.append((name, logs, ajus, sound))

    for well in well_data:
        name, logs, ajus, sound = well
        max_ = np.amax(ajus, axis=0)[0]
        min_ = np.amin(ajus, axis=0)[0]

        logs = logs[np.where(logs[:, 0] <= max_)[0]]
        logs = logs[np.where(logs[:, 0] >= min_)[0]]

        sound = sound[np.where(sound[:, 0] <= max_)[0]]
        sound = sound[np.where(sound[:, 0] >= min_)[0]]

        # It makes the maching between the (depth) sound and ajust
        index = []
        for i in range(0, sound.shape[0]):
            for j in range(0, ajus.shape[0]):
                if sound[i][0] == ajus[j][0]:
                    index.append((i, j))

        index = np.array(index)
        sound = sound[index[:, 0]]
        ajus = ajus[index[:, 1]]

        tmp = {}
        tmp['name'] = name
        tmp['logs'] = logs
        tmp['ajus'] = ajus
        tmp['sound'] = sound
        output.append(tmp)
    return output
