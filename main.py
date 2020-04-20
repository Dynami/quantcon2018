import matplotlib.pyplot as plt
import pandas as pd
from loader.data import DataLoader
from player import Player
from models.sklearn import SkModel, MOSkModel
import numpy as np

from models.env import Game
import warnings

from sklearn.svm import SVC, SVR
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)

def main(debug=False):
    print('Start loading data')
    loader = DataLoader('data.txt')
    print('Loaded data')
    df_full = loader.preprocess(start_hour=9, end_hour=17)
    print('Data preprocessed')
    # select train data
    train_start_idx = 70000
    train_end_idx = 130000
    df_train = df_full.iloc[train_start_idx:train_end_idx]

    if debug:
        plt.plot(df_train.close)
        plt.grid()
        plt.show()

    player = Player()
    # start set custom params for player
    player.epoch = 200
    player.batch_size = 50
    player.run_mode = 'sequential'
    player.max_memory = 200
    player.max_game_len = 10
    player.debug = True
    player.START_IDX = 3000
    # end set custom params for player

    _env = player.init_game(df_train)
    use_tensorflow = False
    if use_tensorflow:
        # import tensorflow as tf
        # from tensorflow.keras.models import Sequential
        # from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
        # from tensorflow.keras.activations import relu, softmax, linear, tanh
        # from tensorflow.keras.losses import MSE
        # from tensorflow.keras.optimizers import SGD, Adam
        #
        # #tf.random.set_seed(42)
        #
        # hidden_size = len(_env.state)*2
        # model = Sequential()
        # model.add(Dense(hidden_size, input_shape=(len(_env.state),), activation=relu))
        # #model.add(BatchNormalization())
        # model.add(Dense(hidden_size, activation=relu))
        # #model.add(BatchNormalization())
        # model.add(Dense(player.num_actions))
        # model.compile(SGD(lr=.05), MSE)
        # print(model.summary())
        # #model.load_weights('indicator_model.h5')
        pass
    else:
        import torch
        hidden_size = len(_env.state) * 2
        D_in, H, D_out = len(_env.state), hidden_size, player.num_actions
        from models.torch import TorchModel
        model = TorchModel(D_in, H, D_out)
        #np.random.seed(42)
        #model = MOSkModel(SVR(), input_shape=len(_env.state), num_action=3)

    stats, model, exp = player.train(df_train, model=model)

    player.stats(stats)
    return stats, model, exp

if __name__ == "__main__":
    main()
