import matplotlib.pyplot as plt
import pandas as pd
from loader.data import DataLoader
from player import Player

from models.env import Game
import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import SGD, Adam

from sklearn.svm import SVC, SVR

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)

def main(debug=False):
    loader = DataLoader('data.txt')
    df_full = loader.preprocess(start_hour=9, end_hour=17)

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
    player.epoch = 100
    # end set custom params for player

    _env = player.init_game(df_train)
    hidden_size = len(_env.state)*2
    model = Sequential()

    model.add(Dense(hidden_size, input_shape=(len(_env.state),), activation=relu))
    model.add(BatchNormalization())
    model.add(Dense(hidden_size, activation=relu))
    model.add(BatchNormalization())
    model.add(Dense(player.num_actions, activation=softmax))
    model.compile(Adam(lr=.005), MSE)
    print(model.summary())


    #model.load_weights('indicator_model__.h5')
    stats, model, exp = player.train(df_train, model=model)

    player.stats(stats)

    pass


if __name__ == "__main__":
    main()
