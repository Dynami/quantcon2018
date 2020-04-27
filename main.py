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
from models.torch import TorchModel
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)

def train(debug=False):
    print('Start loading data')
    loader = DataLoader('data.txt')
    print('Loaded data')
    df_full = loader.resample()
    #df_full = loader.preprocess(start_hour=7, end_hour=19)
    print('Data preprocessed')
    # select train data
    train_start_idx = 1000
    train_end_idx = 30000
    df_train = df_full.iloc[train_start_idx:train_end_idx]
    weight_file = 'torch_model.pt1'
    if debug:
        plt.plot(df_train.close)
        plt.grid()
        plt.show()

    player = Player()
    # start set custom params for player
    player.epoch = 100
    player.batch_size = 30
    player.run_mode = 'random'
    player.max_memory = 200
    player.max_game_len = 12
    player.debug = False
    player.START_IDX = 1000
    # end set custom params for player

    _env = player.init_game(df_train)

    hidden_size = len(_env.state) * 2
    D_in, H, D_out = len(_env.state), hidden_size, player.num_actions
    model = TorchModel(D_in, H, D_out)

    #model.load_weights(weight_file)

    stats, model, exp = player.run(df_train, model=model, weights_file=weight_file)
    model.save_weights(output=weight_file)

    player.stats(stats)
    return stats, model, exp

# def test(model=None, debug=False):
#     print('Start loading data')
#     loader = DataLoader('data.txt')
#     print('Loaded data')
#     df_full = loader.preprocess(start_hour=9, end_hour=17)
#     print('Data preprocessed')
#     # select train data
#     test_start_idx = 130000
#     df_test = df_full.iloc[test_start_idx:]
#
#
#
#     if debug:
#         plt.plot(df_test.close)
#         plt.grid()
#         plt.show()
#
#     player = Player()
#     # start set custom params for player
#     player.epoch = 10
#     player.batch_size = 50
#     player.run_mode = 'sequential'
#     player.max_memory = 200
#     player.max_game_len = 15
#     player.debug = False
#     player.START_IDX = 3000
#     # end set custom params for player
#
#     _env = player.init_game(df_test)
#
#     hidden_size = len(_env.state) * 2
#     D_in, H, D_out = len(_env.state), hidden_size, player.num_actions
#     if model is None:
#         model = TorchModel(D_in, H, D_out)
#
#     #model.load_weights(weight_file)
#
#     stats, model, exp = player.run(df_test, model=model, learn=False, weights_file=weight_file)
#
#     player.stats(stats)
#     return stats, model, exp

if __name__ == "__main__":
    stats, model, exp = train()
    # test(model=None)


