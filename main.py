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

def run(debug=False):
    train = 0

    print('Start loading data')
    loader = DataLoader('data.txt')
    print('Loaded data')
    #df_full = loader.resample()
    df_full = loader.preprocess(start_hour=7, end_hour=18)
    print('Data preprocessed', df_full.shape)
    # select train data
    if train:
        start_idx = 3000
        end_idx = 30000 # 500000
    else:
        start_idx = 200000
        end_idx = 500000
    df = df_full.iloc[start_idx:end_idx]
    weight_file = 'torch_model.pt1'
    if debug or not train:
        plt.plot(df.close)
        plt.grid()
        plt.show()

    player = Player()
    # start set custom params for player
    if train:
        player.epoch = 1000
    else:
        player.epoch = 5000
    player.batch_size = 30
    player.n_last_bars_in_state = 5
    player.lookback = 50

    if train:
        player.run_mode = 'random'
    else:
        player.run_mode = 'sequential'

    player.max_memory = 1000
    player.max_game_len = 12
    player.debug = False
    player.START_IDX = 1000
    # end set custom params for player

    _env = player.init_game(df)

    hidden_size = len(_env.state) * 2
    D_in, H, D_out = len(_env.state), hidden_size, player.num_actions
    model = TorchModel(D_in, H, D_out)

    if train:
        stats, model, exp = player.run(df, model=model, weights_file=weight_file)
    else:
        model.load_weights(weight_file)
        stats, model, exp = player.run(df, model=model, weights_file=None, learn=False)

    #model.save_weights(output=weight_file)

    player.stats(stats)
    return stats, model, exp


if __name__ == "__main__":
    stats, model, exp = run(debug=False)
    # test(model=None)


