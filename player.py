import pandas as pd
import numpy as np
from models.env import Game
from models.experience import ExperienceReplay
from functs.functions import sharpe_ratio
import time

class Player(object):
    def __init__(self):
        # parameters
        self.debug = False
        self.epsilon_0 = .1
        self.num_actions = 3
        self.epoch = 300  # 11500
        self.max_memory = 100  # 10000
        self.max_game_len = 25  # 1000
        self.batch_size = 20  # 500
        self.lookback = 50
        self.START_IDX = 3000
        self.run_mode = 'random'

        self.force_close_position_at = 1755
        pass

    def init_game(self, df, prev: Game = None, debug=False):
        if prev is None:
            self.env = Game(df, look_back=self.lookback, max_game_len=self.max_game_len,
                            #init_idx=self.START_IDX if prev is None or self.run_mode == 'random' else prev.curr_idx+1,
                            init_idx=self.START_IDX,
                            run_mode=self.run_mode, debug=self.debug)
            self.env.observe(next_state=True)

        #self.env.reset()

        return self.env

    def run(self, df:pd.DataFrame, model_a, env_dim, model_b=None, learn=True, weights_file=None):
        '''
        position = 0 #flat
        position = 1 # long
        position = -1 #short

        action = 0 # do nothing
        action = 1 # sell
        action = 2 # buy

        :param df:
        :param model_a:
        :param exp_replay:
        :return:
        '''
        # collect stats from training
        stats = []
        exp_replay = ExperienceReplay(max_memory=self.max_memory)
        win_cnt = 0
        loss_cnt = 0
        wins = []
        losses = []
        pnls = []
        if self.debug: print('--------- START TRAINING ----------')
        for e in range(1, self.epoch+1):
            # control variables
            epsilon = self.epsilon_0 ** (np.log10(e)/1.3)
            game_over = False
            loss = 0
            cnt = 0
            entered_at = 0
            # set the game for each epoch
            #self.env = self.init_game(df, self.env)
            self.env = Game(df, look_back=self.lookback, max_game_len=self.max_game_len,
                            init_idx=self.START_IDX if self.env is None or self.run_mode == 'random' else self.env.curr_idx+1,
                            run_mode=self.run_mode, debug=self.debug)
            self.env.reset()
            input_t = self.env.observe()
            if self.debug: print('Player::run()', 'start epoch')
            while not game_over:
                start = time.time()
                cnt += 1
                input_tm1 = input_t

                #input_t = self.env.observe()
                # get next action
                # random action
                if np.random.rand() <= epsilon:  # not self.env.position or
                    action = np.random.randint(0, self.num_actions, size=1)[0]
                    #sets exit_action based on first movement in case of random access
                    #if self.env.position == 0 and action != 0: print('enter')

                    if self.env.position == 0:
                        if action == 2:  # buy
                            exit_action = 1  # sell
                        elif action == 1:  # sell
                            exit_action = 2  # buy
                else:  # non random action
                    if self.env.position == 0:  # flat
                        q = model_a.predict(input_tm1)
                        action = np.argmax(q[0])

                        if action == 2:  # buy
                            exit_action = 1  # sell
                        elif action == 1:  # sell
                            exit_action = 2  # buy
                    else:  # on market
                        q = model_a.predict(input_tm1)
                        action = np.argmax(q[0])

                # max length starts from market enter
                if entered_at == 0 and action != 0:
                    entered_at = cnt
                    if self.debug: print('Player::train() entered_at', entered_at)

                force_exit = False
                #if self.env.position and (cnt >= self.max_game_len+entered_at or self.env.curr_time.hour * 100 + self.env.curr_time.minute >= self.force_close_position_at ):
                if self.env.position and cnt >= self.max_game_len + entered_at:
                    #print('***Time Exit***')
                    action = exit_action
                    force_exit = True

                # apply action, get rewards and new state
                input_t, reward, game_over = self.env.act(action, force_exit=force_exit)
                #print('Player::train()', reward)
                if game_over and self.env.pnl > 0:
                    win_cnt += 1
                elif game_over and self.env.pnl <= 0:
                    loss_cnt += 1


                # train model
                if learn:
                    # print(len(exp_replay.memory))
                    models = [model_a] if model_b is None else (model_a, model_b)

                    inputs, targets = exp_replay.get_batch(models, env_dim=env_dim, num_actions=self.num_actions, batch_size=self.batch_size)
                    self.env.pnl_sum = sum(pnls)
                    zz = model_a.train_on_batch(inputs, targets)
                    loss += zz

                    if model_b is not None:
                        model_b.train_on_batch(inputs, targets)

                # store experience
                # if action or len(exp_replay.memory) < 20 or np.random.rand() < 0.1:
                if game_over or len(exp_replay.memory) < int(self.max_memory * .1) or np.random.rand() < 0.05:
                    if (np.sum(input_tm1) == np.sum(input_t)):
                        print('############# VERY STRANGE ###########')
                    # Implement prioritized memory
                    exp_replay.remember([input_tm1, action, reward, input_t], game_over, loss=(loss if learn else None))  #

                end = time.time()
                if self.debug: print('elapsed', cnt, end-start)


            prt_str = ("Epoch {:03d} | Loss {:.2f} | pos {} | len {} | at {} | reward {:.5f} | pnl {:.2f}% @ {:.2f}% | eps {:,.4f} | win {:04d} | loss {:04d} @{}||{}".format(
                e,
                loss,
                #zz,
                self.env.position,
                self.env.trade_len,
                entered_at,
                self.env.reward,
                sum(pnls) * 100,
                self.env.pnl * 100,
                epsilon,
                win_cnt,
                loss_cnt,
                #self.env.entry,
                self.env.start_time,
                self.env.curr_time
            ))
            stats.append({'loss': loss, 'pos': self.env.position, 'side': self.env.side, 'reward': self.env.reward, 'len': self.env.trade_len, 'cum_pnl': sum(pnls), 'cur_pnl': self.env.pnl, \
                      'win': 1 if reward > 0 else -1, 'entry': self.env.start_time, 'exit': self.env.curr_time})
            print(prt_str)
            # fid = open(fname, 'a')
            # fid.write(prt_str + '\n')
            # fid.close()
            pnls.append(self.env.pnl)

            if weights_file is not None and not e % 50:
                print('----saving weights-----')
                model_a.save_weights(weights_file, overwrite=True)

        return np.array(stats), model_a, exp_replay

    def test___(self):
        pass

    def stats(self, stats:np.ndarray):
        import matplotlib.pyplot as plt
        res = pd.DataFrame.from_records(stats)

        res.cum_pnl.plot()
        plt.title('cumulative returns')
        plt.grid()
        plt.show()

        plt.title('Sides')
        plt.hist(res.side, bins=3)
        plt.grid()
        plt.show()

        plt.title('Positions')
        plt.hist(res.pos, bins=3)
        plt.grid()
        plt.show()

        plt.figure(figsize=(14, 6))
        plt.title('Side vs Position over time')
        plt.plot(res.side, label='side')
        plt.plot(res.pos, label='position')
        plt.legend()
        plt.grid()
        plt.show()

        plt.title('Time distribution')
        plt.hist(res.len, bins=25)
        plt.grid()
        plt.show()

        plt.title('Sides vs Positions cumulative distribution')
        plt.plot(np.cumsum(res.pos), label='pos')
        plt.plot(np.cumsum(res.side), label='side')
        plt.legend()
        plt.grid()
        plt.show()

        plt.title('Returns distribution')
        plt.hist(res.cur_pnl, bins=100)
        plt.grid()
        plt.show()

        plt.title('Reward distribution')
        plt.hist(res.reward, bins=100)
        plt.grid()
        plt.show()

        plt.title('Losses')
        plt.plot(res.loss)
        plt.grid()
        plt.show()

        print('Sharpe ratio', sharpe_ratio(res.cur_pnl.values))
