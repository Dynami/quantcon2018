from functs.functions import cycle, scale, lowPassFilter
import pandas as pd
import numpy as np
import talib
import traceback
from datetime import datetime, timedelta


class Game(object):
    '''This is the game. It starts, then takes an action (buy or sell) at some point and finally the reverse
    action, at which point it is game over. This is where the reward is given. The state consists of a vector
    with different bar sizes for OLHC. They are just concatenated.
    lkbk: determines how many bars to use - larger lkbk - bigger state
    '''

    def __init__(self, df, lkbk=20, max_game_len=1000, run_mode='sequential', init_idx=None):
        self.df = df
        self.lkbk = lkbk
        self.max_game_len = max_game_len

        self.is_over = False
        self.reward = 0
        self.run_mode = run_mode
        self.pnl_sum = 0
        if run_mode == 'sequential' and init_idx == None:
            print('------No init_idx set for "sequential": stopping------')
            return
        else:
            self.init_idx = init_idx
        self.reset()

    def _update_state(self, action):

        '''Here we update our state'''
        self.curr_idx += 1
        self.curr_time = self.df.index[self.curr_idx]
        self.curr_price = self.df['close'][self.curr_idx]
        self.pnl = 0 if self.entry == 0 else self.position * (self.curr_price - self.entry) / self.entry
        # if I can't calculate pnl set to zero
        if np.isnan(self.pnl):
            self.pnl = 0.0
        if(self.position):
            self.side = 1 if self.curr_price > self.entry else -1
        else:
            self.side = 0
        self._assemble_state()
        _h = self.curr_time.hour - 9  # starts at 9:00 ends at 18:00
        _m = self.curr_time.minute
        # _k = list(map(float,str(self.curr_time.time()).split(':')[:2]))
        # print('Game::_update_state() _k', _k)
        self._time_of_day = cycle(_h * 60 + _m, 9 * 60)
        self._day_of_week = cycle(self.curr_time.weekday(), 7)
        self.norm_epoch = (self.df.index[self.curr_idx] - self.df.index[0]).total_seconds() / self.t_in_secs

        '''This is where we define our policy and update our position'''
        #self.trade_len = self.curr_idx - self.start_idx
        if action == 0:
            pass

        elif action == 2:
            if self.position == -1:
                self.is_over = True
                self._get_reward()
                # self.trade_len = self.curr_idx - self.start_idx

            elif self.position == 0:
                self.position = 1
                self.entry = self.curr_price
                self.start_idx = self.curr_idx
            else:
                pass

        elif action == 1:
            if self.position == 1:
                self.is_over = True
                self._get_reward()
                # self.trade_len = self.curr_idx - self.start_idx

            elif self.position == 0:
                self.position = -1
                self.entry = self.curr_price
                self.start_idx = self.curr_idx
            else:
                pass

        if self.position:
            self.trade_len += 1


    def _assemble_state(self):
        '''Here we can add other things such as indicators and times'''
        self._get_last_N_timebars()
        bars = [self.last5m]  # , self.last1h, self.last1d
        state = []
        candles = {j: {k: np.array([]) for k in ['open', 'high', 'low', 'close']} for j in range(len(bars))}
        for j, bar in enumerate(bars):
            for col in ['open', 'high', 'low', 'close']:
                candles[j][col] = np.asarray(bar[col])
                state += (list(np.asarray(bar[col]))[-10:])

        state = scale(np.array(state))
        self.state = np.array([])
        self.state = np.append(self.state, scale(self.trade_len/self.max_game_len, min=0, max=1))
        # self.state = np.append(self.state, state)
        self.state = np.append(self.state, self.position)
        # self.state = np.append(self.state, np.sign(self.pnl_sum))
        #print("Game::_ensamble_state() pnl", self.pnl)
        self.state = np.append(self.state, scale(0 if np.isnan(self.pnl) else self.pnl, min=-0.02, max=0.02))
        self.state = np.append(self.state, self._time_of_day)
        self.state = np.append(self.state, self._day_of_week)

        # print('Game::_assemble_state() First block', self.state)
        self.lookback = 5
        for c in candles:
            try:

                f1 = lowPassFilter(candles[c]['close'], self.lkbk - 1)[-self.lookback:]
                f2 = lowPassFilter(candles[c]['close'], self.lkbk - 8)[-self.lookback:]
                tmp = scale((f2 - f1) / f1, -0.01, 0.01)
                # print('Game::_assemble_state() LowPass cross', tmp)
                self.state = np.append(self.state, tmp)

                # sma1 = talib.EMA(candles[c]['close'],self.lkbk-1)[-1]
                # sma2 = talib.EMA(candles[c]['close'],self.lkbk-8)[-1]
                # tmp = scale((sma1-sma2)/sma2, -0.2, 0.2)
                # self.state = np.append(self.state, tmp)

                # self.state = np.append(self.state,sma1)
                tmp = scale(talib.RSI(candles[c]['close'])[-self.lookback:], 0, 100)
                # print('Game::_assemble_state() RSI', tmp)
                self.state = np.append(self.state, tmp)

                tmp = scale(talib.MOM(candles[c]['close'])[-self.lookback:], -3, 3)
                # print('Game::_assemble_state() MON', tmp)
                self.state = np.append(self.state, tmp)
                ##self.state = np.append(self.state,talib.MACD(candles[c]['close'],fastperiod=11, slowperiod=22, signalperiod=9)[0][0])
                # self.state = np.append(self.state,talib.BOP(candles[c]['open'],
                #                               candles[c]['high'],
                #                               candles[c]['low'],
                #                               candles[c]['close'])[-1])
                ##self.state = np.append(self.state,talib.ADXR(candles[c]['high'],
                ##                               candles[c]['low'],
                ##                               candles[c]['close'],
                ##                               self.lkbk-3)[-1])
                ##self.state = np.append(self.state,talib.STOCH(candles[c]['high'],
                ##                               candles[c]['low'],
                ##                               candles[c]['close'],5,3,0,3,0)[-1][0])
                tmp = talib.AROONOSC(candles[c]['high'],
                                     candles[c]['low'])[-self.lookback:]
                # print('Game::_assemble_state() AROONOSC', tmp)
                tmp = scale(tmp, 0, 100)
                self.state = np.append(self.state, tmp)

                if(np.isnan(self.state).any()):
                    print("Error on data", self.state)

            except:
                print(traceback.format_exc())
        # print(np.min(self.state), np.max(self.state))
        # print('-->',self.state.shape)
        # self.state = (np.array(self.state)-np.mean(self.state))/np.std(self.state)

    def _get_last_N_timebars(self):
        '''The lengths of the time windows are currently hardcoded.'''
        # TODO: find better way to calculate window lengths
        wdw5m = 9
        #wdw1h = np.ceil(self.lkbk * 15 / 24.)
        #wdw1d = np.ceil(self.lkbk * 15)

        self.last5m = self.df[self.curr_time - timedelta(wdw5m):self.curr_time].iloc[-self.lkbk:]
        #self.last1h = self.bars1h[self.curr_time - timedelta(wdw1h):self.curr_time].iloc[-self.lkbk:]
        #self.last1d = self.bars1d[self.curr_time - timedelta(wdw1d):self.curr_time].iloc[-self.lkbk:]

        '''Making sure that window lengths are sufficient'''
        try:
            assert (len(self.last5m) == self.lkbk)
            #assert (len(self.last1h) == self.lkbk)
            # assert(len(self.last1d)==self.lkbk)
        except:
            print('****Window length too short****')
            print(len(self.last5m) ) #, len(self.last1h), len(self.last1d)
            if self.run_mode == 'sequential':
                self.init_idx = self.curr_idx
                self.reset()
            else:
                self.reset()

    def _original_get_reward(self):
        if self.position == 1 and self.is_over:
            pnl = (self.curr_price - self.entry) / self.entry
            self.reward = np.sign(pnl)  # -(self.curr_idx - self.start_idx)/1000.
        elif self.position == -1 and self.is_over:
            pnl = (-self.curr_price + self.entry) / self.entry
            self.reward = np.sign(pnl)  # -(self.curr_idx - self.start_idx)/1000.
        return self.reward

    def _get_reward(self):
        if self.position:
            pnl = self.position * (self.curr_price - self.entry) / self.entry
        else:
            pnl = 0

        pnl = pnl * 100
        dist = self.curr_idx - self.start_idx
        # factor = 1 if pnl > 0 else 0
        penalty = ((dist / self.max_game_len) ** 6) / 2.0
        if (pnl > 0):
            reward = pnl * (1 - penalty)
        else:
            reward = pnl * (1 + penalty)

        if self.position == 1 and self.is_over:
            self.reward = reward  # np.sign(pnl) - penalty
        elif self.position == -1 and self.is_over:
            self.reward = reward  # np.sign(pnl) - penalty
        # print('entry:',self.entry,'exit:',self.curr_price,'pos:',self.position,'pnl:',pnl,self.reward)
        # print('Game::_get_reward() Passed:', (self.curr_idx - self.start_idx), 'Reward:', self.reward ,\
        #      'Penalty:', penalty,'Position:', self.position, 'Entry:', self.entry, 'Exit:', self.curr_price)
        return self.reward

    def observe(self):
        return np.array([self.state])

    def act(self, action, force_exit=False):
        self._update_state(action)
        reward = self.reward
        game_over = self.is_over
        if force_exit:
            self.is_over = True
            game_over = True
        return self.observe(), reward, game_over

    def reset(self):
        self.pnl = 0
        self.entry = 0
        self._time_of_day = 0
        self._day_of_week = 0

        if self.run_mode == 'random':
            self.curr_idx = np.random.randint(0, len(self.df) - self.init_idx)

        elif self.run_mode == 'sequential':
            self.curr_idx = self.init_idx
        else:
            assert "Unhandled Run Mode ["+self.run_mode+"]"

        self.t_in_secs = (self.df.index[-1] - self.df.index[0]).total_seconds()
        self.start_idx = self.curr_idx
        self.curr_time = self.df.index[self.curr_idx]
        #self.bars1h = self.df['close'].resample('1H', label='right', closed='right').ohlc().dropna()
        #self.bars1d = self.df['close'].resample('1D', label='right', closed='right').ohlc().dropna()
        self._get_last_N_timebars()
        self.state = []
        self.position = 0
        self.trade_len = 0
        self._update_state(0)