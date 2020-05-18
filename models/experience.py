import numpy as np
from functs.functions import my_softmax

class ExperienceReplay(object):
    """
    This class gathers and delivers the experience
    """
    def __init__(self, max_memory=100, gamma=.9):
        self.debug = False
        self.max_memory = max_memory
        self.memory = list()
        self.memory_losses = np.array([])
        self.memory_idx = list()
        self.gamma = gamma

    def remember(self, states, game_over, loss=None):
        if self.debug: print("ExperienceReplay::remember()")
        # memory[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
        self.memory.append([states, game_over])
        if loss is not None:
            self.memory_losses = np.append(self.memory_losses, [loss]);
            self.memory_idx = np.argsort(self.memory_losses)
            self.memory_idx = np.flip(self.memory_idx)

        if len(self.memory) > self.max_memory:
            if len(self.memory_idx) > 0:
                self.memory_losses = np.delete(self.memory_losses, 0)

            del self.memory[0]

    def get_batch(self, models, env_dim: int, num_actions: int = 3, batch_size: int = 10):
        if self.debug: print("ExperienceReplay::get_batch()")
        len_memory = len(self.memory)
        #env_dim = self.memory[0][0][0].shape[1]
        #print('ExperienceReplay::get_batch()', env_dim)
        #num_actions = model.output_shape[-1]
        inputs = np.zeros((min(len_memory, batch_size), env_dim))
        targets = np.zeros((inputs.shape[0], num_actions))
        tmp_targets = np.zeros((inputs.shape[0], num_actions, len(models)))

        if len(self.memory_losses) > 0:
            _len = min(len_memory, batch_size)
            _enumerate = enumerate(self.memory_idx[:_len])
        else:
            _enumerate = enumerate(np.random.randint(0, len_memory, size=inputs.shape[0]))

        for i, idx in _enumerate:
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            game_over = self.memory[idx][1]
            inputs[i:i + 1] = state_t

            for j, model in enumerate(models):
                # There should be no target values for actions not taken.
                # Thou shalt not correct actions not taken #deep
                targets[i] = model.predict(state_t)
                if game_over:  # if game_over is True
                    tmp_targets[i, action_t, j] = reward_t
                else:
                    # reward_t + gamma * max_a' Q(s', a')
                    Q_sa = np.max(model.predict(state_tp1))
                    tmp_targets[i, action_t, j] = reward_t + self.gamma * Q_sa

            targets[i, action_t] = np.min(tmp_targets[i, action_t])

        return inputs, targets

    def _get_batch(self, model, num_actions: int = 3, batch_size=10):
        if self.debug: print("ExperienceReplay::get_batch()")
        len_memory = len(self.memory)
        num_actions = model.output_shape[-1]
        env_dim = self.memory[0][0][0].shape[1]
        inputs = np.zeros((min(len_memory, batch_size), env_dim))
        targets = np.zeros((inputs.shape[0], num_actions))

        for i, idx in enumerate(np.random.randint(0, len_memory, size=inputs.shape[0])):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            game_over = self.memory[idx][1]

            inputs[i:i + 1] = state_t
            # There should be no target values for actions not taken.
            # Thou shalt not correct actions not taken #deep
            targets[i] = model.predict(state_t)
            if game_over:  # if game_over is True
                targets[i, action_t] = reward_t
            else:
                # reward_t + gamma * max_a' Q(s', a')
                Q_sa = np.max(model.predict(state_tp1))
                targets[i, action_t] = reward_t + self.gamma * Q_sa
        return inputs, targets
