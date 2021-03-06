import numpy as np

class ExperienceReplay(object):
    '''This class gathers and delivers the experience'''

    def __init__(self, max_memory=100, discount=.9):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, states, game_over):
        # memory[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
        self.memory.append([states, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=10):
        len_memory = len(self.memory)
        num_actions = model.output_shape[-1]
        env_dim = self.memory[0][0][0].shape[1]
        inputs = np.zeros((np.minimum(len_memory, batch_size), env_dim))
        targets = np.zeros((inputs.shape[0], num_actions))
        for i, idx in enumerate(np.random.randint(0, len_memory, size=inputs.shape[0])):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            game_over = self.memory[idx][1]

            inputs[i:i + 1] = state_t
            # There should be no target values for actions not taken.
            # Thou shalt not correct actions not taken #deep
            # pred = model.predict(state_t)
            # print('ExperienceReplay::get_batch() 1', pred)

            targets[i] = model.predict(state_t)[0]
            # pred = model.predict(state_tp1)
            # print('ExperienceReplay::get_batch() 2', pred)
            Q_sa = np.max(model.predict(state_tp1)[0])
            if game_over:  # if game_over is True
                targets[i, action_t] = reward_t
            else:
                # reward_t + gamma * max_a' Q(s', a')
                # print('reward_t + gamma * max_a1 Q(s1, a1)', reward_t + self.discount * Q_sa)
                targets[i, action_t] = reward_t + self.discount * Q_sa
        return inputs, targets