import numpy as np

class SkModel(object):
    def __init__(self, models, input_shape):
        print('SkModel()', input_shape)
        self.models = models
        self.num_action = len(self.models)
        self.output_shape = np.array([len(self.models)])

        init_x_data = np.random.randn(input_shape).reshape(1, -1)
        init_y_data = np.random.randn(3).reshape(1, -1)

        self.train_on_batch(init_x_data, init_y_data)

        pass;

    def predict(self, x_data):
        out = np.zeros((self.num_action))
        for i in range(self.num_action):
            out[i] = self.models[i].predict(x_data)

        return out

    def train_on_batch(self, x_data, y_data):
        loss = 0
        for i in range(self.num_action):
            self.models[i].fit(x_data, y_data[:, i])
            loss += self.models[i].score(x_data, y_data[:, i])

        return loss

    def save_weights(self, output, overwrite=False):
        pass