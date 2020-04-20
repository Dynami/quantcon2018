import numpy as np
from functs.functions import my_softmax
from sklearn.multioutput import MultiOutputRegressor

class MOSkModel(object):
    def __init__(self, model, input_shape, num_action):
        print('MOSkModel()', )
        self.model = MultiOutputRegressor(model)

        self.debug = False
        self.output_shape = np.array([num_action])
        self.input_shape = input_shape

        init_x_data = np.random.randn(input_shape).reshape(1, -1)
        init_y_data = np.random.randn(num_action).reshape(1, -1)

        self.train_on_batch(init_x_data, init_y_data)

        pass

    def predict(self, x_data):
        out = self.model.predict(x_data)
        #print('MOSkModel::predict()', out)
        return out

    def train_on_batch(self, x_data, y_data):
        self.model.fit(x_data, y_data)

        loss = self.model.score(x_data, y_data)
        return loss

    def save_weights(self, output, overwrite=False):
        pass

class SkModel(object):
    def __init__(self, models, input_shape):
        print('SkModel()', input_shape)
        self.debug = False
        self.models = models
        self.num_action = len(self.models)
        self.output_shape = np.array([len(self.models)])

        init_x_data = np.random.randn(input_shape).reshape(1, -1)
        init_y_data = np.random.randn(3).reshape(1, -1)

        self.train_on_batch(init_x_data, init_y_data)
        pass;

    def predict(self, x_data):
        if self.debug: print("Model::predict()")
        out = np.zeros((self.num_action))
        for i in range(self.num_action):
            out[i] = self.models[i].predict(x_data)
        print('SkModel::predict()', out)
        return out

    def train_on_batch(self, x_data, y_data):
        if self.debug: print("Model::train_on_batch()")
        loss = 0
        for i in range(self.num_action):
            self.models[i].fit(x_data, y_data[:, i])
            loss += self.models[i].score(x_data, y_data[:, i])

        return loss

    def save_weights(self, output, overwrite=False):
        pass