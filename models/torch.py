import torch
import numpy as np

class TorchModel(object):

    def __init__(self, D_in, H, D_out):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(D_in, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, D_out)
        )

        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        self.learning_rate = 1e-4
        self.output_shape = np.array([D_out])
        pass

    def predict(self, x_data):
        x_data_tensor = torch.from_numpy(x_data).float()
        y_data = self.model(x_data_tensor)
        return y_data.detach().numpy()

    def train_on_batch(self, x_data, y_data):
        x_data_tensor = torch.from_numpy(x_data).float()
        y_data_tensor = torch.from_numpy(y_data).float()
        y_pred_tensor = self.model(x_data_tensor)

        loss = self.loss_fn(y_pred_tensor, y_data_tensor)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # with torch.no_grad():
        #     for param in self.model.parameters():
        #         param -= self.learning_rate * param.grad

        return loss.item()

    def save_weights(self, output, overwrite=False):
        pass