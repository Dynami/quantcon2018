import torch
import torch.nn.functional as F
import numpy as np

# Define model
class TModel(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TModel, self).__init__()
        self.fc1 = torch.nn.Linear(D_in, H)
        self.fc2 = torch.nn.Linear(H, H)
        self.fc3 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class TorchModel(object):

    def __init__(self, D_in, H, D_out):
        self.model = TModel(D_in=D_in, H=H, D_out=D_out)

        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        self.learning_rate = 1e-3
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
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

        # compute loss
        loss = self.loss_fn(y_pred_tensor, y_data_tensor)
        # reset optimizer gradinent
        self.optimizer.zero_grad()
        # backward losses
        loss.backward()
        # optimize net
        self.optimizer.step()

        return loss.item()

    def save_weights(self, output, overwrite=False):
        torch.save(self.model.state_dict(), output)
        pass

    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()