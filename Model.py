import torch
import os


deviceG = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#torch.set_flush_denormal(True)
#torch.set_num_threads(2)
#torch.backends.cuda.benchmark = True


class Linear_QNet(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size, device = deviceG)
       
        self.linear2 = torch.nn.Linear(hidden_size, output_size, device = deviceG)

    def forward(self, x):
        
        x = torch.nn.functional.relu(self.linear1(x))
     
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:


    def __init__(self, model, lr):
        self.lr = lr
        self.gamma = 0.9
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        self.criterion = torch.nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float, device = deviceG)
        next_state = torch.tensor(next_state, dtype=torch.float, device = deviceG)
        action = torch.tensor(action, dtype=torch.long, device = deviceG)
        reward = torch.tensor(reward, dtype=torch.short, device = deviceG)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        del state, next_state, action, reward