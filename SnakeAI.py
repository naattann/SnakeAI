import torch
import random
from collections import deque
from Game import GameClass
from Model import Linear_QNet, QTrainer
from Plotter import plot
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


MAX_MEMORY = 1000000
BATCH_SIZE = 50000
learningRate = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 
        self.randomness = 40
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(14, 196, 3)
        self.model.cuda()
        self.trainer = QTrainer(self.model, lr=learningRate)


    def get_state(self, game):

        head = game.snake_body[0]
       
        dang1 = game.dang1
        dang2 = game.dang2
        dang3 = game.dang3
        dang4 = game.dang4

        CELL_NUM = game.CELL_NUM
        
        
        dir_l = game.direction == (-1,0)
        dir_r = game.direction == (1,0)
        dir_u = game.direction == (0,1)
        dir_d = game.direction == (0,-1)    

       

        state = [
         

            # Danger straight
            (dir_r and head[0] == CELL_NUM - 1) or 
            (dir_l and head[0] == 0) or 
            (dir_u and head[1] == CELL_NUM - 1) or 
            (dir_d and head[1] == 0),
            # Danger right
            (dir_u and head[0] == CELL_NUM - 1) or 
            (dir_d and head[0] == 0) or 
            (dir_l and head[1] == CELL_NUM - 1) or 
            (dir_r and head[1] == 0),
            # Danger left
            (dir_d and head[0] == CELL_NUM - 1) or 
            (dir_u and head[0] == 0) or 
            (dir_r and head[1] == CELL_NUM - 1) or 
            (dir_l and head[1] == 0),
            #Body Straight
            (dir_u and dang1 == True) or 
            (dir_d and dang2 == True) or 
            (dir_l and dang3 == True) or 
            (dir_r and dang4 == True),

            #Body Left
            (dir_u and dang1 == True) or 
            (dir_d and dang2 == True) or 
            (dir_l and dang3 == True) or 
            (dir_r and dang4 == True),

            #Body Right
            (dir_u and dang1 == True) or 
            (dir_d and dang2 == True) or 
            (dir_l and dang3 == True) or 
            (dir_r and dang4 == True),
            
            #Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.apple_pos[0] < head[0],  # food left
            game.apple_pos[0] > head[0],  # food right
            game.apple_pos[1] > head[1],  # food up
            game.apple_pos[1] < head[1],  # food down
            #game.apple_pos[0] == head[0],  # food up
            #game.apple_pos[1] == head[1],

            ]


        return state


    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def get_action(self, state):

        self.epsilon = self.randomness - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 100) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float,device = 'cuda:0')
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_rewards = []
    high_score = 0
    plot_reward = 0
    mean_rewards = 0
    total_rewards = 0

    agent = Agent()
    game = GameClass()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        plot_reward += reward

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > high_score:
                high_score = score

                if agent.n_games >= 100:
                    agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'High Score:', high_score)
           

            plot_scores.append(plot_reward)
            total_rewards += plot_reward
            mean_rewards = total_rewards / agent.n_games
            plot_mean_rewards.append(mean_rewards)

            plot(plot_scores, plot_mean_rewards)

            plot_reward = 0

if __name__ == '__main__':
    train()