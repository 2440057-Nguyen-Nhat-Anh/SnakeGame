import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
import os
import argparse
import json
from datetime import datetime

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3)
        model_path = '/home/spring/SnakeGame/snake-ai-pytorch/model/model.pth'
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load('/home/spring/SnakeGame/snake-ai-pytorch/model/model.pth'))
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
    
    
    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached
        
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    max_episodes = 100000
    episode = 0
    while episode < max_episodes:
        
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)
        
        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()
                #save the best path along with model:
                # Save best episode information

                best_episode = {
                    'episode': agent.n_games,
                    'score': score,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'epsilon': agent.epsilon,
                    'gamma': agent.gamma,
                    'n_games': agent.n_games
                }
                
                # Save memory to file
                memory_path = 'model/best_memory.pkl'
                try:
                    memory_list = list(agent.memory)
                    torch.save(memory_list, memory_path)
                    print(f"Memory saved to {memory_path}")
                except Exception as e:
                    print(f"Error saving memory: {e}")

                with open('model/best_episode.json', 'w') as f:
                    json.dump(best_episode, f, indent=4)

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
        
        episode += 1

def test():
    """
    Load the best memory and model, and test the agent
    """
    # Load the best model
    agent = Agent() 
    agent.model.load_state_dict(torch.load('model.pth'))
    # Load the best memory
    memory_path = 'model/best_memory.pkl'
    try:
        memory_list = torch.load(memory_path)
        agent.memory = deque(memory_list, maxlen=MAX_MEMORY)
        print(f"Memory loaded from {memory_path}")
    except Exception as e:
        print(f"Error loading memory: {e}")
    # Load the best episode
    with open('model/best_episode.json', 'r') as f:
        best_episode = json.load(f)
        print(f"Best episode loaded: {best_episode}")
    # Test the agent
    game = SnakeGameAI()     
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

        if done:
            game.reset()
            print('Game', agent.n_games, 'Score', score)
            break    
if __name__ == '__main__':
    
    # argparser = argparse.ArgumentParser(description='Train or test the snake game AI')
    # argparser.add_argument('--mode', type=str, default='train', help='train or test')
    # args = argparser.parse_args()
    
    # mode = args.mode
    # print(f"Running in {mode} mode")
    # if mode == 'train':
    train()
    print("Training finished")
    print("Testing the agent")
    test()
    print("Testing finished")
    