import sys
import time
import random
import numpy as np
from collections import deque
from gomoku_torch import Gomoku
from mcts_torch import MCTS
from neural_network_torch import Net
import torch.optim as optim
import torch
import torch.nn as nn
import pickle

#======================
# Configuration
#======================
# 8x8
"""game_board_width = 15
mcts_playout_itermax_train = 400
mcts_playout_itermax_play = 1000
model_file = 'Simple_CNN_15x15_3000'
policy_network = Simple_CNN # or Residual_CNN"""
#======================
# 19x19
game_board_width = 8
mcts_playout_itermax_train = 200
mcts_playout_itermax_play = 100
model_file = 'Torch_Residual_CNN_8x8_3000'
mcts_file = 'mcst_torch_residual_cnn_8x8_3000.pickle'
policy_network = Net
#======================

def random_play(game):
    return random.choice(game.actions())

def human_play():
    t = input('[*] Your turn (i j): ')
    a, b = t.split(' ')
    i, j = int(a), int(b)
    return (i, j)

def play_game():
    game = Gomoku(game_board_width)
    policy = policy_network()
    policy.load(model_file)
    mcts_player = MCTS(policy, mcts_playout_itermax_play)

    starting_player = random.choice([1,2])
    game.reset(starting_player)
    mcts_player.set_rootnode(starting_player)
    while not game.is_end:
        print(game)
        # print(game.nn_input)

        if game.current_player == 1: # Player X
            action, _ = mcts_player.get_move(game)
        else: # Player O
            action = human_play()
        
        game.move(action)
        mcts_player.update_with_move(action, game)

        print("[*] Player %s move: %s\n" % (['X', 'O'][game.player_just_moved-1], action))

    print(game)
    if game.winner > 0:
        print("[*] Player %s win" % ['X', 'O'][game.winner-1])
    else:
        print("[*] Player draw")

def self_play(game, player, render=False):
    starting_player = random.choice([1,2])
    game.reset(starting_player)
    player.set_rootnode(starting_player)
    board_states, mcts_probs, cur_players = [], [], []

    while not game.is_end:
        if render: print(game)
        action, action_probs = player.get_move(game, stochastically=True, show_node=render)
        board_states.append(game.nn_input)
        mcts_probs.append(action_probs)
        cur_players.append(game.current_player)

        game.move(action)
        player.update_with_move(action, game)

        if render: print("[*] Player %s move: %s\n" % (['X', 'O'][game.player_just_moved-1], action))

    rewards = list(map(game.reward, cur_players))

    if render:
        print(game)
        if game.winner > 0:
            print("[*] Player %s win" % ['X', 'O'][game.winner-1])
        else:
            print("[*] Player draw")

    return list(zip(board_states, mcts_probs, rewards)), game.winner, starting_player

def augment_data(play_data):
    # augment the data set by rotation and flipping
    extend_data = []
    for state, pi, z in play_data:
        w = state.shape[-1]
    

        for i in [1, 2, 3, 4]:
            # rotate counterclockwise
            equi_state = torch.rot90(state, 1, [1,2])
            equi_pi = torch.rot90(pi.view(w, w), 1, [0, 1])
            extend_data.append((equi_state, equi_pi.flatten(), z))
            # flip horizontally
            equi_state = torch.stack([torch.fliplr(equi_state[i]) for i in range(len(equi_state))])
            equi_pi =torch.fliplr(equi_pi)
            extend_data.append((equi_state, equi_pi.flatten(), z))

    return extend_data
    

def train():
    game_episode_num = 3000
    selfplay_batch_size = 1
    data_buffer_size = 10000
    check_step = 20
    train_batch_size = 512

    data_buffer = deque(maxlen=data_buffer_size)

    game = Gomoku(game_board_width)


    from os.path import exists
    file_exists = exists(model_file+".pth")
 
    epoch = 0

    if not file_exists:
        policy = policy_network().cuda()
        criterion_1 = nn.CrossEntropyLoss().cuda()
        criterion_2 = nn.MSELoss().cuda()
        """policy = policy_network()
        criterion_1 = nn.CrossEntropyLoss()
        criterion_2 = nn.MSELoss()"""
#        optimizer = optim.SGD(policy.parameters(), lr=0.1, momentum=0.9)
        optimizer = optim.Adam(policy.parameters(), lr=0.1)
    else:
        policy = policy_network().cuda()
        criterion_1 = nn.CrossEntropyLoss().cuda()
        criterion_2 = nn.MSELoss().cuda()
        """policy = policy_network()
        criterion_1 = nn.CrossEntropyLoss()
        criterion_2 = nn.MSELoss()"""
        optimizer = optim.Adam(policy.parameters(), lr=0.1)
        checkpoint = torch.load(model_file+'.pth')
        policy.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        print("load model success!")
    
    mcts_player = MCTS(policy, mcts_playout_itermax_train)
            
    winner_num = [0] * 3
    

    print('[*] Start self play')
    # game episode
    for i in range(game_episode_num):
        epoch += 1
        # get train data
        start_time = time.time()
        for _ in range(selfplay_batch_size):
            play_data, winner, starting_player = self_play(game, mcts_player)
            episode_len = len(play_data)
            extend_data = augment_data(play_data)
            data_num = len(extend_data)
            data_buffer.extend(extend_data)
            winner_num[winner] += 1
        end_time = time.time()

        print('[*] Episode: {}, length: {}, start: {}, winner: {}, data: {}, time: {}s, win ratio: X {:.1f}%, O {:.1f}%, - {:.1f}%'.format(
            epoch, episode_len, ['-', 'X', 'O'][starting_player], ['-', 'X', 'O'][winner], data_num, int(end_time - start_time),
            winner_num[1] / (i+1) * selfplay_batch_size * 100,
            winner_num[2] / (i+1) * selfplay_batch_size * 100,
            winner_num[0] / (i+1) * selfplay_batch_size * 100,
        ))

        # train
        if len(data_buffer) > train_batch_size:
            mini_batch = random.sample(data_buffer, train_batch_size)

            state_batch = torch.stack([d[0] for d in mini_batch]).cuda()
            pi_batch = torch.stack([d[1] for d in mini_batch]).cuda()
            z_batch = torch.tensor([d[2] for d in mini_batch]).cuda()

            """state_batch = torch.stack([d[0] for d in mini_batch])
            pi_batch = torch.stack([d[1] for d in mini_batch])
            z_batch = torch.tensor([d[2] for d in mini_batch])"""
            optimizer.zero_grad()
            predict_v, predict_p = policy(state_batch)
            loss_1 = criterion_1(predict_p, pi_batch)
            loss_2 = criterion_2(predict_v, z_batch)
            l2_c = 0.0001
            l2_norm = sum(p.pow(2.0).sum() for p in policy.parameters())
            loss = loss_1 + loss_2 + l2_c*l2_norm
            loss.backward()
            optimizer.step()
            running_loss = loss.item()
            print("loss: ", running_loss, "policy_loss:", loss_1, "value_loss:", loss_2)
            
        # check current policy model and save the params
            if (i + 1) % check_step == 0:
                torch.save({'epoch': (epoch),
                            'model_state_dict': policy.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss}, 
                            model_file+".pth")
                print("saved")


if __name__ == "__main__":
    if sys.argv[1] == '--train':
        train()
    elif sys.argv[1] == '--play':
        play_game()
