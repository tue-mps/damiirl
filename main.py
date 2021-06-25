import numpy as np
import numpy.random as random
import torch
import os

from objectworld import ObjectWorld
from binaryworld import BinaryWorld
from sem import SEM
from mcem import MCEM
from mdp import MDP
from drawing import Drawing
torch.manual_seed(0)

mirl_type = 'SEM' # either 'SEM' or 'MCEM', where 'SEM' : SEM-MIIRL and 'MCEM' : MCEM-MIIRL 
game_type = 'bw' # either 'ow' or 'bw', where 'ow' : M-ObjectWorld and 'bw' : M-BinaryWorld
checkpoint_dir = './checkpoints'
sample_length = 8 # the length of each demonstration sample
alpha = 1 # concentration parameter
sample_size = 16 # the number of demonstrations for each reward/intention
rewards_types = ['A','B','C'] # intention/reward types which are in total six, ['A','B','C','D','E','F']
exp_n = 1
seed = 1
mirl_maxiter = 200


checkpoint = {

            'seed': [],
            'game_type': [],
            'mirl_type': [],
            'game': [],
            'model': [],
            'rewards': [],
            'rewards_types': [],
            'rewardssquence' : [],
            'linmodel_solutions': [],
            'all_example_samples': [],
            'n_samples' : [],
            'mirl_solutions' : [],
            'EVDs' : [],
        }


checkpoint_name = str(exp_n)+mirl_type+game_type
checkpoint_path = os.path.join(checkpoint_dir,checkpoint_name+'.pt')
image_path = os.path.join(checkpoint_dir,checkpoint_name+'.png')

if game_type == 'ow':
    game = ObjectWorld(seed=seed)
elif game_type == 'bw':
    game = BinaryWorld(seed=seed)

model = MDP(game)
checkpoint['seed'] = seed
checkpoint['game_type'] = game_type
checkpoint['mirl_type'] = mirl_type
checkpoint['game'] = game
checkpoint['model'] = model

print('Saving game and model to {}'.format(checkpoint_path))
torch.save(checkpoint, checkpoint_path)
print('Done.')

rewards = []
linmodel_solutions = []
all_example_samples = []
n_samples = []

for r in range(len(rewards_types)):
    reward = game.gamereward(rewards_types[r])
    linmodel_solution = model.linearmdpsolve(reward)
    n_sample = sample_size
    example_samples = model.sampleexamples(linmodel_solution, training_samples = n_sample, training_sample_length=sample_length)
    for i in range(len(example_samples)):
        all_example_samples.append(example_samples[i])
    rewards.append(reward)
    linmodel_solutions.append(linmodel_solution)
    n_samples.append(n_sample)

if mirl_type == 'SEM':

    Mirl = SEM(
        game, rewards, model, linmodel_solutions, 
        all_example_samples, n_samples, 1, alpha
    )
elif mirl_type == 'MCEM':

    Mirl = MCEM(
        game, rewards, model, linmodel_solutions,
        all_example_samples, n_samples, 1, alpha
    )

print('solving for sample_size '+str(sample_size)+':')
print('solving for reward types '+str(rewards_types)+':')
print('MIRL training:')

mirl_solutions, EVDs, rewardssquence = Mirl.momaxentrun(maxIter = mirl_maxiter)
Drawing(game, rewards, rewardssquence, model, linmodel_solutions,all_example_samples, mirl_solutions,image_path)

checkpoint['rewards'] = rewards
checkpoint['rewards_types'] = rewards_types
checkpoint['rewardssquence'] = rewardssquence
checkpoint['linmodel_solutions'] = linmodel_solutions
checkpoint['all_example_samples'] = all_example_samples
checkpoint['n_samples'] = n_samples
checkpoint['mirl_solutions'] = mirl_solutions
checkpoint['EVDs'] = EVDs
print('Saving mirl solutions to {}'.format(checkpoint_path))
torch.save(checkpoint, checkpoint_path)
print('Done.')

    


