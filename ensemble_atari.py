import gymnasium as gym
from gymnasium.wrappers import GrayScaleObservation, FrameStack, ResizeObservation, TransformObservation
import numpy as np
import torch
import setproctitle
import os
import argparse
import warnings
warnings.filterwarnings("ignore")
from utils import *
from evaluator import *

parser = argparse.ArgumentParser()
parser.add_argument('--manual_seed', type=int, default=1, help='manual seed for reproducibility')
parser.add_argument('--fixed_epsilon', type=float, default=0.1, help='fixed value of epsilon')
parser.add_argument('--cpu_num', type=int, default=1, help='number of CPU cores to use')
parser.add_argument('--env_name', type=str, default='Battlezone-v5', help='name of the environment to run')
parser.add_argument('--ensemble_method', type=str, default='LLM', choices=['majority_voting','rank_voting','boltzmann_multiplication','boltzmann_addition','aggregation','none','LLM'], help='Ensemble method to use')
parser.add_argument('--LLM_name', type=str, default='gpt', help='name of the LLM')
parser.add_argument('--episodes', type=int, default=10, help='number of episodes to run')
parser.add_argument('--LLM_max_try', type=int, default=100, help='number of tries to get the LLM output')
parser.add_argument('--sample_rate', type=int, default=30, help='sample rate of the LLM, frames/steps')
parser.add_argument('--api_id', type=int, default=1, choices=[1,2,3,4], help='1 or 2, api idx')
parser.add_argument('--llm_mode', type=str, default='eval', choices=['eval','infer'], help='eval agents or main exps')
args = parser.parse_args()
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    torch.manual_seed(seed)
    if args.device=='cuda':
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
set_seed(args.manual_seed)
set_cpu_num(args.cpu_num)

model_dirs = []
env = FrameStack(TransformObservation(ResizeObservation(GrayScaleObservation(gym.make(f'ALE/{args.env_name}', render_mode='rgb_array', full_action_space=False)), (110,84)), lambda x: np.array(x[18:102,:]).astype(np.float32) / 255.0), 4)

args.action_dim = env.action_space.n
args.env = env
args.model_dirs = model_dirs
evaluator = Evaluator(**vars(args))

if args.ensemble_method == 'LLM':
    if args.llm_mode == 'eval':
        evaluator.LLM_eval_agents()
    else:
        mean_reward, std_reward = evaluator.evaluate_LLM()
        write_csv(args.env_name,args.ensemble_method,args.episodes,args.manual_seed,mean_reward,std_reward)