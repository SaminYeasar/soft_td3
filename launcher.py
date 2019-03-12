import os
import itertools
import numpy as np
import subprocess
import argparse
parser = argparse.ArgumentParser()

def grid_search(args_vals):
    """ arg_vals: a list of lists, each one of format (argument, list of possible values) """
    lists = []
    for arg_vals in args_vals:
        arg, vals = arg_vals
        ll = []
        for val in vals:
            ll.append("-" + arg + " " + str(val) + " ")
        lists.append(ll)
    return ["".join(item) for item in itertools.product(*lists)]


# parser = argparse.ArgumentParser()
# parser.add_argument('--experiments', type=int, default=3)
# parser.add_argument('--policy_name', type=str, default="DDPG")          # Policy name
# parser.add_argument('--env_name', type=str, default="HalfCheetah-v1")         # OpenAI gym environment name
# parser.add_argument('--start_timesteps', default=10000, type=int)     # How many time steps purely random policy is run for
# parser.add_argument('--eval_freq', default=5e3, type=float)         # How often (time steps) we evaluate
# parser.add_argument('--max_timesteps', default=1e6, type=float)     # Max time steps to run environment for
# parser.add_argument('--save_models', default=True,type=bool)           # Whether or not models are saved
# parser.add_argument('--expl_noise', default=0.1, type=float)        # Std of Gaussian exploration noise
# parser.add_argument('--batch_size', default=100, type=int)          # Batch size for both actor and critic
# parser.add_argument('--discount', default=0.99, type=float)         # Discount factor
# parser.add_argument('--tau', default=0.005, type=float)             # Target network update rate
# parser.add_argument('--policy_noise', default=0.2, type=float)      # Noise added to target policy during critic update
# parser.add_argument('--noise_clip', default=0.5, type=float)        # Range to clip target policy noise
# parser.add_argument('--policy_freq', default=2, type=int)           # Frequency of delayed policy updates
# parser.add_argument("--ent_weight", default=0.005, type=float)
# parser.add_argument('-g',  type=str, default='0', help=['specify GPU'])
# parser.add_argument('--folder', type=str, default="./results/")          # Folder to save results in
# parser.add_argument("--trust_actor_weight", default=0.01, type=float)
# parser.add_argument("--trust_critic_weight", default=0.01, type=float)
#
#
# parser.add_argument("--diversity_expl", type=bool, default=False, help='whether to use diversity driven exploration')
# parser.add_argument("--use_baseline_in_target", type=bool, default=False, help='use baseline in target')
# parser.add_argument("--use_critic_regularizer", type=bool, default=False, help='use regularizer in critic')
# parser.add_argument("--use_actor_regularizer", type=bool, default=False, help='use regularizer in actor')
# parser.add_argument("--use_log_prob_in_policy", type=bool, default=False, help='use log prob in actor loss as in SAC')
# parser.add_argument("--use_value_baseline", type=bool, default=False, help='use value function baseline in actor loss to reduce variance')
# parser.add_argument("--use_regularization_loss", type=bool, default=False, help='use simple regularizion losses for mean and log std of policy')
#
# parser.add_argument("--use_dueling", type=bool, default=False, help='use dueling network architectures')
# parser.add_argument("--use_logger", type=bool, default=False, help='whether to use logging or not')

parser = argparse.ArgumentParser()
parser.add_argument('--experiments', type=int, default=3)
parser.add_argument("--policy_name", default="td3_doubly_robust", help='DDPG')					# Policy name
parser.add_argument("--env_name", default="HalfCheetah-v2")			# OpenAI gym environment name
parser.add_argument("--seed", default=0, type=int)					# Sets Gym, PyTorch and Numpy seeds
parser.add_argument("--start_timesteps", default=10000, type=int)		# How many time steps purely random policy is run for
parser.add_argument("--eval_freq", default=5e3, type=float)			# How often (time steps) we evaluate
parser.add_argument("--max_timesteps", default=1e6, type=float)		# Max time steps to run environment for
parser.add_argument("--save_models", default=True)			# Whether or not models are saved
parser.add_argument("--expl_noise", default=0.1, type=float)		# Std of Gaussian exploration noise
parser.add_argument("--batch_size", default=100, type=int)			# Batch size for both actor and critic
parser.add_argument("--discount", default=0.99, type=float)			# Discount factor
parser.add_argument("--tau", default=0.005, type=float)				# Target network update rate
parser.add_argument("--policy_noise", default=0.2, type=float)		# Noise added to target policy during critic update
parser.add_argument("--noise_clip", default=0.5, type=float)		# Range to clip target policy noise
parser.add_argument("--policy_freq", default=2, type=int)			# Frequency of delayed policy updates
parser.add_argument("--ent_weight", default=0.01, type=float)		# Range to clip target policy noise
parser.add_argument("--folder", type=str, default='./results/')

parser.add_argument("--trust_actor_weight", default=0.01, type=float)
parser.add_argument("--trust_critic_weight", default=0.01, type=float)
parser.add_argument("--diversity_expl", type=bool, default=False, help='whether to use diversity driven exploration')

parser.add_argument("--use_baseline_in_target", type=bool, default=False, help='use baseline in target')
parser.add_argument("--use_critic_regularizer", type=bool, default=False, help='use regularizer in critic')
parser.add_argument("--use_actor_regularizer", type=bool, default=False, help='use regularizer in actor')
parser.add_argument("--use_log_prob_in_policy", type=bool, default=False, help='use log prob in actor loss as in SAC')
parser.add_argument("--use_value_baseline", type=bool, default=False, help='use value function baseline in actor loss to reduce variance')
parser.add_argument("--use_regularization_loss", type=bool, default=False, help='use simple regularizion losses for mean and log std of policy')
parser.add_argument("--use_dueling", type=bool, default=False, help='use dueling network architectures')
parser.add_argument("--use_logger", type=bool, default=True, help='whether to use logging or not')






locals().update(parser.parse_args().__dict__)    


job_prefix = "python "
exp_script = './main.py ' 
job_prefix += exp_script

args = parser.parse_args()

experiments = args.experiments
policy_name = args.policy_name
env_name = args.env_name
start_timesteps = args.start_timesteps
eval_freq = args.eval_freq
max_timesteps = args.max_timesteps
expl_noise = args.expl_noise
batch_size = args.batch_size
discount = args.discount
tau = args.tau
policy_noise = args.policy_noise
noise_clip = args.noise_clip
policy_freq = args.policy_freq
ent_weight = args.ent_weight
folder = args.folder
save_models = args.save_models
trust_actor_weight = args.trust_actor_weight
trust_critic_weight = args.trust_critic_weight
diversity_expl = args.diversity_expl
use_baseline_in_target = args.use_baseline_in_target
use_critic_regularizer = args.use_critic_regularizer
use_actor_regularizer = args.use_actor_regularizer
use_log_prob_in_policy = args.use_log_prob_in_policy
use_value_baseline = args.use_value_baseline
use_regularization_loss = args.use_regularization_loss
use_dueling = args.use_dueling
use_logger = args.use_logger


grid = [] 
grid += [['-policy_name', [policy_name]]]
grid += [['-env_name', [env_name]]]
grid += [['-start_timesteps', [start_timesteps]]]
grid += [['-eval_freq', [eval_freq]]]
grid += [['-max_timesteps', [max_timesteps]]]
grid += [['-save_models',[args.save_models]]]
grid += [['-expl_noise', [expl_noise]]]
grid += [['-batch_size', [batch_size]]]
grid += [['-batch_size', [batch_size]]]
grid += [['-tau', [tau]]]
grid += [['-policy_noise', [policy_noise]]]
grid += [['-noise_clip', [noise_clip]]]
grid += [['-policy_freq', [policy_freq]]]
grid += [['-ent_weight', [ent_weight]]]
grid += [['-folder', [folder]]]
grid += [['-trust_actor_weight', [trust_actor_weight]]]
grid += [['-trust_critic_weight', [trust_critic_weight]]]
grid += [['-diversity_expl', [diversity_expl]]]
grid += [['-use_baseline_in_target', [use_baseline_in_target]]]
grid += [['-use_critic_regularizer', [use_critic_regularizer]]]
grid += [['-use_actor_regularizer', [use_actor_regularizer]]]
grid += [['-use_log_prob_in_policy', [use_log_prob_in_policy]]]
grid += [['-use_value_baseline', [use_value_baseline]]]
grid += [['-use_regularization_loss', [use_regularization_loss]]]
grid += [['-use_dueling', [use_dueling]]]
grid += [['-use_logger', [use_logger]]]



job_strs = []
for settings in grid_search(grid):
    for e in range(experiments):    
        job_str = job_prefix + settings
        job_strs.append(job_str)
print("njobs", len(job_strs))

for job_str in job_strs:
    os.system(job_str)