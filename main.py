import numpy as np
import torch
import gym
import argparse
import os

import utils
import TD3
import OurDDPG
import DDPG
import softTD3
import DDPGadv
import DDPGq
from utils import Logger
import td3q
import td3adv
import td3doubleq
from utils import create_folder



# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, eval_episodes=10):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print ("---------------------------------------")
    print ("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print ("---------------------------------------")
    return avg_reward


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default='td3doubleq', help='DDPGq')					# Policy name
    parser.add_argument("--env_name", default="HalfCheetah-v2")			# OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)					# Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4, type=int)		# How many time steps purely random policy is run for
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
    args = parser.parse_args()

    if args.use_logger:
        file_name = "%s_%s_%s" % (args.policy_name, args.env_name, str(args.seed))

        logger = Logger(experiment_name = args.policy_name, environment_name = args.env_name, folder = "./results/")
        logger.save_args(args)

        print ('Saving to', logger.save_folder)


    if not os.path.exists("./results"):
        os.makedirs("./results")

    env = gym.make(args.env_name)

    # Set seeds
    seed = np.random.randint(10)
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if args.use_logger:
        print ("---------------------------------------")
        print ("Settings: %s" % (file_name))
        print ("Seed : %s" % (seed))
        print ("---------------------------------------")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize policy
    if args.policy_name == "TD3": policy = TD3.TD3(state_dim, action_dim, max_action)
    elif args.policy_name == "td3q": policy = td3q.TD3(state_dim, action_dim, max_action)
    elif args.policy_name == "td3adv":policy = td3adv.TD3(state_dim, action_dim, max_action)
    elif args.policy_name == "td3doubleq": policy = td3doubleq.TD3(state_dim, action_dim, max_action)
    elif args.policy_name == "softTD3": policy = softTD3.softTD3(state_dim, action_dim, max_action)
    elif args.policy_name == "OurDDPG": policy = OurDDPG.DDPG(state_dim, action_dim, max_action)
    elif args.policy_name == "DDPG": policy = DDPG.DDPG(state_dim, action_dim, max_action)
    elif args.policy_name == "DDPGq": policy = DDPGq.DDPGq(state_dim, action_dim, max_action)
    elif args.policy_name == "DDPGadv":policy = DDPGadv.DDPGadv(state_dim, action_dim, max_action)

    replay_buffer = utils.ReplayBuffer()

    # Evaluate untrained policy
    evaluations = [evaluate_policy(policy)]
    episode_reward = 0
    training_evaluations = [episode_reward]

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True

    while total_timesteps < args.max_timesteps:

        if done:

            if total_timesteps != 0:
                print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (total_timesteps, episode_num, episode_timesteps, episode_reward))
                if args.policy_name == "TD3":
                    policy.train(logger, replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau, args.policy_noise, args.noise_clip, args.policy_freq)
                elif args.policy_name == "softTD3":
                    policy.train(args, replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau, args.policy_noise, args.noise_clip, args.policy_freq)
                elif args.policy_name == "td3q" or args.policy_name == "td3adv" or args.policy_name == "td3doubleq":
                    policy.train(logger, replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau, args.policy_noise, args.noise_clip, args.policy_freq)
                else:
                    policy.train(logger, replay_buffer, episode_timesteps, total_timesteps, args.batch_size, args.discount, args.tau)


            # Evaluate episode
            if timesteps_since_eval >= args.eval_freq:
                timesteps_since_eval %= args.eval_freq
                evaluations.append(evaluate_policy(policy))
                if args.use_logger:
                    logger.record_reward(evaluations)
                    logger.save()
                    if args.save_models: logger.save_policy(policy, file_name)
                    """Samin: changed the saving file to keep track"""
                    #if args.save_models: policy.save(file_name, directory="./pytorch_models_DR{}".format(args.use_DR))
                    #np.save("./results/{}_{}".format(file_name, args.use_DR), evaluations)

            # Reset environment
            obs = env.reset()
            done = False
            training_evaluations.append(episode_reward)

            if args.use_logger:
                logger.training_record_reward(training_evaluations)
                logger.save_2()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Select action randomly or according to policy
        if total_timesteps < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.select_action(np.array(obs))
            if args.policy_name=="softTD3":
                action = (action).clip(env.action_space.low, env.action_space.high)
            else:
                action = (action + np.random.normal(0, args.expl_noise, size=env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)

        # Perform action
        new_obs, reward, done, _ = env.step(action)
        done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
        episode_reward += reward

        # Store data in replay buffer
        replay_buffer.add((obs, new_obs, action, reward, done_bool))

        obs = new_obs

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1

    # Final evaluation
    evaluations.append(evaluate_policy(policy))
    training_evaluations.append(episode_reward)
    if args.use_logger:
        logger.record_reward(evaluations)
        logger.training_record_reward(training_evaluations)
        logger.save()
        logger.save_2()
        logger.save_critic_loss()  # save the critic loss

        if args.save_models: logger.save_policy(policy, file_name)
        #if args.save_models: policy.save(file_name, directory="./pytorch_models_DR{}".format(args.use_DR))
        #np.save("./results/{}_{}".format(file_name, args.use_DR), evaluations)
        #if args.save_models: policy.save("%s" % (file_name), directory="./pytorch_models")
        #np.save("./results/%s" % (file_name), evaluations)
