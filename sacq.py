import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
#from utils import soft_update, hard_update
from sac_model import GaussianPolicy, QNetwork, ValueNetwork, soft_update, hard_update
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#############################################
# Samin: Network to predict r_hat from buffer
#############################################
class compute_r_hat(nn.Module):
	"""
	INPUT: s,a from replay buffer
	OUTPUT: r_hat
	"""
	def __init__(self, state_dim, action_dim):
		super(compute_r_hat, self).__init__()

		self.l1 = nn.Linear(state_dim+action_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, 1)

	"""add softmax at the output"""
	def forward(self, state, action):
		x = F.relu(self.l1(torch.cat([state, action], 1)))
		x = F.relu(self.l2(x))
		x = self.l3(x)
		return x

class Compute_Q_hat(nn.Module):
	"""
	basically same architecture as Critic as both computes Q(s_t,a_t)
	"""
	def __init__(self, state_dim, action_dim):
		super(Compute_Q_hat, self).__init__()

		self.l1 = nn.Linear(state_dim, 400)
		self.l2 = nn.Linear(400 + action_dim, 300)
		self.l3 = nn.Linear(300, 1)


	def forward(self, x, u):
		x = F.relu(self.l1(x))
		x = F.relu(self.l2(torch.cat([x, u], 1)))
		x = self.l3(x)
		return x





class SAC(object):
	def __init__(self, num_inputs, action_space, args):

		self.num_inputs = num_inputs
		self.action_space = action_space.shape[0]
		self.gamma = args.gamma
		self.tau = args.tau
		self.scale_R = args.scale_R
		self.reparam = args.reparam
		self.deterministic = args.deterministic
		self.target_update_interval = args.target_update_interval

		self.policy = GaussianPolicy(self.num_inputs, self.action_space, args.hidden_size).to(device)
		self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

		self.critic = QNetwork(self.num_inputs, self.action_space, args.hidden_size).to(device)
		self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

		if self.deterministic == False:
			self.value = ValueNetwork(self.num_inputs, args.hidden_size).to(device)
			self.value_target = ValueNetwork(self.num_inputs, args.hidden_size).to(device)
			self.value_optim = Adam(self.value.parameters(), lr=args.lr)
			hard_update(self.value_target, self.value)
			self.value_criterion = nn.MSELoss()
		else:
			self.critic_target = QNetwork(self.num_inputs, self.action_space, args.hidden_size).to(device)
			hard_update(self.critic_target, self.critic)

		self.soft_q_criterion = nn.MSELoss()

		""" Samin: initialize network for r_hat """

		self.compute_r_hat = compute_r_hat(num_inputs, action_space).to(device)
		self.r_hat_optimizer = torch.optim.Adam(self.compute_r_hat.parameters(), lr=1e-4)

		self.Q_hat_network = Compute_Q_hat(num_inputs, action_space).to(device)
		self.Q_hat_target_network = Compute_Q_hat(num_inputs, action_space).to(device)
		self.Q_hat_target_network.load_state_dict(self.Q_hat_network.state_dict())
		self.Q_hat_network_optimizer = torch.optim.Adam(self.Q_hat_network.parameters(), lr=1e-4)

	def select_action(self, state, eval=False):
		state = torch.FloatTensor(state).unsqueeze(0)
		if eval == False:
			self.policy.train()
			_, _, action, _, _ = self.policy.evaluate(state)
		else:
			self.policy.eval()
			_, _, _, action, _ = self.policy.evaluate(state)

		action = torch.tanh(action)
		action = action.detach().cpu().numpy()
		return action[0]

	def update_parameters(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch, updates):
		state_batch = torch.FloatTensor(state_batch)
		next_state_batch = torch.FloatTensor(next_state_batch)
		action_batch = torch.FloatTensor(action_batch)
		reward_batch = torch.FloatTensor(reward_batch)
		mask_batch = torch.FloatTensor(np.float32(mask_batch))

		reward_batch = reward_batch.unsqueeze(1)  # reward_batch = [batch_size, 1]
		mask_batch = mask_batch.unsqueeze(1)  # mask_batch = [batch_size, 1]

		"""
		Use two Q-functions to mitigate positive bias in the policy improvement step that is known
		to degrade performance of value based methods. Two Q-functions also significantly speed
		up training, especially on harder task.
		"""
		expected_q1_value, expected_q2_value = self.critic(state_batch, action_batch)
		new_action, log_prob, x_t, mean, log_std = self.policy.evaluate(state_batch, reparam=self.reparam)

		###################################################################################################

		################
		# Compute r_hat
		################
		""" Samin : compute r_hat"""
		r_hat = self.compute_r_hat(state_batch, action_batch)
		r_hat_loss = F.mse_loss(r_hat, reward_batch)

		# Optimize one-step r_hat
		self.r_hat_optimizer.zero_grad()
		r_hat_loss.backward(retain_graph=True)  # need to be sure why it' necessary to retain the graph here
		self.r_hat_optimizer.step()

		################
		# Compute Q hat
		################
		"""using the critic network to compute Q_hat
		If I use different network then will have to compute loss for that network as well
		"""
		Q_hat_target = self.Q_hat_target_network(next_state_batch, self.actor_target(next_state_batch))
		Q_hat_target = r_hat + (mask_batch * self.gamma* Q_hat_target).detach()
		Q_hat = self.Q_hat_network(state_batch, action_batch)
		# Compute Q_hat loss
		Q_hat_loss = F.mse_loss(Q_hat, Q_hat_target)

		# Optimize the Q_hat
		self.Q_hat_network_optimizer.zero_grad()
		Q_hat_loss.backward(retain_graph=True)
		self.Q_hat_network_optimizer.step()

		###################################################################################################


		"""
		Including a separate function approximator for the soft value can stabilize training.
		"""
		expected_value = self.value(state_batch)
		target_value = self.value_target(next_state_batch)

		"""Samin: added Q_hat"""
		next_q_value = self.scale_R * reward_batch + mask_batch * self.gamma * target_value + Q_hat # Reward Scale * r(st,at) - γV(target)(st+1))

		"""
		Soft Q-function parameters can be trained to minimize the soft Bellman residual
		JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
		∇JQ = ∇Q(st,at)(Q(st,at) - r(st,at) - γV(target)(st+1))
		"""
		q1_value_loss = self.soft_q_criterion(expected_q1_value, next_q_value.detach())
		q2_value_loss = self.soft_q_criterion(expected_q2_value, next_q_value.detach())

		q1_new, q2_new = self.critic(state_batch, new_action)
		expected_new_q_value = torch.min(q1_new, q2_new)

		"""
		Including a separate function approximator for the soft value can stabilize training and is convenient to 
		train simultaneously with the other networks
		Update the V towards the min of two Q-functions in order to reduce overestimation bias from function approximation error.
		JV = 𝔼st~D[0.5(V(st) - (𝔼at~π[Qmin(st,at) - log π(at|st)]))^2]
		∇JV = ∇V(st)(V(st) - Q(st,at) + logπ(at|st))
		"""
		next_value = expected_new_q_value - log_prob
		value_loss = self.value_criterion(expected_value, next_value.detach())
		log_prob_target = expected_new_q_value - expected_value

		if self.reparam == True:
			"""
			Reparameterization trick is used to get a low variance estimator
			f(εt;st) = action sampled from the policy
			εt is an input noise vector, sampled from some fixed distribution
			Jπ = 𝔼st∼D,εt∼N[logπ(f(εt;st)|st)−Q(st,f(εt;st))]
			∇Jπ =∇log π + ([∇at log π(at|st) − ∇at Q(st,at)])∇f(εt;st)
			"""
			policy_loss = (log_prob - expected_new_q_value).mean()
		else:
			policy_loss = (
						log_prob * (log_prob - log_prob_target).detach()).mean()  # likelihood ratio gradient estimator

		# Regularization Loss
		mean_loss = 0.001 * mean.pow(2).mean()
		std_loss = 0.001 * log_std.pow(2).mean()

		policy_loss += mean_loss + std_loss

		self.critic_optim.zero_grad()
		q1_value_loss.backward()
		self.critic_optim.step()

		self.critic_optim.zero_grad()
		q2_value_loss.backward()
		self.critic_optim.step()

		if self.deterministic == False:
			self.value_optim.zero_grad()
			value_loss.backward()
			self.value_optim.step()

		self.policy_optim.zero_grad()
		policy_loss.backward()
		self.policy_optim.step()

		"""
		We update the target weights to match the current value function weights periodically
		Update target parameter after every n(args.target_update_interval) updates
		"""
		if updates % self.target_update_interval == 0 and self.deterministic == True:
			soft_update(self.critic_target, self.critic, self.tau)
			return 0, q1_value_loss.item(), q2_value_loss.item(), policy_loss.item()
		elif updates % self.target_update_interval == 0 and self.deterministic == False:
			soft_update(self.value_target, self.value, self.tau)
			return value_loss.item(), q1_value_loss.item(), q2_value_loss.item(), policy_loss.item()

		# following is exactly doing soft update
		if updates % self.target_update_interval == 0:
			for param, target_param in zip(self.Q_hat_network.parameters(), self.Q_hat_target_network.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	# Save model parameters
	def save_model(self, env_name, suffix="", actor_path=None, critic_path=None, value_path=None):
		if not os.path.exists('models/'):
			os.makedirs('models/')

		if actor_path is None:
			actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
		if critic_path is None:
			critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
		if value_path is None:
			value_path = "models/sac_value_{}_{}".format(env_name, suffix)
		print('Saving models to {}, {} and {}'.format(actor_path, critic_path, value_path))
		torch.save(self.value.state_dict(), value_path)
		torch.save(self.policy.state_dict(), actor_path)
		torch.save(self.critic.state_dict(), critic_path)

	# Load model parameters
	def load_model(self, actor_path, critic_path, value_path):
		print('Loading models from {}, {} and {}'.format(actor_path, critic_path, value_path))
		if actor_path is not None:
			self.policy.load_state_dict(torch.load(actor_path))
		if critic_path is not None:
			self.critic.load_state_dict(torch.load(critic_path))
		if value_path is not None:
			self.value.load_state_dict(torch.load(value_path))
