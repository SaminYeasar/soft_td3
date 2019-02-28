import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971
# [Not the implementation used in the TD3 paper]


class Actor(nn.Module):
	"""
	INPUT: state,s_t
	OUTPUT: action,a_
	t"""
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, action_dim)

		self.max_action = max_action


	def forward(self, x):
		x = F.relu(self.l1(x))
		x = F.relu(self.l2(x))
		x = self.max_action * torch.tanh(self.l3(x))
		return x


class Critic(nn.Module):
	"""
	INPUT: state,s_t and action, a_t
	OUTPUT: Q(s_t,a_t)
	"""
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		self.l1 = nn.Linear(state_dim, 400)
		self.l2 = nn.Linear(400 + action_dim, 300)
		self.l3 = nn.Linear(300, 1)


	def forward(self, x, u):
		x = F.relu(self.l1(x))
		x = F.relu(self.l2(torch.cat([x, u], 1)))
		x = self.l3(x)
		return x

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



class DDPGq(object):
	def __init__(self, state_dim, action_dim, max_action):
		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = Critic(state_dim, action_dim).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), weight_decay=1e-2)

		""" Samin: initialize network for r_hat """
		self.compute_r_hat = compute_r_hat(state_dim, action_dim).to(device)
		self.r_hat_optimizer = torch.optim.Adam(self.compute_r_hat.parameters(), lr=1e-4)

		self.Q_hat_network = Compute_Q_hat(state_dim, action_dim).to(device)
		self.Q_hat_target_network = Compute_Q_hat(state_dim, action_dim).to(device)
		self.Q_hat_target_network.load_state_dict(self.Q_hat_network.state_dict())
		self.Q_hat_network_optimizer = torch.optim.Adam(self.Q_hat_network.parameters(), lr=1e-4)


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, logger, replay_buffer, iterations, total_timesteps, batch_size=64, discount=0.99, tau=0.001):
		episodic_critic_loss = []
		for it in range(iterations):

			# Sample replay buffer
			x, y, u, r, d= replay_buffer.sample(batch_size)
			state = torch.FloatTensor(x).to(device)
			action = torch.FloatTensor(u).to(device)
			next_state = torch.FloatTensor(y).to(device)
			done = torch.FloatTensor(1 - d).to(device)
			reward = torch.FloatTensor(r).to(device)

			################
			# Compute r_hat
			################
			""" Samin : compute r_hat"""
			r_hat = self.compute_r_hat(state, action)
			r_hat_loss = F.mse_loss(r_hat, reward)

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
			Q_hat_target = self.Q_hat_target_network(next_state, self.actor_target(next_state))
			Q_hat_target = r_hat + (done * discount * Q_hat_target).detach()
			Q_hat = self.Q_hat_network(state, action)
			# Compute Q_hat loss
			Q_hat_loss = F.mse_loss(Q_hat, Q_hat_target)

			# Optimize the Q_hat
			self.Q_hat_network_optimizer.zero_grad()
			Q_hat_loss.backward(retain_graph=True)
			self.Q_hat_network_optimizer.step()


			# Compute the target Q value
			target_Q = self.critic_target(next_state, self.actor_target(next_state))

			# if total_timesteps >= 30000:
			#     print("stop")

			""" Equation: y_t = r_t + Q(s',a') + Q_hat -V_hat (ignored V_hat for the moment) """
			target_Q = reward + (done * discount * target_Q).detach() + Q_hat
			# Get current Q estimate
			current_Q = self.critic(state, action)

			# Compute critic loss
			critic_loss = F.mse_loss(current_Q, target_Q)
			# record logger
			episodic_critic_loss.append(critic_loss.detach())

			# Optimize the critic
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()

			# Compute actor loss
			actor_loss = -self.critic(state, self.actor(state)).mean()

			# Optimize the actor
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

		logger.record_critic_loss(torch.stack(episodic_critic_loss).mean().numpy())
	def save(self, filename, directory):
		torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
		torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))


	def load(self, filename, directory):
		self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
		self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))