import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils
from torch.distributions import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Policy weights
def weights_init_policy(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight, mean=0, std=0.1)

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon=1e-6


class SoftActor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(SoftActor, self).__init__()

		self.l1 = nn.Linear(state_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, action_dim)

		self.mean_linear = nn.Linear(300, action_dim)
		self.log_std_linear = nn.Linear(300, action_dim)

		self.apply(weights_init_policy)
		
		self.max_action = max_action


	def forward(self, x, reparam=True):
		x = F.relu(self.l1(x))
		x = F.relu(self.l2(x))
		action = self.max_action * torch.tanh(self.l3(x))

		mean = self.mean_linear(x)
		log_std = self.log_std_linear(x)
		log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
		std = log_std.exp()

		normal = Normal(mean, std)

		if reparam == True:
			x_t = normal.rsample()
		else:
			x_t = normal.sample()
		# action = torch.tanh(x_t)

		log_prob = normal.log_prob(x_t)

		log_prob -= torch.log(1 - action.pow(2) + epsilon)
		log_prob = log_prob.sum(-1, keepdim=True)

		entropy = normal.entropy()
		dist_entropy = entropy.sum(-1).mean()

		return action, dist_entropy, mean, log_std, log_prob


# Initialize QNetwork and Value Network weights
def weights_init_vf(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight)


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 400)
		self.l5 = nn.Linear(400, 300)
		self.l6 = nn.Linear(300, 1)

		self.apply(weights_init_vf)

	def forward(self, x, u):
		xu = torch.cat([x, u], 1)

		x1 = F.relu(self.l1(xu))
		x1 = F.relu(self.l2(x1))
		x1 = self.l3(x1)

		x2 = F.relu(self.l4(xu))
		x2 = F.relu(self.l5(x2))
		x2 = self.l6(x2)
		return x1, x2


	def Q1(self, x, u):
		xu = torch.cat([x, u], 1)

		x1 = F.relu(self.l1(xu))
		x1 = F.relu(self.l2(x1))
		x1 = self.l3(x1)
		return x1 






class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()

        hidden_dim = 300
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_vf)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x




class softTD3(object):
	def __init__(self, state_dim, action_dim, max_action):
		self.actor = SoftActor(state_dim, action_dim, max_action).to(device)
		self.actor_target = SoftActor(state_dim, action_dim, max_action).to(device)
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = Critic(state_dim, action_dim).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters())		


		self.value = ValueNetwork(state_dim).to(device)
		self.value_target = ValueNetwork(state_dim).to(device)
		self.value_target.load_state_dict(self.value.state_dict())
		self.value_optimizer = torch.optim.Adam(self.value.parameters())

		self.max_action = max_action


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		action, _, _, _, _ = self.actor(state)
		# action = torch.tanh(action)
		action = action.cpu().data.numpy().flatten()

		return action


	def train(self, args, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):

		for it in range(iterations):

			# Sample replay buffer 
			x, y, u, r, d = replay_buffer.sample(batch_size)
			state = torch.FloatTensor(x).to(device)
			action = torch.FloatTensor(u).to(device)
			next_state = torch.FloatTensor(y).to(device)
			done = torch.FloatTensor(1 - d).to(device)
			reward = torch.FloatTensor(r).to(device)

			# Select action according to policy and add clipped noise 
			noise = torch.FloatTensor(u).data.normal_(0, policy_noise).to(device)
			noise = noise.clamp(-noise_clip, noise_clip)

			next_action, _, _, _, _ = self.actor_target(next_state)
			next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
			next_action = next_action.clamp(-self.max_action, self.max_action)

			action_current_state, dist_entropy, mean_actor, log_std_actor, log_probs = self.actor(state)

			# Compute the value V
			expected_value = self.value(state)
			target_value = self.value_target(next_state)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			
			## target Q
			if args.use_baseline_in_target:
				target_Q = reward + (done * discount * target_Q - done * discount * target_value).detach()
			else:
				target_Q = reward + (done * discount * target_Q).detach()

			# Get current Q estimates
			current_Q1, current_Q2 = self.critic(state, action)
			# to compute critic regularizer
			a_target, _, _, _, _ = self.actor_target(state)
			q1, q2 = self.critic_target(state, a_target)
			q_t = torch.min(q1, q2)
			q_t = q_t.detach()

			a_current, _, _, _, _ = self.actor(state)
			q1_s, q2_s = self.critic(state, a_current)
			q_s = torch.min(q1_s, q2_s)

			# Compute critic loss
			critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) 
			critic_regularizer = F.mse_loss(q_s, q_t)

			if args.use_critic_regularizer:
				total_critic_loss = critic_loss + args.trust_critic_weight * critic_regularizer
			else:
				total_critic_loss = critic_loss


			# for target V
			q1_new, q2_new = self.critic(state, next_action)
			expected_q_value = torch.min(q1_new, q2_new)
			next_expected_value = expected_q_value - log_probs

			# Compute the value loss
			value_loss = F.mse_loss(expected_value, next_expected_value.detach())


			# Optimize the critic
			self.critic_optimizer.zero_grad()
			total_critic_loss.backward()
			self.critic_optimizer.step()

			# Optimize the value 
			self.value_optimizer.zero_grad()
			value_loss.backward()
			self.value_optimizer.step()

			# Delayed policy updates
			if it % policy_freq == 0:

				# Compute actor loss
				target_action_current_state, _, _, _, _ = self.actor_target(state)
				target_action_current_state = target_action_current_state.detach()

				if args.use_log_prob_in_policy:
					actor_loss = log_probs.mean() - self.critic.Q1(state, action_current_state).mean() - args.ent_weight * dist_entropy - self.value(state).mean() 
				elif args.use_value_baseline:
					actor_loss = - self.critic.Q1(state, action_current_state).mean() - args.ent_weight * dist_entropy - self.value(state).mean()
				else:
					actor_loss = - self.critic.Q1(state, action_current_state).mean() - args.ent_weight * dist_entropy				

				regularization_loss = 0.001 * mean_actor.pow(2).mean() + 0.001 * log_std_actor.pow(2).mean()
				actor_trust_regularizer = F.mse_loss(action_current_state, target_action_current_state)


				if args.use_regularization_loss:
					total_actor_loss = actor_loss + regularization_loss
				elif args.use_actor_regularizer:
					total_actor_loss = actor_loss + args.trust_actor_weight * actor_trust_regularizer
				elif args.diversity_expl:
					sampled_action = action
					new_sampled_action, _, _, _, _ = self.actor(state)
					actor_trust_regularizer = F.mse_loss(new_sampled_action, sample)
					total_actor_loss = actor_loss - 0.01 * actor_trust_regularizer
				else:
					total_actor_loss = actor_loss 


				# Optimize the actor 
				self.actor_optimizer.zero_grad()
				total_actor_loss.backward()
				self.actor_optimizer.step()

				# Update the frozen target models
				for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
					target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

				for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
					target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

				for param, target_param in zip(self.value.parameters(), self.value_target.parameters()):
					target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)



	def save(self, filename, directory):
		torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
		torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))


	def load(self, filename, directory):
		self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
		self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
