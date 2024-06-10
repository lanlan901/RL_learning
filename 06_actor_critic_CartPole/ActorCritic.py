from torch import tensor, log
from torch.distributions import Categorical
from torch.optim import Adam

from Networks import PolicyNet, ValueNet


class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, target_update_frequency, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.target_update_frequency = target_update_frequency
        self.device = device
        self.count = 0

    def take_action(self, state):
        probs = self.actor(tensor(state).to(self.device))
        categorical = Categorical(probs)
        return categorical.sample().item()

    def update(self, transitions):
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        for state, action, next_state, reward, done in transitions:
            state = state.to(self.device)
            td_target = reward.to(self.device) + self.gamma * self.critic(next_state.to(self.device)) * (1 - done)
            delta = td_target - self.critic(state)

            critic_loss = pow(td_target.detach() - self.critic(state), 2)
            actor_loss = -log(self.actor(state)[action]) * delta.detach()
            critic_loss.backward()
            actor_loss.backward()
        self.critic_optimizer.step()
        self.actor_optimizer.step()

