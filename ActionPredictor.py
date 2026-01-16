import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActionPredictor(nn.Module):
    def __init__(self, obs_dim, action_dim, lr=1e-4):
        super(ActionPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.current_head = nn.Linear(64, action_dim)
        self.next_head = nn.Linear(64, action_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()
        self.cur_losses = []
        self.next_losses = []

        self.cur_loss = np.inf
        self.next_loss = np.inf

    def forward(self, obs):
        x = self.net(obs.to('cpu'))
        current_action_logits = self.current_head(x)
        next_action_logits = self.next_head(x)
        #cur_prob = torch.softmax(current_action_logits, dim=-1)
        #next_prob = torch.softmax(next_action_logits, dim=-1)
        cur_act = torch.argmax(current_action_logits, dim=-1)
        next_prob = torch.argmax(next_action_logits, dim=-1)
        return current_action_logits, next_action_logits

    def sample_action(self, obs):
        cur_logit, next_logit = self.forward(obs)
        cur_act = torch.argmax(cur_logit, dim=-1)
        next_act = torch.argmax(next_logit, dim=-1)
        return cur_act, next_act

    def update(self, obs_batch, current_action_labels, next_action_labels):
        """
        obs_batch: (batch_size, obs_dim)
        current_action_labels: (batch_size,)
        next_action_labels: (batch_size,)
        """
        self.train()
        self.optimizer.zero_grad()

        current_action_logits, next_action_logits = self.forward(obs_batch)

        current_action_logits = current_action_logits.squeeze().float().to('cpu')

        next_action_logits = next_action_logits.squeeze().float().to('cpu')
        current_action_labels = current_action_labels.squeeze()
        next_action_labels = next_action_labels.squeeze()
        next_action_logits.requires_grad_()
        current_action_logits.requires_grad_()
        current_action_labels.requires_grad_()
        next_action_labels.requires_grad_()
        #print(current_action_logits.shape, current_action_labels.shape)
        loss_current = self.loss_fn(current_action_logits, current_action_labels.long())

        loss_next = self.loss_fn(next_action_logits, next_action_labels.long())
        loss = loss_current + loss_next
        loss.backward()
        self.optimizer.step()
        # update loss value
        if self.cuda:
            get_cur_loss = loss_current.cpu().data.numpy()
            get_next_loss = loss_current.cpu().data.numpy()

        else:
            get_cur_loss = loss_current.detach().numpy()
            get_next_loss = loss_current.detach().numpy()


        self.cur_losses.append(get_cur_loss)
        self.next_losses.append(get_next_loss)

        if len(self.cur_losses)>100:
            del self.cur_losses[0]
            del self.next_losses[0]

        self.cur_loss = np.mean(self.cur_losses)
        self.next_loss = np.mean(self.next_losses)
        #return loss.item(), loss_current.item(), loss_next.item()