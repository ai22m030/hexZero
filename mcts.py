import numpy as np
import math
from copy import deepcopy
from hex_engine import HexPosition

import torch
import torch.nn as nn
import torch.nn.functional as F


def add_dirichlet_noise(prior_probs, legal_actions, alpha=0.03):
    noise = np.random.dirichlet([alpha] * len(legal_actions))
    prior_probs[legal_actions] = 0.75 * prior_probs[legal_actions] + 0.25 * noise
    return prior_probs


class MCTSNode:
    def __init__(self, state: HexPosition, action=None, parent=None, prior=0):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.total_reward = 0
        self.move_reward = 0
        self.action = action
        self.prior = prior


class HexModel(nn.Module):
    def __init__(self, board_size, num_channels=64):
        super(HexModel, self).__init__()
        self.board_size = board_size
        self.input_size = board_size * board_size

        self.conv1 = nn.Conv2d(1, num_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 1, kernel_size=1),
            nn.Flatten(),
            nn.Linear(self.input_size, self.input_size),
            nn.LogSoftmax(dim=-1),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 1, kernel_size=1),
            nn.Flatten(),
            nn.Linear(self.input_size, num_channels),
            nn.ReLU(),
            nn.Linear(num_channels, 1),
            nn.Tanh(),
        )

    def forward(self, board):
        x = board.view(-1, 1, self.board_size, self.board_size)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        policy = self.policy_head(x)
        value = self.value_head(x)

        return policy, value

    def predict(self, board: np.ndarray, valid_actions):
        board_tensor = torch.tensor(board, dtype=torch.float32).unsqueeze(0)
        log_policy, value = self(board_tensor)
        policy_np = np.exp(log_policy.squeeze().detach().numpy())  # Exponentiating log probabilities

        # Set non-valid move probabilities to 0
        for i in range(len(policy_np)):
            if i not in valid_actions:
                policy_np[i] = 0

        # Normalize the probabilities of valid moves
        normalized_policy = policy_np / policy_np.sum()
        return normalized_policy, value.item()


class MCTS:
    def __init__(self, model, num_simulations, c):
        self.model = model
        self.num_simulations = num_simulations
        self.c = c

    def search(self, root: HexPosition):
        child_priors, value = self.model.predict(np.array(root.board), list(range(self.model.input_size)))
        root_node = MCTSNode(root, prior=1)
        self.expand(root_node, child_priors)

        for _ in range(self.num_simulations):
            leaf = self.traverse(deepcopy(root_node))  # leaf = unvisited node
            reward, child_priors = self.rollout(leaf)
            self.backpropagate(leaf, reward, child_priors)

        return root_node  # returning the root_node

    def traverse(self, node):
        while node.children:
            max_u, best_action = max(
                (self.ucb_score(node, action), action) for action in node.children
            )
            node = node.children[best_action]
        return node

    def rollout(self, node: MCTSNode):
        # Use the predict method of the HexModel
        legal_actions = [node.state.coordinate_to_scalar(action) for action in node.state.get_action_space()]
        policy, value = self.model.predict(np.array(node.state.board), legal_actions)
        return value, policy  # make sure policy is the second value

    def backpropagate(self, node: MCTSNode, reward, child_priors):
        # Add prior=1 when expanding nodes
        self.expand(node, child_priors)
        while node:
            node.total_reward += reward
            node.visit_count += 1
            reward = -reward
            node = node.parent

    def expand(self, node, child_priors):
        # Make sure child_priors is an iterable before using it
        if not isinstance(child_priors, (list, tuple, np.ndarray)):
            raise ValueError("child_priors should be an iterable")
        for action, prior in enumerate(child_priors):
            node.children[action] = MCTSNode(
                node.state.move(action), action=action, parent=node, prior=prior
            )

    def ucb_score(self, parent, action):
        node = parent.children[action]
        return (
            node.total_reward / node.visit_count
            + self.c
            * node.prior
            * math.sqrt(parent.visit_count) / (1 + node.visit_count)
        )


def self_play(model, num_games, num_simulations, c):
    games = []
    for _ in range(num_games):
        game = HexPosition()
        mcts = MCTS(model, num_simulations, c)

        while game.winner == 0:
            root_node = mcts.search(game)  # getting root_node from search method
            max_visits = max(
                (child.visit_count, action)
                for action, child in root_node.children.items()
            )[1]
            game.move(game.scalar_to_coordinates(max_visits))

        games.append(game.history)

    return games


def train(model, games, optimizer, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        for game_history in games:
            for state in game_history:
                board = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                policy_logits, value = model(board)
                targets = compute_targets(game_history, state)  # ensure that this function is implemented properly
                loss = compute_loss(policy_logits, value, targets)  # ensure that this function is implemented properly
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

    return model  # return the model after training


def compute_targets(game, state):
    policies = []
    values = []

    hex_position = HexPosition(size=len(game['gameplay'][0]))

    for idx, move in enumerate(game['moves']):
        action = hex_position.coordinate_to_scalar(move)
        policy = np.zeros(hex_position.size * hex_position.size, dtype=np.float32)
        policy[action] = 1.0
        value = game['winner'] * (1 if idx % 2 == 0 else -1)

        policies.append(policy)
        values.append(value)

        hex_position.move(move)

    policies = torch.tensor(np.stack(policies), dtype=torch.float32)
    values = torch.tensor(values).unsqueeze(1)

    return policies, values


def compute_loss(policy_logits, value, targets):
    # Unpack the targets
    target_policies, target_values = targets

    # Define an MSE loss for the values
    value_loss = F.mse_loss(value, target_values)

    # Define a negative log likelihood loss for the policies
    policy_loss = F.nll_loss(policy_logits, torch.max(target_policies, 1)[1])

    # Return the sum of the value and policy loss
    return value_loss + policy_loss
