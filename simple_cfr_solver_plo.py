import numpy as np

class SimpleCFRSolverPLO:
    def __init__(self, stack_size, bounty, player_with_bounty):
        self.stack_size = stack_size
        self.bounty = bounty
        self.player_with_bounty = player_with_bounty
        self.actions = ['fold', 'call', 'raise']
        self.num_actions = len(self.actions)
        self.regret_sum = {}
        self.strategy = {}
        self.strategy_sum = {}

    def get_strategy(self, node):
        if node not in self.regret_sum:
            self.regret_sum[node] = np.zeros(self.num_actions)
            self.strategy[node] = np.zeros(self.num_actions)
            self.strategy_sum[node] = np.zeros(self.num_actions)
        normalizing_sum = 0
        for a in range(self.num_actions):
            self.strategy[node][a] = max(self.regret_sum[node][a], 0)
            normalizing_sum += self.strategy[node][a]
        for a in range(self.num_actions):
            if normalizing_sum > 0:
                self.strategy[node][a] /= normalizing_sum
            else:
                self.strategy[node][a] = 1.0 / self.num_actions
            self.strategy_sum[node][a] += self.strategy[node][a]
        return self.strategy[node]

    def cfr(self, history, p0, p1):
        plays = len(history)
        player = plays % 2
        opponent = 1 - player

        if plays >= 2:
            return self.get_payoff(history)

        strategy = self.get_strategy(history)
        util = np.zeros(self.num_actions)
        node_util = 0
        for a in range(self.num_actions):
            next_history = history + self.actions[a]
            next_p0 = p0 * strategy[a] if player == 0 else p0
            next_p1 = p1 * strategy[a] if player == 1 else p1
            util[a] = -self.cfr(next_history, next_p0, next_p1)
            node_util += strategy[a] * util[a]
        
        for a in range(self.num_actions):
            regret = util[a] - node_util
            self.regret_sum[history][a] += (p1 if player == 0 else p0) * regret
        return node_util

    def get_payoff(self, history):
        if history == 'fold':
            return 1 if self.player_with_bounty == 1 else 0
        elif history == 'call':
            return self.bounty if self.player_with_bounty == 0 else -self.bounty
        elif history == 'raisefold':
            return 1 + self.bounty if self.player_with_bounty == 1 else -(1 + self.bounty)
        elif history == 'raisecall':
            return self.stack_size if self.player_with_bounty == 0 else -self.stack_size
        return 0

    def train(self, iterations):
        for _ in range(iterations):
            self.cfr('', 1, 1)

    def get_average_strategy(self):
        avg_strategy = {}
        for node in self.strategy_sum:
            normalizing_sum = np.sum(self.strategy_sum[node])
            if normalizing_sum > 0:
                avg_strategy[node] = self.strategy_sum[node] / normalizing_sum
            else:
                avg_strategy[node] = np.ones(self.num_actions) / self.num_actions
        return avg_strategy


# Пример использования
stack_size = 5
bounty = 1
player_with_bounty = 1  # Игрок с баунти - BB

solver = SimpleCFRSolverPLO(stack_size, bounty, player_with_bounty)
solver.train(iterations=1000)
average_strategy = solver.get_average_strategy()

for history, strategy in average_strategy.items():
    print(f"История: {history}, Стратегия: {strategy}")