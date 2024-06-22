import numpy as np
import tkinter as tk
from tkinter import ttk

# Define the environment
class GameEnvironment:
    def __init__(self, width, height, rewards, p, step_cost):
        self.width = width
        self.height = height
        self.rewards = rewards
        self.p = p
        self.step_cost = step_cost
        self.grid = np.zeros((height, width))
        self.populate_rewards()

    def populate_rewards(self):
        for (x, y, r) in self.rewards:
            if 0 <= y < self.height and 0 <= x < self.width:
                self.grid[y, x] = r

    def is_terminal(self, x, y):
        return any((x, y) == (i, j) for (i, j, r) in self.rewards)

# Define Value Iteration Algorithm
def value_iteration(env, gamma=0.5, epsilon=0.01):
    H, W = env.height, env.width
    V = np.zeros((H, W))
    policy = np.zeros((H, W), dtype=str)
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    action_names = ["up", "down", "left", "right"]

    def get_reward(x, y):
        for (i, j, r) in env.rewards:
            if (x, y) == (i, j):
                return r
        return env.step_cost

    while True:
        delta = 0
        for y in range(H):
            for x in range(W):
                if env.is_terminal(x, y):
                    continue
                v = V[y, x]
                new_v = float('-inf')
                for action, name in zip(actions, action_names):
                    dy, dx = action
                    next_y, next_x = y + dy, x + dx
                    if 0 <= next_y < H and 0 <= next_x < W:
                        success_reward = get_reward(next_x, next_y) + gamma * V[next_y, next_x]
                        failure_reward = (
                            (get_reward(x + dx, y + dy) + gamma * V[y + dy, x + dx] if 0 <= y + dy < H and 0 <= x + dx < W else get_reward(x, y)) +
                            (get_reward(x - dx, y - dy) + gamma * V[y - dy, x - dx] if 0 <= y - dy < H and 0 <= x - dx < W else get_reward(x, y))
                        ) / 2
                        value = env.p * success_reward + (1 - env.p) * failure_reward
                        if value > new_v:
                            new_v = value
                            policy[y, x] = name
                V[y, x] = get_reward(x, y) + new_v
                delta = max(delta, abs(v - V[y, x]))
        if delta < epsilon:
            break

    return V, policy

# Define Policy Iteration Algorithm
def policy_iteration(env, gamma=0.5, epsilon=0.01):
    H, W = env.height, env.width
    V = np.zeros((H, W))
    policy = np.random.choice(["up", "down", "left", "right"], size=(H, W))
    actions = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1)
    }

    def get_reward(x, y):
        for (i, j, r) in env.rewards:
            if (x, y) == (i, j):
                return r
        return env.step_cost

    is_policy_stable = False
    while not is_policy_stable:
        # Policy Evaluation
        while True:
            delta = 0
            for y in range(H):
                for x in range(W):
                    if env.is_terminal(x, y):
                        continue
                    v = V[y, x]
                    dy, dx = actions[policy[y, x]]
                    next_y, next_x = y + dy, x + dx
                    if 0 <= next_y < H and 0 <= next_x < W:
                        success_reward = get_reward(next_x, next_y) + gamma * V[next_y, next_x]
                        failure_reward = (
                            (get_reward(x + dx, y + dy) + gamma * V[y + dy, x + dx] if 0 <= y + dy < H and 0 <= x + dx < W else get_reward(x, y)) +
                            (get_reward(x - dx, y - dy) + gamma * V[y - dy, x - dx] if 0 <= y - dy < H and 0 <= x - dx < W else get_reward(x, y))
                        ) / 2
                        V[y, x] = env.p * success_reward + (1 - env.p) * failure_reward
                        delta = max(delta, abs(v - V[y, x]))
            if delta < epsilon:
                break

        # Policy Improvement
        is_policy_stable = True
        for y in range(H):
            for x in range(W):
                if env.is_terminal(x, y):
                    continue
                old_action = policy[y, x]
                best_action = old_action
                best_value = float('-inf')
                for action in ["up", "down", "left", "right"]:
                    dy, dx = actions[action]
                    next_y, next_x = y + dy, x + dx
                    if 0 <= next_y < H and 0 <= next_x < W:
                        success_reward = get_reward(next_x, next_y) + gamma * V[next_y, next_x]
                        failure_reward = (
                            (get_reward(x + dx, y + dy) + gamma * V[y + dy, x + dx] if 0 <= y + dy < H and 0 <= x + dx < W else get_reward(x, y)) +
                            (get_reward(x - dx, y - dy) + gamma * V[y - dy, x - dx] if 0 <= y - dy < H and 0 <= x - dx < W else get_reward(x, y))
                        ) / 2
                        value = env.p * success_reward + (1 - env.p) * failure_reward
                        if value > best_value:
                            best_value = value
                            best_action = action
                policy[y, x] = best_action
                if old_action != best_action:
                    is_policy_stable = False

    return V, policy

# Define Q-Learning Algorithm
def q_learning(env, gamma=0.5, alpha=0.1, epsilon=0.1, episodes=1000):
    H, W = env.height, env.width
    Q = np.zeros((H, W, 4))
    actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # left, right, up, down
    action_indices = ["left", "right", "up", "down"]

    def get_reward(x, y):
        for (i, j, r) in env.rewards:
            if (x, y) == (i, j):
                return r
        return env.step_cost

    for _ in range(episodes):
        x, y = np.random.randint(0, W), np.random.randint(0, H)
        while not env.is_terminal(x, y):
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.choice(range(4))
            else:
                action = np.argmax(Q[y, x])

            dy, dx = actions[action]
            next_y, next_x = y + dy, x + dx
            if next_x < 0 or next_x >= W or next_y < 0 or next_y >= H:
                next_x, next_y = x, y  # Stay in place if out of bounds

            reward = get_reward(next_x, next_y)
            if env.is_terminal(next_x, next_y):
                Q[y, x, action] = Q[y, x, action] + alpha * (reward - Q[y, x, action])
                break
            else:
                next_action = np.argmax(Q[next_y, next_x])
                Q[y, x, action] = Q[y, x, action] + alpha * (reward + gamma * Q[next_y, next_x, next_action] - Q[y, x, action])

            x, y = next_x, next_y

    policy = np.zeros((H, W), dtype=str)
    for y in range(H):
        for x in range(W):
            best_action = np.argmax(Q[y, x])
            policy[y, x] = action_indices[best_action]

    return Q, policy

# Create the UI
class GameUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Reinforcement Learning Game")
        self.geometry("600x400")

        self.grid_frame = tk.Frame(self)
        self.grid_frame.pack()

        self.control_frame = tk.Frame(self)
        self.control_frame.pack()

        self.setup_controls()
        self.env = None

    def setup_controls(self):
        tk.Label(self.control_frame, text="Grid Size (W,H):").grid(row=0, column=0)
        self.grid_size_entry = tk.Entry(self.control_frame)
        self.grid_size_entry.grid(row=0, column=1)

        tk.Label(self.control_frame, text="Rewards (x,y,r):").grid(row=1, column=0)
        self.rewards_entry = tk.Entry(self.control_frame)
        self.rewards_entry.grid(row=1, column=1)

        tk.Label(self.control_frame, text="Step Success Probability:").grid(row=2, column=0)
        self.p_entry = tk.Entry(self.control_frame)
        self.p_entry.grid(row=2, column=1)

        tk.Label(self.control_frame, text="Step Cost:").grid(row=3, column=0)
        self.step_cost_entry = tk.Entry(self.control_frame)
        self.step_cost_entry.grid(row=3, column=1)

        self.create_env_button = tk.Button(self.control_frame, text="Create Environment", command=self.create_environment)
        self.create_env_button.grid(row=4, column=0, columnspan=2)

        self.value_iteration_button = tk.Button(self.control_frame, text="Value Iteration", command=self.run_value_iteration)
        self.value_iteration_button.grid(row=5, column=0)

        self.policy_iteration_button = tk.Button(self.control_frame, text="Policy Iteration", command=self.run_policy_iteration)
        self.policy_iteration_button.grid(row=5, column=1)

        self.q_learning_button = tk.Button(self.control_frame, text="Q-Learning", command=self.run_q_learning)
        self.q_learning_button.grid(row=6, column=0, columnspan=2)

    def create_environment(self):
        grid_size = tuple(map(int, self.grid_size_entry.get().split(',')))
        rewards = [tuple(map(int, r.split(','))) for r in self.rewards_entry.get().split(';')]
        p = float(self.p_entry.get())
        step_cost = float(self.step_cost_entry.get())
        self.env = GameEnvironment(grid_size[0], grid_size[1], rewards, p, step_cost)
        self.draw_grid()

    def draw_grid(self):
        for widget in self.grid_frame.winfo_children():
            widget.destroy()
        for y in range(self.env.height):
            for x in range(self.env.width):
                value = self.env.grid[y, x]
                color = "white"
                if value > 0:
                    color = "green"
                elif value < 0:
                    color = "red"
                tk.Label(self.grid_frame, text=str(value), bg=color, width=4, height=2, borderwidth=1, relief="solid").grid(row=y, column=x)

    def run_value_iteration(self):
        V, policy = value_iteration(self.env)
        self.display_policy(policy)

    def run_policy_iteration(self):
        V, policy = policy_iteration(self.env)
        self.display_policy(policy)

    def run_q_learning(self):
        Q, policy = q_learning(self.env)
        self.display_policy(policy)

    def display_policy(self, policy):
        for widget in self.grid_frame.winfo_children():
            widget.destroy()
        for y in range(self.env.height):
            for x in range(self.env.width):
                action = policy[y, x]
                color = "white"
                if self.env.grid[y, x] > 0:
                    color = "green"
                elif self.env.grid[y, x] < 0:
                    color = "red"
                tk.Label(self.grid_frame, text=str(action), bg=color, width=4, height=2, borderwidth=1, relief="solid").grid(row=y, column=x)

if __name__ == "__main__":
    app = GameUI()
    app.mainloop()
