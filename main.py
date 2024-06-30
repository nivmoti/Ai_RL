from collections import defaultdict
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
        return any((x, y) == (i, j) and r != 0 for (i, j, r) in self.rewards)

    def is_wall(self, x, y):
        return any((x, y) == (i, j) and r == 0 for (i, j, r) in self.rewards)


def simulate_step(env, y, x, action):
    """
    Simulates the next state and reward based on the action taken,
    considering the probability of movement success and failure.
    """
    actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # left, right, up, down
    action_failures = [((1, 0), (-1, 0)), ((-1, 0), (1, 0)), ((0, -1), (0, 1)), ((0, 1), (0, -1))]
    dy, dx = actions[action]
    next_y, next_x = y + dy, x + dx

    if np.random.rand() < env.p:  # Successful move
        if next_x < 0 or next_x >= env.width or next_y < 0 or next_y >= env.height or env.is_wall(next_x, next_y):
            return y, x, env.step_cost  # Stay in the same place if moving into a wall
        return next_y, next_x, env.grid[next_y, next_x]
    else:  # Failed move, move right or left relative to the original move
        fail1, fail2 = action_failures[action]
        fail1_y, fail1_x = y + fail1[0], x + fail1[1]
        fail2_y, fail2_x = y + fail2[0], x + fail2[1]

        p_right = (1 - env.p) / 2
        p_left = 1 - env.p - p_right

        if np.random.rand() < p_right:  # Move right
            if fail1_x < 0 or fail1_x >= env.width or fail1_y < 0 or fail1_y >= env.height or env.is_wall(fail1_x,
                                                                                                          fail1_y):
                return y, x, env.step_cost  # Stay in the same place if moving into a wall
            return fail1_y, fail1_x, env.grid[fail1_y, fail1_x]
        else:  # Move left
            if fail2_x < 0 or fail2_x >= env.width or fail2_y < 0 or fail2_y >= env.height or env.is_wall(fail2_x,
                                                                                                          fail2_y):
                return y, x, env.step_cost  # Stay in the same place if moving into a wall
            return fail2_y, fail2_x, env.grid[fail2_y, fail2_x]


def value_iteration(env, gamma=0.5, epsilon=0.01):
    H, W = env.height, env.width
    V = np.zeros((H, W))
    policy = np.full((H, W), None)  # Initialize policy with None
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    action_names = ["up", "down", "left", "right"]
    action_failures = [((0, -1), (0, 1)), ((0, 1), (0, -1)), ((-1, 0), (1, 0)), ((1, 0), (-1, 0))]

    def get_reward(x, y):
        for (i, j, r) in env.rewards:
            if (x, y) == (i, j):
                return r
        return env.step_cost

    while True:
        delta = 0
        for y in range(H):
            for x in range(W):
                if env.is_terminal(x, y) or env.is_wall(x, y):
                    continue
                v = V[y, x]
                new_v = float('-inf')
                for action, name, failures in zip(actions, action_names, action_failures):
                    dy, dx = action
                    fail1, fail2 = failures
                    next_y, next_x = y + dy, x + dx
                    fail1_y, fail1_x = y + fail1[0], x + fail1[1]
                    fail2_y, fail2_x = y + fail2[0], x + fail2[1]

                    if next_x < 0 or next_x >= W or next_y < 0 or next_y >= H or env.is_wall(next_x, next_y):
                        next_y, next_x = y, x  # Stay in the same place if moving into a wall
                    if fail1_x < 0 or fail1_x >= W or fail1_y < 0 or fail1_y >= H or env.is_wall(fail1_x, fail1_y):
                        fail1_y, fail1_x = y, x  # Stay in the same place if moving into a wall
                    if fail2_x < 0 or fail2_x >= W or fail2_y < 0 or fail2_y >= H or env.is_wall(fail2_x, fail2_y):
                        fail2_y, fail2_x = y, x  # Stay in the same place if moving into a wall

                    success_reward = get_reward(next_x, next_y) + gamma * V[next_y, next_x]
                    failure_reward = (
                                             (get_reward(fail1_x, fail1_y) + gamma * V[fail1_y, fail1_x]) +
                                             (get_reward(fail2_x, fail2_y) + gamma * V[fail2_y, fail2_x])
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
                    if env.is_terminal(x, y) or env.is_wall(x, y):
                        continue
                    v = V[y, x]
                    dy, dx = actions[policy[y, x]]
                    next_y, next_x = y + dy, x + dx
                    if next_x < 0 or next_x >= W or next_y < 0 or next_y >= H or env.is_wall(next_x, next_y):
                        next_y, next_x = y, x  # Stay in the same place if moving into a wall
                    success_reward = get_reward(next_x, next_y) + gamma * V[next_y, next_x]
                    failure_reward = (
                                             (get_reward(x + dx, y + dy) + gamma * V[
                                                 y + dy, x + dx] if 0 <= y + dy < H and 0 <= x + dx < W and not env.is_wall(
                                                 x + dx, y + dy) else get_reward(x, y)) +
                                             (get_reward(x - dx, y - dy) + gamma * V[
                                                 y - dy, x - dx] if 0 <= y - dy < H and 0 <= x - dx < W and not env.is_wall(
                                                 x - dx, y - dy) else get_reward(x, y))
                                     ) / 2
                    V[y, x] = env.p * success_reward + (1 - env.p) * failure_reward
                    delta = max(delta, abs(v - V[y, x]))
            if delta < epsilon:
                break

        # Policy Improvement
        is_policy_stable = True
        for y in range(H):
            for x in range(W):
                if env.is_terminal(x, y) or env.is_wall(x, y):
                    continue
                old_action = policy[y, x]
                best_action = old_action
                best_value = float('-inf')
                for action in ["up", "down", "left", "right"]:
                    dy, dx = actions[action]
                    next_y, next_x = y + dy, x + dx
                    if next_x < 0 or next_x >= W or next_y < 0 or next_y >= H or env.is_wall(next_x, next_y):
                        next_y, next_x = y, x  # Stay in the same place if moving into a wall
                    success_reward = get_reward(next_x, next_y) + gamma * V[next_y, next_x]
                    failure_reward = (
                                             (get_reward(x + dx, y + dy) + gamma * V[
                                                 y + dy, x + dx] if 0 <= y + dy < H and 0 <= x + dx < W and not env.is_wall(
                                                 x + dx, y + dy) else get_reward(x, y)) +
                                             (get_reward(x - dx, y - dy) + gamma * V[
                                                 y - dy, x - dx] if 0 <= y - dy < H and 0 <= x - dx < W and not env.is_wall(
                                                 x - dx, y - dy) else get_reward(x, y))
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
    actions = ["left", "right", "up", "down"]

    def get_reward(x, y):
        for (i, j, r) in env.rewards:
            if (x, y) == (i, j):
                return r
        return env.step_cost

    for _ in range(episodes):
        x, y = np.random.randint(0, W), np.random.randint(0, H)
        while not env.is_terminal(x, y) and not env.is_wall(x, y):
            if np.random.rand() < epsilon:
                action = np.random.choice(range(4))
            else:
                action = np.argmax(Q[y, x])

            next_y, next_x, reward = simulate_step(env, y, x, action)

            if env.is_terminal(next_x, next_y):
                Q[y, x, action] = Q[y, x, action] + alpha * (reward - Q[y, x, action])
                break
            else:
                next_action = np.argmax(Q[next_y, next_x])
                Q[y, x, action] = Q[y, x, action] + alpha * (
                            reward + gamma * Q[next_y, next_x, next_action] - Q[y, x, action])

            x, y = next_x, next_y

    policy = np.full((H, W), None, dtype=object)  # Initialize policy with None
    reward = np.zeros((H, W))
    for y in range(H):
        for x in range(W):
            if not env.is_terminal(x, y) and not env.is_wall(x, y):
                best_action = np.argmax(Q[y, x])
                reward[y, x] = Q[y, x, best_action]
                policy[y, x] = actions[best_action]

    return Q, policy, reward


def learn_mdp_from_experience(experience):
    Rai = defaultdict(lambda: defaultdict(float))
    Nai = defaultdict(lambda: defaultdict(int))
    Naij = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for (i, a, r, j) in experience:
        Rai[i][a] += r
        Nai[i][a] += 1
        Naij[i][a][j] += 1

    reward_model = defaultdict(lambda: defaultdict(float))
    transition_model = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    for state in Nai:
        for action in Nai[state]:
            reward_model[state][action] = Rai[state][action] / Nai[state][action]
            for next_state in Naij[state][action]:
                transition_model[state][action][next_state] = Naij[state][action][next_state] / Nai[state][action]

    return transition_model, reward_model


def model_based_rl(env, gamma=0.9, epsilon=0.01, episodes=1000, ep_length=10):
    H, W = env.height, env.width
    V = np.zeros((H, W))
    policy = np.random.choice(["up", "down", "left", "right"], size=(H, W))
    experience = []

    # number of epochs
    for k in range(episodes):
        x, y = np.random.randint(0, W), np.random.randint(0, H)
        state = (x, y)  # Change later only for clarity

        # cant start from this one
        if env.is_terminal(x, y) or env.is_wall(x, y):
            continue

        # inner run
        for i in range(ep_length):
            action = policy[state]
            next_state, reward = env.step(action)  # how can make take the next based on the action
            experience.append((state, action, reward, next_state))
            state = next_state

        transition_model, reward = learn_mdp_from_experience(experience)
        new_env = GameEnvironment(W, H, reward, env.p, env.step_cost)  # Step cost still not sure how
        V, new_policy = value_iteration(new_env)
        policy = new_policy

    return V, policy


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

        self.create_env_button = tk.Button(self.control_frame, text="Create Environment",
                                           command=self.create_environment)
        self.create_env_button.grid(row=4, column=0, columnspan=2)

        self.value_iteration_button = tk.Button(self.control_frame, text="Value Iteration",
                                                command=self.run_value_iteration)
        self.value_iteration_button.grid(row=5, column=0)

        self.model_based_button = tk.Button(self.control_frame, text="Model Based",
                                            command=self.run_model_based)
        self.model_based_button.grid(row=5, column=1)

        self.q_learning_button = tk.Button(self.control_frame, text="Q-Learning", command=self.run_q_learning)
        self.q_learning_button.grid(row=6, column=0, columnspan=2)

        self.result_button = tk.Button(self.control_frame, text="Results", command=self.results)
        self.result_button.grid(row=7, column=0, columnspan=2)

    def results(self):
        MDP_V, MDP_policy = value_iteration(self.env)
        MBRL_V, MBRL_policy = policy_iteration(self.env)
        Q, MFRL_policy, MFRL_reward = q_learning(self.env)

        np.set_printoptions(formatter={'float': '{: 0.6f}'.format})

        print("-" * 30 + "\n d(MDP,MBRL)")
        resul_1 = MBRL_V - MDP_V
        print(resul_1)
        print(f"average: {np.mean(resul_1[resul_1 != 0])}")

        print("-" * 30 + "\n d(MDP,MFRL)")
        resul_2 = MFRL_reward - MDP_V
        print(resul_2)
        print(f"average: {np.mean(resul_2[resul_2 != 0])}")

        print("-" * 30 + "\n d(MBRL,MFRL)")
        resul_3 = MFRL_reward - MBRL_V
        print(resul_3)
        print(f"average: {np.mean(resul_3[resul_3 != 0])}")

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
                elif self.env.is_wall(x, y):
                    color = "black"  # Wall
                tk.Label(self.grid_frame, text=str(value), bg=color, width=4, height=2, borderwidth=1,
                         relief="solid").grid(row=y, column=x)

    def run_value_iteration(self):
        V, policy = value_iteration(self.env)
        self.display_policy(policy)
        # Print results:
        print("-" * 30)
        print("Value Iteration")
        print(V)

    def run_model_based(self):
        V, policy = policy_iteration(self.env)
        self.display_policy(policy)
        # Print results:
        print("-" * 30)
        print("Model Based")
        print(V)

    def run_q_learning(self):
        Q, policy, reward = q_learning(self.env)
        self.display_policy(policy)
        # Print results:
        print("-" * 30)
        print("Q learning")
        print(reward)

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
                elif self.env.is_wall(x, y):
                    color = "black"  # Wall
                text = "" if action is None else action  # Don't display policy for terminal or wall states
                tk.Label(self.grid_frame, text=text, bg=color, width=4, height=2, borderwidth=1, relief="solid").grid(
                    row=y, column=x)


if __name__ == "__main__":
    app = GameUI()
    app.mainloop()
