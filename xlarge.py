import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import math
import os
import sys
from collections import namedtuple, deque
import atexit
import matplotlib.pyplot as plt
import datetime
import imageio


pygame.init()
font = pygame.font.Font(None, 25)


Point = namedtuple("Point", "x, y")


COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_RED = (200, 0, 0)
COLOR_GREEN = (0, 200, 0)
COLOR_BLUE1 = (0, 0, 255)
COLOR_BLUE2 = (0, 100, 255)

COLOR_FOOD1 = (200, 0, 0)
COLOR_FOOD2 = (0, 200, 0)
COLOR_FOOD3 = (0, 0, 200)
COLOR_HEAD = (255, 255, 255)
COLOR_TAIL = (80, 80, 80)


BLOCK_SIZE = 20
WIDTH = 17
HEIGHT = 15
SHOW_SPEED = 10


MAX_MEMORY = 100_000
BATCH_SIZE = 2048
LEARNING_RATE = 0.0001
GAMMA = 0.9
MODEL_PATH = f"./snake_{HEIGHT}_{WIDTH}_v4_model.pth"
SCORE_FILE = f"./highscore_{HEIGHT}_{WIDTH}_v4.txt"
GIF_PATH = f"./snake_{HEIGHT}_{WIDTH}_v4_game.gif"
TRAIN_EVERY_N_GAMES = 10


INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.01
EPSILON_DECAY_GAMES = 200000


DECAY_RATE = (FINAL_EPSILON / INITIAL_EPSILON) ** (1 / EPSILON_DECAY_GAMES)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def timestamp(st: datetime.datetime, ed: datetime.datetime) -> str:
    elapsed = ed - st
    total_seconds = int(elapsed.total_seconds())
    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{days:02}:{hours:02}:{minutes:02}:{seconds:02}"

def make_gif(frames: list[np.ndarray], filename: str, fps: int = 10):
    if not frames:
        return
    
    try:
        imageio.mimsave(filename, frames, fps=fps)
    except Exception as e:
        print(f"error: {e}")

class SnakeGame:

    def __init__(self, w=WIDTH, h=HEIGHT, render_mode=True):
        self.w = w
        self.h = h
        self.render_mode = render_mode

        self.display = None
        if self.render_mode:
            self.display = pygame.display.set_mode(
                (self.w * BLOCK_SIZE, self.h * BLOCK_SIZE)
            )
            pygame.display.set_caption("Snake RL (3 Food, Portals)")

        self.clock = pygame.time.Clock()

        self.direction = None
        self.head = None
        self.snake = None
        self.score = 0
        self.food = []
        self.frame_iteration = 0

        self.reset()

    def _is_occupied(self, pt):

        if pt in self.snake:
            return True
        for pair in self.food:
            if pt == pair[0] or pt == pair[1]:
                return True
        return False

    def _place_pair(self):

        pt1 = None
        pt2 = None
        while True:
            x1 = random.randint(0, self.w - 1)
            y1 = random.randint(0, self.h - 1)
            pt1 = Point(x1, y1)
            if not self._is_occupied(pt1):
                break

        while True:
            x2 = random.randint(0, self.w - 1)
            y2 = random.randint(0, self.h - 1)
            pt2 = Point(x2, y2)
            if pt2 != pt1 and not self._is_occupied(pt2):
                break
        return (pt1, pt2)

    def reset(self):

        self.direction = random.choice(["right", "left", "up", "down"])
        self.head = Point(self.w // 2, self.h // 2)
        self.snake = [
            self.head,
            Point(self.head.x - 1, self.head.y),
            Point(self.head.x - 2, self.head.y),
        ]

        self.score = 0
        self.food = []
        self.frame_iteration = 0

        for _ in range(3):
            self.food.append(self._place_pair())

    def _get_move_vector(self, action):

        dir_vec = None
        if self.direction == "right":
            dir_vec = (1, 0)
        elif self.direction == "left":
            dir_vec = (-1, 0)
        elif self.direction == "down":
            dir_vec = (0, 1)
        elif self.direction == "up":
            dir_vec = (0, -1)

        if action == [1, 0, 0]:
            new_dir_vec = dir_vec
        elif action == [0, 1, 0]:
            new_dir_vec = (dir_vec[1], -dir_vec[0])
        else:
            new_dir_vec = (-dir_vec[1], dir_vec[0])

        if new_dir_vec == (1, 0):
            self.direction = "right"
        elif new_dir_vec == (-1, 0):
            self.direction = "left"
        elif new_dir_vec == (0, 1):
            self.direction = "down"
        elif new_dir_vec == (0, -1):
            self.direction = "up"

    def _move_head(self):

        x, y = self.head.x, self.head.y
        if self.direction == "right":
            x += 1
        elif self.direction == "left":
            x -= 1
        elif self.direction == "down":
            y += 1
        elif self.direction == "up":
            y -= 1
        self.head = Point(x, y)

    def _get_nearest_food_and_pair(self):

        if not self.food:
            return None, None

        def dist(pt1, pt2):
            return math.sqrt((pt1.x - pt2.x) ** 2 + (pt1.y - pt2.y) ** 2)

        all_food_items = []
        for pair in self.food:
            all_food_items.append((pair[0], pair[1]))
            all_food_items.append((pair[1], pair[0]))

        if not all_food_items:
            return None, None

        head_dist = lambda item: dist(self.head, item[0])
        nearest, pair = min(all_food_items, key=head_dist)

        return nearest, pair

    def get_state(self):

        head_grid = np.zeros((self.h, self.w))
        head_grid[self.head.y, self.head.x] = 1.0

        body_grid = np.zeros((self.h, self.w))
        for pt in self.snake[1:]:
            body_grid[pt.y, pt.x] = 1.0

        food_grids = [np.zeros((self.h, self.w)) for _ in range(6)]

        for i, pair in enumerate(self.food):

            food_grids[i * 2][pair[0].y, pair[0].x] = 1.0
            food_grids[i * 2 + 1][pair[1].y, pair[1].x] = 1.0

        board_state = np.stack([head_grid, body_grid] + food_grids, axis=0)

        dir_l = self.direction == "left"
        dir_r = self.direction == "right"
        dir_u = self.direction == "up"
        dir_d = self.direction == "down"
        direction_state = np.array([dir_l, dir_r, dir_u, dir_d], dtype=int)

        return board_state, direction_state

    def _is_collision(self, pt=None):

        if pt is None:
            pt = self.head

        if pt.x >= self.w or pt.x < 0 or pt.y >= self.h or pt.y < 0:
            return True

        if pt in self.snake[1:]:
            return True
        return False

    def step(self, action):

        self.frame_iteration += 1

        if self.render_mode:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

        self._get_move_vector(action)

        self._move_head()
        self.snake.insert(0, self.head)

        reward = -0.2

        game_over = False

        if self._is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -30
            return reward, game_over, self.score

        eaten_pair_index = -1
        teleport_dest = None

        for i, pair in enumerate(self.food):
            if self.head == pair[0]:
                eaten_pair_index = i
                teleport_dest = pair[1]
                break
            elif self.head == pair[1]:
                eaten_pair_index = i
                teleport_dest = pair[0]
                break

        if eaten_pair_index != -1:
            self.score += 1
            reward = 20

            self.head = teleport_dest
            self.snake[0] = self.head

            if self._is_collision():
                game_over = True
                reward = -30
                return reward, game_over, self.score

            self.food.pop(eaten_pair_index)
            self.food.append(self._place_pair())

        else:

            self.snake.pop()

        if self.render_mode:
            self._render_ui()
            self.clock.tick(SHOW_SPEED)

        return reward, game_over, self.score

    def _render_ui(self):

        if self.display is None:
            return

        self.display.fill(COLOR_BLACK)

        body_segments = self.snake[1:]
        n = len(body_segments)

        start_color = np.array(COLOR_GREEN)
        end_color = np.array(COLOR_TAIL)

        for i, pt in enumerate(body_segments):

            t = 0.0
            if n > 1:
                t = i / (n - 1)

            current_color = start_color + t * (end_color - start_color)
            current_color = tuple(current_color.astype(int))

            pygame.draw.rect(
                self.display,
                current_color,
                (pt.x * BLOCK_SIZE, pt.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE),
            )

            pygame.draw.rect(
                self.display,
                COLOR_BLUE2,
                (pt.x * BLOCK_SIZE + 4, pt.y * BLOCK_SIZE + 4, 12, 12),
            )

        pygame.draw.rect(
            self.display,
            COLOR_HEAD,
            (
                self.head.x * BLOCK_SIZE,
                self.head.y * BLOCK_SIZE,
                BLOCK_SIZE,
                BLOCK_SIZE,
            ),
        )

        food_colors = [COLOR_FOOD1, COLOR_FOOD2, COLOR_FOOD3]
        for i, pair in enumerate(self.food):
            color = food_colors[i % len(food_colors)]
            pygame.draw.rect(
                self.display,
                color,
                (
                    pair[0].x * BLOCK_SIZE,
                    pair[0].y * BLOCK_SIZE,
                    BLOCK_SIZE,
                    BLOCK_SIZE,
                ),
            )
            pygame.draw.rect(
                self.display,
                color,
                (
                    pair[1].x * BLOCK_SIZE,
                    pair[1].y * BLOCK_SIZE,
                    BLOCK_SIZE,
                    BLOCK_SIZE,
                ),
            )

        text = font.render(f"Score: {self.score}", True, COLOR_WHITE)
        self.display.blit(text, (0, 0))

        pygame.display.flip()
        
    def screen_to_array(self):
        if self.display is None:
            return None
        data = pygame.surfarray.array3d(pygame.display.get_surface())
        return np.transpose(data, (1, 0, 2))


class QNetwork(nn.Module):
    def __init__(self, w=WIDTH, h=HEIGHT, output_size=3):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )

        def get_pool_out_dim(in_dim, k=2, s=2):
            return math.floor((in_dim - k) / s) + 1

        h_pool1 = get_pool_out_dim(h)
        w_pool1 = get_pool_out_dim(w)

        h_pool2 = get_pool_out_dim(h_pool1)
        w_pool2 = get_pool_out_dim(w_pool1)

        self.flattened_size = 64 * h_pool2 * w_pool2

        fc1_input_size = self.flattened_size + 4

        self.fc1 = nn.Linear(fc1_input_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, output_size)

    def forward(self, x_board, x_dir):
        x_conv = F.relu(self.conv1(x_board))
        x_conv = self.pool1(x_conv)

        x_conv = F.relu(self.conv2(x_conv))
        x_conv = self.pool2(x_conv)

        x_conv = F.relu(self.conv3(x_conv))

        x_flat = x_conv.view(-1, self.flattened_size)

        x_combined = torch.cat([x_flat, x_dir], dim=1)

        x_fc = F.relu(self.fc1(x_combined))
        x_fc = F.relu(self.fc2(x_fc))
        x_fc = F.relu(self.fc3(x_fc))
        x_out = self.fc4(x_fc)

        return x_out

    def save(self, file_path=MODEL_PATH):
        print(f"saving model to {file_path}")
        try:
            torch.save(self.state_dict(), file_path)
            print("saved model")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load(self, file_path=MODEL_PATH):
        if os.path.exists(file_path):
            print(f"loading model from {file_path}")
            try:

                self.load_state_dict(torch.load(file_path, map_location=device))
                self.eval()
                print("loaded model")
            except Exception as e:
                print(f"error: {e}")
        else:
            print(f"model not found: {file_path}")


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = INITIAL_EPSILON
        self.gamma = GAMMA
        self.memory = deque(maxlen=MAX_MEMORY)

        self.output_size = 3

        self.policy_net = QNetwork(w=WIDTH, h=HEIGHT, output_size=self.output_size).to(
            device
        )
        self.target_net = QNetwork(w=WIDTH, h=HEIGHT, output_size=self.output_size).to(
            device
        )

        self.policy_net.load()

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()

    def store_experience(self, state, action, reward, next_state, done):

        self.memory.append((state, action, reward, next_state, done))

    def train_short_memory(self, state, action, reward, next_state, done):

        self.train_step(state, action, reward, next_state, done)

    def train_long_memory(self):

        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.train_step(states, actions, rewards, next_states, dones)

    def train_step(self, state, action, reward, next_state, done):

        is_batch = not isinstance(done, bool)

        if is_batch:

            state_board, state_dir = zip(*state)
            next_state_board, next_state_dir = zip(*next_state)

            state_board = torch.tensor(np.array(state_board), dtype=torch.float).to(
                device
            )
            state_dir = torch.tensor(np.array(state_dir), dtype=torch.float).to(device)
            next_state_board = torch.tensor(
                np.array(next_state_board), dtype=torch.float
            ).to(device)
            next_state_dir = torch.tensor(
                np.array(next_state_dir), dtype=torch.float
            ).to(device)
            action = torch.tensor(np.array(action), dtype=torch.long).to(device)
            reward = torch.tensor(np.array(reward), dtype=torch.float).to(device)
            done = torch.tensor(np.array(done), dtype=torch.bool).to(device)

        else:

            state_board = (
                torch.tensor(np.array(state[0]), dtype=torch.float)
                .unsqueeze(0)
                .to(device)
            )
            state_dir = (
                torch.tensor(np.array(state[1]), dtype=torch.float)
                .unsqueeze(0)
                .to(device)
            )
            next_state_board = (
                torch.tensor(np.array(next_state[0]), dtype=torch.float)
                .unsqueeze(0)
                .to(device)
            )
            next_state_dir = (
                torch.tensor(np.array(next_state[1]), dtype=torch.float)
                .unsqueeze(0)
                .to(device)
            )
            action = (
                torch.tensor(np.array(action), dtype=torch.long).unsqueeze(0).to(device)
            )
            reward = (
                torch.tensor(np.array(reward), dtype=torch.float)
                .unsqueeze(0)
                .to(device)
            )
            done = (done,)

        pred = self.policy_net(state_board, state_dir)

        target = pred.clone()

        with torch.no_grad():
            Q_next_target = self.target_net(next_state_board, next_state_dir)
            max_Q_next = Q_next_target.max(dim=1)[0]

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:

                Q_new = reward[idx] + self.gamma * max_Q_next[idx]

            action_idx = torch.argmax(action[idx]).item()
            target[idx][action_idx] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)

        loss.backward()
        self.optimizer.step()

    def get_action(self, state, train_mode=True):

        final_move = [0, 0, 0]

        use_random_move = False
        if train_mode:
            self.epsilon = max(
                FINAL_EPSILON, INITIAL_EPSILON * (DECAY_RATE**self.n_games)
            )
            if random.random() < self.epsilon:
                use_random_move = True

        if use_random_move:

            move_idx = random.randint(0, 2)
            final_move[move_idx] = 1
        else:

            with torch.no_grad():

                state_tensor_board = (
                    torch.tensor(state[0], dtype=torch.float).unsqueeze(0).to(device)
                )
                state_tensor_dir = (
                    torch.tensor(state[1], dtype=torch.float).unsqueeze(0).to(device)
                )

                prediction = self.policy_net(state_tensor_board, state_tensor_dir)
                move_idx = torch.argmax(prediction).item()
                final_move[move_idx] = 1

        return final_move


def load_high_score():

    if os.path.exists(SCORE_FILE):
        try:
            with open(SCORE_FILE, "r") as f:
                score = int(f.read())
                print(f"previous high score: {score}")
                return score
        except ValueError:
            print(
                f"error reading {SCORE_FILE}"
            )
            return 0
        except Exception as e:
            print(f"error: {e}")
            return 0
    print("high score not found")
    return 0


def save_high_score(score):

    try:
        with open(SCORE_FILE, "w") as f:
            f.write(str(int(score)))
    except Exception as e:
        print(f"error: {e}")


def plot_scores(game_numbers, mean_scores):

    if not game_numbers or not mean_scores:
        return

    plt.figure(figsize=(10, 5))
    plt.plot(game_numbers, mean_scores)
    plt.title("Training Progress: Mean Score vs. Games Played")
    plt.xlabel("Number of Games")
    plt.ylabel("Mean Score (avg. of last 100)")
    plt.grid(True)

    plot_filename = "training_plot.png"
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")


def train():

    scores = []
    mean_scores = []
    game_numbers = []
    total_score = 0
    high_score = load_high_score()
    start_time = datetime.datetime.now()
    highscore_time = start_time

    agent = Agent()
    game = SnakeGame(render_mode=False)

    atexit.register(plot_scores, game_numbers, mean_scores)

    target_update_counter = 0

    print("starting training")
    print(f"model: {agent.policy_net}")

    while True:

        state_old = game.get_state()

        final_move = agent.get_action(state_old, train_mode=True)

        reward, done, score = game.step(final_move)

        if done:

            state_new = state_old
        else:
            state_new = game.get_state()

        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        agent.store_experience(state_old, final_move, reward, state_new, done)

        if done:

            game.reset()
            agent.n_games += 1

            if agent.n_games % TRAIN_EVERY_N_GAMES == 0:
                agent.train_long_memory()

            target_update_counter += 1
            if target_update_counter % 10 == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
                print("updating target network")

            if score > high_score:
                high_score = score
                highscore_time = datetime.datetime.now()
                agent.policy_net.save()
                save_high_score(high_score)

            scores.append(score)
            mean_score = np.mean(scores[-100:])
            mean_scores.append(mean_score)
            game_numbers.append(agent.n_games)

            elapsed = datetime.datetime.now() - start_time
            total_seconds = int(elapsed.total_seconds())
            
            curr_timestamp = timestamp(start_time, datetime.datetime.now())
            highscore_timestamp = timestamp(start_time, highscore_time)

            print(
                f"[{curr_timestamp}] ({highscore_timestamp}) Game: {agent.n_games:>6}, Score: {score:>3}, High: {high_score:>3}, "
                f"Epsilon: {agent.epsilon:.4f}, Mean: {mean_score:.2f}"
            )


def show():

    agent = Agent()
    agent.policy_net.load()
    agent.policy_net.eval()

    game = SnakeGame(render_mode=True)

    print("starting show mode")
    print(f"model: {agent.policy_net}")

    total_score = 0
    game_count = 0

    while True:

        state_old = game.get_state()

        final_move = agent.get_action(state_old, train_mode=False)

        reward, done, score = game.step(final_move)

        if done:
            game.reset()
            game_count += 1
            print(f"Game: {game_count}, Score: {score}")
            total_score += score
            print(f"    Avg: {total_score / game_count:.2f}")

def gif():
    global SHOW_SPEED
    
    SHOW_SPEED = 1000
    agent = Agent()
    game = SnakeGame()
    
    screens = []
    
    while True:
        state_old = game.get_state()
        final_move = agent.get_action(state_old, train_mode=False)
        _, done, _ = game.step(final_move)
        
        screens.append(game.screen_to_array())
        
        if done:
            break
        
    make_gif(screens, GIF_PATH, fps=10)

if __name__ == "__main__":
    mode = "train"
    if len(sys.argv) > 1:
        if sys.argv[1].lower() == "show":
            mode = "show"
        elif sys.argv[1].lower() == "train":
            mode = "train"
        elif sys.argv[1].lower() == "gif":
            mode = "gif"
        else:
            sys.exit()

    print(f"mode: {mode}")
    if mode == "train":
        train()
    elif mode == "show":
        show()
    elif mode == "gif":
        gif()