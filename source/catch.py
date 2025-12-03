from skimage.transform import resize
import random
import numpy as np


class CatchEnv():
    def __init__(self, paddle_width=5, noise=False):
        self.size = 21
        self.image = np.zeros((self.size, self.size))
        self.background = np.zeros((self.size, self.size))
        self.state = []
        self.fps = 4
        self.output_shape = (84, 84)

        self.paddle_width = paddle_width
        self.radius = paddle_width // 2
        self.noise = noise

    def reset_random(self):
        if self.noise:
            self.background = np.random.uniform(0, 0.8, size=self.image.shape)
        else:
            self.background.fill(0)
        self.image = self.background.copy()

        min_pos = self.radius
        max_pos = self.size - self.radius - 1

        self.pos = np.random.randint(min_pos, max_pos + 1)
        self.vx = np.random.randint(5) - 2
        self.vy = 1
        self.ballx, self.bally = np.random.randint(self.size), 4
        self.image[self.bally, self.ballx] = 1

        slice_start = self.pos - self.radius
        slice_end = self.pos - self.radius + self.paddle_width
        self.image[-5, slice_start:slice_end] = 1

        return self.step(2)[0]

    def step(self, action):
        def left():
            if self.pos - 2 >= self.radius:
                self.pos -= 2

        def right():
            if self.pos + 2 < self.size - (self.paddle_width - self.radius):
                self.pos += 2

        def noop():
            pass

        # Erase previous paddle
        slice_start = self.pos - self.radius
        slice_end = self.pos - self.radius + self.paddle_width
        self.image[-5, slice_start:slice_end] = self.background[-5, slice_start:slice_end]

        action_fn = {0: left, 1: right, 2: noop}[action]
        action_fn()

        self.image[self.bally, self.ballx] = self.background[self.bally, self.ballx]
        self.ballx += self.vx
        self.bally += self.vy
        if self.ballx > self.size - 1:
            self.ballx -= 2 * (self.ballx - (self.size-1))
            self.vx *= -1
        elif self.ballx < 0:
            self.ballx += 2 * (0 - self.ballx)
            self.vx *= -1
        self.image[self.bally, self.ballx] = 1

        slice_start = self.pos - self.radius
        slice_end = self.pos - self.radius + self.paddle_width
        self.image[-5, slice_start:slice_end] = 1

        terminal = self.bally == self.size - 1 - 4

        if terminal:
            paddle_left_edge = self.pos - self.radius
            paddle_right_edge = self.pos - self.radius + self.paddle_width - 1
            reward = int(paddle_left_edge <= self.ballx <= paddle_right_edge)
        else:
            reward = 0

        [self.state.append(resize(self.image, (84, 84)))
         for _ in range(self.fps - len(self.state) + 1)]
        self.state = self.state[-self.fps:]

        return np.transpose(self.state, [1, 2, 0]), reward, terminal

    def get_num_actions(self):
        return 3

    def reset(self):
        return self.reset_random()

    def state_shape(self):
        return (self.fps,) + self.output_shape


def run_environment():
    env = CatchEnv(paddle_width=5)
    number_of_episodes = 1

    for ep in range(number_of_episodes):
        env.reset()

        state, reward, terminal = env.step(1)

        while not terminal:
            state, reward, terminal = env.step(random.randint(0, 2))
            print("Reward obtained by the agent: {}".format(reward))
            state = np.squeeze(state)

        print("End of the episode")


if __name__ == "__main__":
    run_environment()