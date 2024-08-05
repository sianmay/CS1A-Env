#import gym
import gymnasium as gym
from gymnasium import spaces
#from gym import spaces
import pygame
import numpy as np

class MyEnv6(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=10, n_seasons=4, lifetime=100, loc=False):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        n_obs = 27
        self.loc = loc
        if loc:
            n_obs = 28
        #self.observation_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(n_obs,), dtype=np.float64)

        # We have 5 actions, corresponding to "right", "up", "left", "down", "eat"
        self.action_space = spaces.Discrete(5)
        
        self.lifetime = lifetime
        self.n_seasons = n_seasons
        self.change_season = int(lifetime/n_seasons)
        
        green = [0,1,0]
        red = [1,0,0]
        yellow = [1,1,0]
        blue= [0,0,1]
        orange = [1,0.65,0]
        purple = [0.5, 0, 0.5]
        brown = [0.4, 0.2, 0.0]
        pink = [1, 0.4, 1]
        
        self.edibles = [green, yellow, orange, brown]
        self.poisons = [red, blue, purple, pink]

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
            4: np.array([0, 0])
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        
    def _get_obs(self):
        x,y = self._agent_location
        obs = self.scene[x-1:x+2,y-1:y+2].reshape([9])
        obs_ = np.zeros([9,2])
        edible, poison = 1, 2
        for i in range(len(obs)):
            if obs[i] == 3:
                obs_[i,0] = 1
                obs_[i,1] = 1
            elif obs[i] > 0:
                obs_[i,int(obs[i]-1)] = 1
        #print(obs)
        obs = obs.reshape([9,])
        obs_ = obs_.reshape([18,])
        obs_r = self.red_scene[x-1:x+2,y-1:y+2].reshape([9])
        obs_g = self.green_scene[x-1:x+2,y-1:y+2].reshape([9])
        obs_b = self.blue_scene[x-1:x+2,y-1:y+2].reshape([9])
        obs_rgb = np.array([obs_r,obs_g,obs_b]).reshape([27])
        if self.loc:
            loc = (len(self.scene)*x)+y
            loc = loc/(len(self.scene)**2)
            obs_rgb = np.append(obs_rgb, loc)
        return obs_rgb#_loc

    
    def _get_info(self):
        return {"total_reward": self.acc_reward, "edible foods": self.edible_foods, "poisonous foods": self.poison_foods, "season": self.season}
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.timestep = 0
        self.acc_reward = 0
        self.season = 0
        self.edible_rgb = self.edibles[0]
        self.poison_rgb = self.poisons[0]
        self.edible_foods = 0
        self.poison_foods = 0

        neg_col, neg_row = 3*np.ones([self.size+2,1]), 3*np.ones([1,self.size])
        zeros = np.zeros([self.size, self.size])
        arr = np.append(np.append(neg_row, zeros, axis=0), neg_row, axis=0)
        arr = np.append(np.append(neg_col, arr, axis=1), neg_col, axis=1)
        self.scene = arr
        self.red_scene = np.copy(self.scene)/3
        self.green_scene = np.copy(self.scene)/3
        self.blue_scene = np.copy(self.scene)/3
        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size+2, size=2, dtype=int)
        while(self.scene[self._agent_location[0]][self._agent_location[1]] != 0):
            self._agent_location = self.np_random.integers(0, self.size+2, size=2, dtype=int)
        
        # add edible foods
        self._target_location = self._agent_location
        for i in range(10):
            self.place_food(1)
            
        # add poisonous foods
        for i in range(10):
            self.place_food(2)
            
        #self.rgb_scene()

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )
            
        self._poison_location = self._target_location
        while np.array_equal(self._poison_location, self._agent_location) or np.array_equal(self._poison_location, self._target_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        #if self.render_mode == "human":
        #    self._render_frame()

        return observation, info
    
    def place_food(self, val):#, edible=[0.4,0.2,0], poison=[1,0.4,1]):
        e, p = 1, 2
        edible = self.edible_rgb
        poison = self.poison_rgb
        self._target_location = self.np_random.integers(0, self.size+2, size=2, dtype=int)
        while(self.scene[self._target_location[0]][self._target_location[1]] != 0):
                self._target_location = self.np_random.integers(0, self.size+2, size=2, dtype=int)
        self.scene[self._target_location[0]][self._target_location[1]] = val
        i, j = self._target_location[0], self._target_location[1]
        if val == e:
                self.red_scene[i,j] = edible[0]
                self.green_scene[i,j] = edible[1]
                self.blue_scene[i,j] = edible[2]
        elif val == p:
                self.red_scene[i,j] = poison[0]
                self.green_scene[i,j] = poison[1]
                self.blue_scene[i,j] = poison[2]
        
    def rgb_scene(self):#, edible=[0.4,0.2,0], poison=[1,0.4,1]):
        e, p = 1, 2
        edible = self.edible_rgb
        poison = self.poison_rgb
        for i in range(len(self.scene)):
            for j in range(len(self.scene)):
                if self.scene[i,j] == e:
                    self.red_scene[i,j] = edible[0]
                    self.green_scene[i,j] = edible[1]
                    self.blue_scene[i,j] = edible[2]
                elif self.scene[i,j] == p:
                    self.red_scene[i,j] = poison[0]
                    self.green_scene[i,j] = poison[1]
                    self.blue_scene[i,j] = poison[2]
                    
    
    def step(self, action):
        self.timestep += 1
        old_loc = self._agent_location
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 1, self.size
        )
        # An episode is done iff the agent has reached the target
        #terminated = np.array_equal(self._agent_location, self._target_location) or np.array_equal(self._agent_location, self._poison_location)
        #edible = np.array_equal(self._agent_location, self._target_location)
        x, y = self._agent_location[0], self._agent_location[1]
        #print("x, y: ", x,y)
        edible, poison, wall = False, False, False
        if action == 4:
            # agent ate edible food
            if self.scene[x,y] == 1:
                edible = True
                self.scene[x,y] = 0
                self.red_scene[x,y] = 0
                self.green_scene[x,y] = 0
                self.blue_scene[x,y] = 0
                self.place_food(1)
            elif self.scene[x,y] == 2:
                # agent ate poisonous food
                poison = True
                self.scene[x,y] = 0
                self.red_scene[x,y] = 0
                self.green_scene[x,y] = 0
                self.blue_scene[x,y] = 0
                self.place_food(2)
        #if np.array_equal(old_loc, self._agent_location):
         #   wall = True
        terminated = False
        if (self.timestep == 100):# or self.acc_reward <= 0):
            terminated = True
        #reward = 1 if edible else 0  # Binary sparse rewards
        reward = -0.01
        if edible:
            reward = 1
            self.edible_foods += 1
        elif poison:
            reward = -1
            self.poison_foods += 1
        #elif wall:
         #   reward = -0.1
            
        self.acc_reward += reward
        
        observation = self._get_obs()
        info = self._get_info()
        
        if not terminated and self.timestep == self.change_season*(self.season+1):
            self.season += 1
            self.edible_rgb = self.edibles[self.season]
            self.poison_rgb = self.poisons[self.season]
            self.rgb_scene()

        #if self.render_mode == "human":
        #    self._render_frame()

        return observation, reward, terminated, False, info
    
    def render(self):
        return self._render_frame()
        #if self.render_mode == "rgb_array":
        #    return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / len(self.scene)
        )  # The size of a single grid square in pixels

        for i in range(len(self.scene)):
            for j in range(len(self.scene)):
                #print(255*self.red_scene[i,j])
                pygame.draw.rect(canvas,(255*self.red_scene[i,j], 255*self.green_scene[i,j], 255*self.blue_scene[i,j]),
                                     pygame.Rect(pix_square_size * np.array([i,j]),(pix_square_size, pix_square_size),),)
            
            
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (255, 255, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 3):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()