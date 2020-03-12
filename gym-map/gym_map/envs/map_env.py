import pygame
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import random
from gym import spaces
import os
from datetime import datetime

def dist_sq(x,y,ox,oy):
    return (x-ox)**2+(y-oy)**2

def move_player(player, dx, dy, walls):
    collidedx = False
    collidedy = False
    # Move each axis separately. Note that this checks for collisions both times.
    if dx != 0:
        player, collidedx = move_single_axis(player, dx, 0, walls)
    if dy != 0:
        player, collidedy = move_single_axis(player, 0, dy, walls)
    collided = False
    if collidedx or collidedy:
        collided = True
    return player, collided
    
def move_single_axis(player, dx, dy, walls):
    collided = False
    # Move the rect
    player.x += dx
    player.y += dy

    # If you collide with a wall, move out based on velocity
    for wall in walls:
        if player.colliderect(wall):
            collided = True
            if dx > 0: # Moving right; Hit the left side of the wall
                player.right = wall.left
            if dx < 0: # Moving left; Hit the right side of the wall
                player.left = wall.right
            if dy > 0: # Moving down; Hit the top side of the wall
                player.bottom = wall.top
            if dy < 0: # Moving up; Hit the bottom side of the wall
                player.top = wall.bottom
    return player, collided
                   
class Wall(object):
    def __init__(self, pos):
        walls.append(self)
        self.rect = pygame.Rect(pos[0], pos[1], 7, 7)

class MapEnv(gym.Env):  
    metadata = {'render.modes': ['human']}   
    def __init__(self):
        self.REN = 1.0
        self.overhowmany = 200000
        self.decay = self.REN/self.overhowmany
        self.score = 0
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 3))
        os.environ["SDL_VIDEO_CENTERED"] = "1"
        pygame.init()
        pygame.display.set_caption("Maze")
        self.screen = pygame.display.set_mode((84,84))
        # self.level = [
        # "WWWWWWWWWWWW",
        # "W          W",
        # "W      WW  W",
        # "W   W      W",
        # "W   W      W",
        # "W   W      W",
        # "W   W      W",
        # "W   W   WWWW",
        # "W          W",
        # "W      W   W",
        # "W      W   W",
        # "WWWWWWWWWWWW",
        # ]        
        self.level = [
        "WWWWWWWWWWWW",
        "W          W",
        "W          W",
        "W          W",
        "W          W",
        "W          W",
        "W          W",
        "W          W",
        "W          W",
        "W          W",
        "W          W",
        "WWWWWWWWWWWW",
        ]
        self.CLEAR_SPACES = 80
        x = y = 0
        count = 0
        self.player = None
        self.exit = None
        self.walls = []
        playerspawn = random.randint(0, self.CLEAR_SPACES)
        endspawn = random.randint(0, self.CLEAR_SPACES)
        while playerspawn == endspawn:
            endspawn = random.randint(0, self.CLEAR_SPACES)
        for row in self.level:
            for col in row:
                if col == "W":
                    self.walls.append(pygame.Rect(x, y, 7, 7))
                elif not self.player or not self.exit:
                    if not self.player:
                        if count == playerspawn:
                            # print("spawned player")
                            self.player = pygame.Rect(x, y, 5, 5)
                    if not self.exit:
                        if count == endspawn:
                            # print("spawned exit")
                            self.exit = pygame.Rect(x, y, 7, 7)
                    count += 1
                x += 7
            y += 7
            x = 0
        
        
    def step(self, action):
        # action is array [left, right, up, down, left up, up right, right down, down left]
        # ideal directions
        self.REN -= self.decay
        useideal = False
        idealright = False
        idealleft = False
        idealup = False
        idealdown = False
        if self.player.x < self.exit.x:
            idealright = True
        if self.player.x > self.exit.x:
            idealleft = True
        if self.player.y < self.exit.y:
            idealdown = True
        if self.player.y > self.exit.y:
            idealup = True
        
        self.score -= 1
        done = False
        reward = 0
        # act = self.action_space.sample()
        act = action
        # print(act)
        agentleft = False
        agentright = False
        agentup = False
        agentdown = False
        if act == 0:
            agentleft = True
        if act == 1:
            agentright = True
        if act == 2:
            agentup = True
        if act == 3:
            agentdown = True
            
        if useideal:
            agentleft = idealleft
            agentright = idealright
            agentup = idealup
            agentdown = idealdown
        # process game controls
        icollided = False
        if agentleft:
            self.player, collided = move_player(self.player, -1, 0, self.walls)
            icollided = icollided or collided
        if agentright:
            self.player, collided = move_player(self.player, 1, 0, self.walls)
            icollided = icollided or collided
        if agentup:
            self.player, collided = move_player(self.player, 0, -1, self.walls)
            icollided = icollided or collided
        if agentdown:
            self.player, collided = move_player(self.player, 0, 1, self.walls)
            icollided = icollided or collided
        if icollided:
            reward -= 0
        if self.player.colliderect(self.exit):
            reward += 5000
            self.score += 5000
            done = True
        distance = dist_sq(self.player.x,self.player.y,self.exit.x,self.exit.y)
        if self.lastdistance == -1:
            donothing = 1
        else:
            if distance >= self.lastdistance:
                reward -= 0
            else:
               reward += 0
        self.lastdistance = distance
        reward -= distance/10000 * self.REN
        
        # observation
        self.screen.fill((0, 0, 0))
        for wall in self.walls:
            x = wall.x
            y = wall.y
            px = self.player.x
            py = self.player.y
            distance = (x-px)**2 + (y-py)**2
            if distance < 800:
                pygame.draw.rect(self.screen, (255, 255, 255), wall)
        exitdist = dist_sq(self.player.x,self.player.y,self.exit.x,self.exit.y)
        if exitdist < 800000:
            pygame.draw.rect(self.screen, (255, 0, 0), self.exit)
        pygame.draw.rect(self.screen, (255, 200, 0), self.player)
        pixels = pygame.surfarray.array3d(self.screen)
        info = {}
        if self.score < -5000:
            done = True
        if done:
            print("DONE: {}".format(self.score))
        return pixels, reward, done, info
        # Box(0, 255, [height, width, 3]
 
    def reset(self):
        random.seed(datetime.now())
        self.score = 0
        self.lastdistance = -1
        xtemp = 0
        ytemp = 0
        count = 0
        playerspawn = random.randint(0, self.CLEAR_SPACES)
        endspawn = random.randint(0, self.CLEAR_SPACES)
        while playerspawn == endspawn:
            endspawn = random.randint(0, self.CLEAR_SPACES)
        for row in self.level:
            for col in row:
                if col == " ":
                    if count == playerspawn:
                        self.player.x = xtemp
                        self.player.y = ytemp
                    if count == endspawn:
                        self.exit.x = xtemp
                        self.exit.y = ytemp
                    count += 1
                xtemp += 7
            ytemp += 7    
            xtemp = 0            
        # observation
        self.screen.fill((0, 0, 0))
        for wall in self.walls:
            x = wall.x
            y = wall.y
            px = self.player.x
            py = self.player.y
            distance = (x-px)**2 + (y-py)**2
            if distance < 800:
                pygame.draw.rect(self.screen, (255, 255, 255), wall)
        exitdist = dist_sq(self.player.x,self.player.y,self.exit.x,self.exit.y)
        if exitdist < 800000:
            pygame.draw.rect(self.screen, (255, 0, 0), self.exit)
        pygame.draw.rect(self.screen, (255, 200, 0), self.player)
        pixels = pygame.surfarray.array3d(self.screen)
        return pixels
        
 
    def render(self, mode='human', close=False):
        RENDER = False
        if RENDER:
            self.screen.fill((0, 0, 0))
            for wall in self.walls:
                x = wall.x
                y = wall.y
                px = self.player.x
                py = self.player.y
                distance = (x-px)**2 + (y-py)**2
                if distance < 800:
                    pygame.draw.rect(self.screen, (255, 255, 255), wall)
        exitdist = dist_sq(self.player.x,self.player.y,self.exit.x,self.exit.y)
        if exitdist < 800000:
            pygame.draw.rect(self.screen, (255, 0, 0), self.exit)
            pygame.draw.rect(self.screen, (255, 200, 0), self.player)
            pygame.display.flip()
