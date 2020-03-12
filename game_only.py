#! /usr/bin/env python

import torch
import torch.nn as nn
import os
import random
import pygame
import numpy as np
buttonhit = True
def save_gamestate():
    tplayerpos = (player.rect.x, player.rect.y)
    texitpos = (end_rect.x, end_rect.y)
    # print(actions)
    tactions = actions[:]
    # print(tactions)
    return tplayerpos, texitpos, tactions
def load_gamestate(tplayerpos, texitpos, tactions):
    player.rect.x = tplayerpos[0]
    player.rect.y = tplayerpos[1]
    end_rect.x = texitpos[0]
    end_rect.y = texitpos[1]
    # print(tactions)
    actions = tactions[:]

class Player(object):
    
    def __init__(self, x, y):
        self.rect = pygame.Rect(x, y, 5, 5)

    def move(self, dx, dy):
        collidedx = False
        collidedy = False
        # Move each axis separately. Note that this checks for collisions both times.
        if dx != 0:
            collidedx = self.move_single_axis(dx, 0)
        if dy != 0:
            collidedy = self.move_single_axis(0, dy)
        if collidedx or collidedy:
            return True
    
    def move_single_axis(self, dx, dy):
        collided = False
        # Move the rect
        self.rect.x += dx
        self.rect.y += dy

        # If you collide with a wall, move out based on velocity
        for wall in walls:
            if self.rect.colliderect(wall.rect):
                collided = True
                if dx > 0: # Moving right; Hit the left side of the wall
                    self.rect.right = wall.rect.left
                if dx < 0: # Moving left; Hit the right side of the wall
                    self.rect.left = wall.rect.right
                if dy > 0: # Moving down; Hit the top side of the wall
                    self.rect.bottom = wall.rect.top
                if dy < 0: # Moving up; Hit the bottom side of the wall
                    self.rect.top = wall.rect.bottom
        return collided

class ActionRecord:
    def __init__(self):
        self.actions = []
        for action in ACTIONS:
            self.actions.append(0)
        self.inputs = []
                    
class Wall(object):
    def __init__(self, pos):
        walls.append(self)
        self.rect = pygame.Rect(pos[0], pos[1], 7, 7)

class Model(nn.Module):   
    def __init__(self):
        super(Model, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(288, CNN_OUTPUT_LENGTH)
        )
        input_dim = CNN_OUTPUT_LENGTH
        # hidden_dim = 256
        # n_layers = 1
        # self.lstm_layer = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True, dropout=1)
        # self.hidden_state = torch.randn(1, n_layers, hidden_dim)
        # self.cell_state = torch.randn(1, n_layers, hidden_dim)
        self.actor = nn.Sequential(
            nn.Linear(256, 4),
            nn.ReLU(inplace=True),
        )
        self.critic = nn.Linear(260, 1)
        
    def conv(self, x):
        x = x.float()
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
    def act(self, x):
        return self.actor(x)
    def critique(self, x):
        return self.critic(x)
        
# constants
LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3
ACTIONS = [LEFT, RIGHT, UP, DOWN]
# ML stuff
NUM_ROLLOUTS = 10
ROLLOUT_LENGTH = 5
CNN_OUTPUT_LENGTH = 256
BASERANDMOVE = 0.5
DECAYSTEPS = 3000
BASERANDDECAY = BASERANDMOVE / DECAYSTEPS

# conv layer
# convmodel = model.ConvBlock()
model = Model()
# game setup
os.environ["SDL_VIDEO_CENTERED"] = "1"
pygame.init()
pygame.display.set_caption("Maze")
screen = pygame.display.set_mode((84, 84))
clock = pygame.time.Clock()
walls = []
player = None
end_rect = None
button = None
level = [
"WWWWWWWWWWWW",
"W          W",
"W      WW  W",
"W   W      W",
"W   W      W",
"W   W      W",
"W   W      W",
"W   W   WWWW",
"W          W",
"W      W   W",
"W      W   W",
"WWWWWWWWWWWW",
]
CLEAR_SPACES = 88
x = y = 0
count = 0
playerspawn = random.randint(0, CLEAR_SPACES)
endspawn = random.randint(0, CLEAR_SPACES)
keyspawn = random.randint(0, CLEAR_SPACES)
while playerspawn == endspawn:
    endspawn = random.randint(0, CLEAR_SPACES)
while playerspawn == keyspawn or endspawn == keyspawn:
    keyspawn = random.randint(0, CLEAR_SPACES)
buttonhit = True
for row in level:
    for col in row:
        if col == "W":
            Wall((x, y))
        elif not player or not end_rect or not button:
            if not player:
                if count == playerspawn:
                    player = Player(x, y)
            if not end_rect:
                if count == endspawn:
                    end_rect = pygame.Rect(x, y, 7, 7)
            if not button:
                if count == keyspawn:
                    button = pygame.Rect(x, y, 7, 7)
            count += 1
        x += 7
    y += 7
    x = 0

score = 0
allscores = 0
runs = 0
lastscore = 0
lastdistance = -1
history = []
actions = []
running = True
agentaction = None
while running:
    clock.tick(60)    
    lastscore = score
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
            running = False
    something_pressed = False
    key = pygame.key.get_pressed()
    # insert agent decisions here?
    agentleft = False
    agentright = False
    agentup = False
    agentdown = False
    # if agentaction:
        # if agentaction[LEFT]:
            # agentleft = True
        # if agentaction[RIGHT]:
            # agentright = True
        # if agentaction[UP]:
            # agentup = True
        # if agentaction[DOWN]:
            # agentdown = True
        # agentaction = None
    # process game controls
    collided = False
    if key[pygame.K_LEFT] or agentleft:
        collided = collided or player.move(-1, 0)
        something_pressed = True
    if key[pygame.K_RIGHT] or agentright:
        collided = collided or player.move(1, 0)
        something_pressed = True
    if key[pygame.K_UP] or agentup:
        collided = collided or player.move(0, -1)
        something_pressed = True
    if key[pygame.K_DOWN] or agentdown:
        collided = collided or player.move(0, 1)
        something_pressed = True
    if something_pressed:
        action = ActionRecord()
        if key[pygame.K_LEFT]:
            action.actions[LEFT] = 1
        if key[pygame.K_RIGHT]:
            action.actions[RIGHT] = 1
        if key[pygame.K_UP]:
            action.actions[UP] = 1
        if key[pygame.K_DOWN]:
            action.actions[DOWN] = 1
        actions.append(action)
        score -= 1
    # if collided:
        # score -= 1
    
    # for taking a step
    
    # score diff from being closer or further from goal
    if not buttonhit:
        goalx = button.x
        goaly = button.y
    elif buttonhit:
        goalx = end_rect.x
        goaly = end_rect.y
    px = player.rect.x
    py = player.rect.y
    distancefromgoal = (px-goalx)**2+(py-goaly)**2
    if lastdistance == -1:
        # forgive
        donothing = 1
    if distancefromgoal >= lastdistance:
        score -= 1
    lastdistance = distancefromgoal
            
    # collision with button
    if player.rect.colliderect(button) and not buttonhit:
        score += 10000
        buttonhit = True
        
    # reset if reach goal
    if (player.rect.colliderect(end_rect) and buttonhit) or len(actions) > 5000:
        lastdistance = -1
        if len(actions) <= 5000:
            score += 5000
        print(score)
        allscores += score
        runs += 1
        if runs % 10 == 0:
            print("{} runs, av score = {}".format(runs, allscores/runs))
        score = 0
        xtemp = ytemp = 0
        print("resetting")
        history.append(actions)
        print("{} actions".format(len(actions)))
        actions.clear()
        count = 0
        playerspawn = random.randint(0, CLEAR_SPACES)
        endspawn = random.randint(0, CLEAR_SPACES)
        keyspawn = random.randint(0, CLEAR_SPACES)
        while playerspawn == endspawn:
            endspawn = random.randint(0, CLEAR_SPACES)
        while playerspawn == keyspawn or endspawn == keyspawn:
            keyspawn = random.randint(0, CLEAR_SPACES)
        # buttonhit = False
        for row in level:
            for col in row:
                if col == " ":
                    if count == playerspawn:
                        player.rect.x = xtemp
                        player.rect.y = ytemp
                    if count == endspawn:
                        end_rect.x = xtemp
                        end_rect.y = ytemp
                    if count == keyspawn:
                        button.x = xtemp
                        button.y = ytemp
                    count += 1
                xtemp += 7
            ytemp += 7    
            xtemp = 0
    
    # Draw the scene
    screen.fill((0, 0, 0))
    for wall in walls:
        x = wall.rect.left
        y = wall.rect.top
        px = player.rect.left
        py = player.rect.top
        distance = (x-px)**2 + (y-py)**2
        if distance < 800000:
            pygame.draw.rect(screen, (255, 255, 255), wall.rect)
    pygame.draw.rect(screen, (255, 0, 0), end_rect)
    pygame.draw.rect(screen, (255, 200, 0), player.rect)
    pixels = pygame.surfarray.array3d(screen)
    # rgb = [0,1,2]
    # pixelsp = []
    # for c in rgb:
        # channel = []
        # for row in pixels:
            # channelrow = []
            # for col in row:
                # channelrow.append(col[c])
            # channel.append(channelrow)
        # pixelsp.append(channel)
    # pixelsnp = np.array(pixelsp)
    # pixelsr = pixelsnp.reshape(1, 3, 84, 84)
    # pixeltensor = torch.from_numpy(pixelsr)
    # observation = model.conv(pixeltensor)
    # action = model.act(observation)
    # a,b,c = save_gamestate()
    # obs = observation[0]
    # obs = torch.cat((obs, action[0]), -1)
    # observation = obs.unsqueeze(0)
    # feedback = model.critique(observation)

    # print(c)
    # simulate a gamestate change
    
    # print(loss)
    # loss.backward()
    # print(feedback)
    
    # print("LOADED")
    # load_gamestate(a,b,c)
    # agentaction = []
    # for act in action.tolist()[0]:
        # print(act)
        # actprob = act
        # randmovechance = BASERANDMOVE - BASERANDDECAY * len(actions)
        # if random.uniform(0,1) < actprob or random.uniform(0,1) < randmovechance:
            # agentaction.append(True)
        # else:
            # agentaction.append(False)
    # print(agentaction)
    # print(value)
    # advantage = torch.tensor(0., requires_grad=True)
    # advantage = advantage + torch.tensor(score) - torch.tensor(lastscore)
    # advantage = advantage.unsqueeze(0)
    # loss = (advantage - value)**2
    # print("action: {} / advantage: {}".format(len(actions), score-lastscore))
    # print(pixels)
    pygame.display.flip()
    
# after game over