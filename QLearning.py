# -*- coding: utf-8 -*-

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import SnakeGame
import matplotlib.pyplot as plt

SIZE = 8


class QLearning(nn.Module):
    def __init__(self):
        super(QLearning, self).__init__()
        self.conv1 = nn.Conv2d(2, 3, 4, 1, 2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(3, 5, 4, 1, 2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(5, 9, 4, 1, 2)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.Qlinear = nn.Linear(36, 4)

    def forward(self, p, lp):
        pic = torch.cat([p, lp], 1)
        pic = self.conv1(pic)
        pic = torch.nn.functional.relu(pic, inplace=True)
        pic = self.pool1(pic)
        pic = self.conv2(pic)
        pic = torch.nn.functional.relu(pic, inplace=True)
        pic = self.pool2(pic)
        pic = self.conv3(pic)
        pic = torch.nn.functional.relu(pic, inplace=True)
        pic = self.pool3(pic)
        pic = pic.view(pic.size(0), -1)
        Q = self.Qlinear(pic)
        return Q


model = QLearning()
model.load_state_dict(torch.load('./QLparam.pkl'))
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)
game = SnakeGame.SnakeGame(size=[SIZE, SIZE])
lp = torch.from_numpy(np.array([[game.getNumpy()]]))
DRAW = True

ii = 0
if DRAW:
    plt.ion()
    fig = plt.figure()
    ti = np.array([])
    ts = np.array([])
    line, = plt.plot(ti, ts)

t = 0
avr = 0
while not game.over:
    t += 1
    p = torch.from_numpy(np.array([[game.getNumpy()]]))

    if DRAW:
        game.render(wait=False)
    game.getKey()
    game.next()

    nQ = torch.Tensor([game.Q])
    for _ in range(1):
        Q = model(p, lp)
        loss = criterion(Q[0][game.direction - 1], nQ[0])
        loss.backward()
        optimizer.step()

    lp = p
    with torch.no_grad():
        model.eval()
        Q = model(p, lp)
        Qs = np.zeros(4)
        for i in range(4):
            Qs[i] = Q[0][i] + 1
        Rnum = random.uniform(0, np.sum(Qs))
        Qsum = 0
        for i in range(4):
            Qsum += Qs[i]
            if Qsum > Rnum:
                game.direction = i + 1
                break
        model.train()

    if game.down:
        optimizer.zero_grad()
        ii += 1
        if DRAW:
            ti = np.append(ti, ii)
            ts = np.append(ts, t)
            line.set_xdata(ti)
            line.set_ydata(ts)
            plt.axis([0, ii, 0, 100])
            plt.pause(0.1)
        else:
            avr += 0.002 * t
            if (not ii % 500):
                print(avr)
                ii = 0
                avr = 0
                torch.save(model.state_dict(), './QLparam.pkl')
        t = 0
        game.restart()

torch.save(model.state_dict(), './QLparam.pkl')
