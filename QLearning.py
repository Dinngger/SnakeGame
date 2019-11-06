# -*- coding: utf-8 -*-

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import SnakeGame
import matplotlib.pyplot as plt

DECAY = math.exp(-2)
SIZE = 4
CONVDEPTH = 4
HIDESIZE = 2 * 2 * CONVDEPTH
OUTSIZE = 21


class QLearning(nn.Module):
    def __init__(self):
        super(QLearning, self).__init__()
        self.conv = nn.Conv2d(1, CONVDEPTH, 3, 1, bias=False)
        self.linear = nn.Linear(HIDESIZE + 5, OUTSIZE)
        self.Qlinear = nn.Linear(OUTSIZE, 1)
        self.cov = torch.zeros((OUTSIZE, OUTSIZE))
        self.linearT = nn.Linear(OUTSIZE, HIDESIZE + 5)
        self.convT = nn.ConvTranspose2d(CONVDEPTH, 1, 3, 1, bias=False)

    def forward(self, p, a, r):
        pic = self.conv(p).view(1, HIDESIZE)
        pic = torch.sigmoid(pic)
        h = torch.cat((pic, a, r), 1)
        x = self.linear(h)
        x = torch.tanh(x)  # + torch.transpose(self.cov.mm(torch.transpose(x, 0, 1)), 0, 1))
        Q = torch.tanh(self.Qlinear(x))
        h2 = self.linearT(x)
        a2 = h2[0, HIDESIZE:HIDESIZE + 4].view((1, 4))
        r2 = h2[0, HIDESIZE + 4].view((1, 1))
        pic2 = h2[0, 0:HIDESIZE].view((1, CONVDEPTH, 2, 2))
        p2 = self.convT(pic2)
        return (p2, a2, r2, h, h2, Q, x)


model = QLearning()
# model.load_state_dict(torch.load('./QLparam.pkl'))
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)
game = SnakeGame.SnakeGame(size=[SIZE, SIZE])

plt.ion()
fig = plt.figure()

t = 0
while not game.over:
    t += 1
    p = torch.from_numpy(np.array([[game.getNumpy()]]))
    a = torch.zeros((1, 4)).scatter_(1, torch.LongTensor([[game.direction - 1]]), 1)
    r = torch.Tensor([[game.Q]])

    # game.render(wait=False)
    game.getKey()
    game.next()

    nQ = torch.Tensor([[game.Q]])
    for _ in range(1):
        for i in range(1):
            p2, a2, r2, h, h2, Q, x = model(p, a, r)
            loss = criterion(p2, p)
            # loss = criterion(h2, h) if i == 0 else (
            #        criterion(p2, p) if i == 1 else
            #        criterion(a2, a) if i == 2 else
            #        criterion(r2, r) if i == 3 else
            #        criterion(Q, nQ))
            # optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if t % 300 == 1:
        plt.imshow(p2.data.numpy()[0][0].T, cmap='gray')
        plt.pause(0.1)
        t = 1

    with torch.no_grad():
        model.eval()
        Qs = np.zeros(4)
        for i in range(4):
            a = torch.zeros((1, 4)).scatter_(1, torch.LongTensor([[i]]), 1)
            p2, a2, r2, h, h2, Q, x = model(p, a, r)
            Qs[i] = Q[0] + 1
        Rnum = random.uniform(0, np.sum(Qs))
        Qsum = 0
        for i in range(4):
            Qsum += Qs[i]
            if Qsum > Rnum:
                game.direction = i + 1
                break
        model.train()

    if game.down:
        game.restart()

torch.save(model.state_dict(), './QLparam.pkl')
