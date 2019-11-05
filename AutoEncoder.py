# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import SnakeGame
import matplotlib.pyplot as plt


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(4, 1, 4, 2, 1, bias=False),
        )

    def forward(self, x):
        x = self.main(x)
        return x


model = AutoEncoder()
model.load_state_dict(torch.load('./AEparam.pkl'))
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)
game = SnakeGame.SnakeGame(size=[24, 24])
plt.ion()
fig = plt.figure()

train = True

if not train:
    model.eval()
    fig1 = fig.add_subplot(221)
    fig2 = fig.add_subplot(222)
    fig3 = fig.add_subplot(223)
    fig4 = fig.add_subplot(224)

t = 0
while not game.over:
    t += 1
    x = torch.from_numpy(np.array([[game.getNumpy()]]))

    if train:
        for _ in range(20):
            # forward
            out = model(x)
            loss = criterion(out, x)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if t % 50 == 1:
            plt.imshow(out.data.numpy()[0][0].T, cmap='gray')
            plt.pause(0.1)
        if t == 1:
            plt.pause(2)
    else:
        if t % 50 == 1:
            x = model.main[0](x)
            x = model.main[1](x)
            fig1.imshow(x.data.numpy()[0][0].T, cmap='gray')
            fig2.imshow(x.data.numpy()[0][1].T, cmap='gray')
            fig3.imshow(x.data.numpy()[0][2].T, cmap='gray')
            fig4.imshow(x.data.numpy()[0][3].T, cmap='gray')
            plt.pause(0.1)
        if t == 1:
            plt.pause(2)

    game.render()
    game.getKey()
    game.next()

if train:
    torch.save(model.state_dict(), './AEparam.pkl')
