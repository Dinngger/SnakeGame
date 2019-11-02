# encoding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)
y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)
y_train += 3
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)


# Linear Regression Model
class linearRegression(nn.Module):
    def __init__(self):
        super(linearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # input and output is 1 dimension

    def forward(self, x):
        out = self.linear(x)
        return out


model = linearRegression()
# 定义loss和优化函数
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

# 开始训练
num_epochs = 2000
epochs = np.array([])
losses = np.array([])
for epoch in range(num_epochs):
    inputs = x_train
    target = y_train

    # forward
    out = model(inputs)
    loss = criterion(out, target)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 200 == 0:
        print(f'Epoch[{epoch+1}/{num_epochs}], loss: {loss.item():.6f}')
        epochs = np.append(epochs, [epoch / 100])
        losses = np.append(losses, [loss.item()])

model.eval()
with torch.no_grad():
    predict = model(x_train)
predict = predict.data.numpy()

fig = plt.figure(figsize=(10, 5))
fig1 = fig.add_subplot(121)
fig1.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original data')
fig1.plot(x_train.numpy(), predict, label='Fitting Line')
fig1.legend()
fig2 = fig.add_subplot(122)
fig2.plot(epochs, losses, label='loss')
fig2.legend()
# 显示图例
plt.show()
