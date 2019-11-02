import pygame
import numpy as np
import random


class SnakeGame:

    class Direction:
        UNKNOW = 0
        RIGHT = 1
        LEFT = 2
        UP = 3
        DOWN = 4

    def __init__(self, size=[32, 24], block_size=20):
        self.block_size = block_size
        self.size = size
        self.display = False
        pygame.init()
        self.over = False

        self.snakePosition = [size[0] // 2, size[1] // 2]
        self.snakeSegments = [[size[0] // 2, size[1] // 2]]
        self.raspberryPosition = [size[0] // 2, size[1] // 2]
        self.raspberrySpawned = 1
        self.direction = self.Direction.RIGHT

    def gameOver(self):
        print("Game Over!")
        pygame.quit()
        self.over = True

    def getKey(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.gameOver()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT or event.key == ord('d'):
                    self.direction = self.Direction.RIGHT
                if event.key == pygame.K_LEFT or event.key == ord('a'):
                    self.direction = self.Direction.LEFT
                if event.key == pygame.K_UP or event.key == ord('w'):
                    self.direction = self.Direction.UP
                if event.key == pygame.K_DOWN or event.key == ord('s'):
                    self.direction = self.Direction.DOWN
                if event.key == pygame.K_ESCAPE:
                    pygame.event.post(pygame.event.Event(pygame.QUIT))

    def next(self):
        # 根据方向移动蛇头的坐标
        if self.direction == self.Direction.RIGHT:
            self.snakePosition[0] += 1
        if self.direction == self.Direction.LEFT:
            self.snakePosition[0] -= 1
        if self.direction == self.Direction.UP:
            self.snakePosition[1] -= 1
        if self.direction == self.Direction.DOWN:
            self.snakePosition[1] += 1
        # 增加蛇的长度
        self.snakeSegments.insert(0, list(self.snakePosition))
        # 判断是否吃掉了树莓
        if self.snakePosition[0] == self.raspberryPosition[0] and self.snakePosition[1] == self.raspberryPosition[1]:
            self.raspberrySpawned = 0
        else:
            self.snakeSegments.pop()
        # 如果吃掉树莓，则重新生成树莓
        if self.raspberrySpawned == 0:
            x = random.randrange(1, self.size[0])
            y = random.randrange(1, self.size[1])
            ok = False
            while not ok:
                ok = True
                for snakeBody in self.snakeSegments:
                    if x == snakeBody[0] and y == snakeBody[1]:
                        x = random.randrange(1, self.size[0])
                        y = random.randrange(1, self.size[1])
                        ok = False
                        break
            self.raspberryPosition = [int(x), int(y)]
            self.raspberrySpawned = 1

        # 判断是否死亡，后面几行和原文有改动
        if self.snakePosition[0] >= self.size[0] or self.snakePosition[0] < 0:
            self.gameOver()
        if self.snakePosition[1] >= self.size[1] or self.snakePosition[1] < 0:
            self.gameOver()
        for snakeBody in self.snakeSegments[1:]:
            if self.snakePosition[0] == snakeBody[0] and self.snakePosition[1] == snakeBody[1]:
                self.gameOver()

    def render(self):
        if not self.display:
            self.display = True
            self.redColour = pygame.Color(255, 0, 0)
            self.blackColour = pygame.Color(0, 0, 0)
            self.whiteColour = pygame.Color(255, 255, 255)
            self.greyColour = pygame.Color(150, 150, 150)
            self.playSurface = pygame.display.set_mode(
                (self.size[0] * self.block_size, self.size[1] * self.block_size),
                pygame.DOUBLEBUF)
            self.fpsClock = pygame.time.Clock()
            pygame.display.set_caption('Snake')
        # 刷新pygame显示层
        self.playSurface.fill(self.blackColour)
        for position in self.snakeSegments:
            pygame.draw.rect(self.playSurface, self.whiteColour,
                             pygame.Rect(position[0] * self.block_size, position[1] * self.block_size,
                                         self.block_size, self.block_size))
        pygame.draw.rect(self.playSurface, self.redColour,
                         pygame.Rect(self.raspberryPosition[0] * self.block_size, self.raspberryPosition[1] * self.block_size,
                                     self.block_size, self.block_size))
        pygame.display.flip()
        self.fpsClock.tick(10)

    def getNumpy(self):
        surface = np.zeros(self.size, dtype=np.float32)
        for position in self.snakeSegments:
            surface[position[0], position[1]] = 1
        surface[self.raspberryPosition[0], self.raspberryPosition[1]] = 1
        return surface


if __name__ == "__main__":
    game = SnakeGame()
    while not game.over:
        game.render()
        game.getKey()
        game.next()
