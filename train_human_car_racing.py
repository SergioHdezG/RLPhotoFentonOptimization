from environments import CarRacing
from IRL_Problem.base.utils.callbacks import Callbacks
import pygame
import numpy as np
from collections import deque


class game_loop():
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Car tutorial")
        width = 192
        height = 192
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.ticks = 60
        self.exit = False
        self.n_stack = 1

    def run(self):
        pygame.display.init()
        pygame.joystick.init()
        pygame.joystick.Joystick(0).init()

        cb = Callbacks()
        env = CarRacing.env()
        obs = env.reset()
        flip_obs = obs
        # image = pygame.surfarray.make_surface(obs)
        # self.screen.blit(image, (0, 0))
        # pygame.display.flip()
        pygame.display.update()
        self.clock.tick(self.ticks)

        images_list = []
        images_list.append(obs)

        if self.n_stack is not None and self.n_stack > 1:
            obs_queue = deque(maxlen=self.n_stack)
            obs_queue_flip = deque(maxlen=self.n_stack)

        if self.n_stack is not None and self.n_stack > 1:
            for i in range(self.n_stack):
                obs_queue.append(obs)
                obs_queue_flip.append(flip_obs)

        done = False
        # Prints the values for axis0
        while not self.exit:
            # Event queue
            # for event in pygame.event.get():
            #     if event.type == pygame.QUIT:
            #         self.exit = True
            #
            # User input

            if done:
                obs = env.reset()

            pressed = pygame.key.get_pressed()
            if pressed[pygame.K_ESCAPE]:
                self.exit = True
            if pressed[pygame.K_SPACE]:
                env.close()
                env = CarRacing.env()
                obs = env.reset()
                if self.n_stack is not None and self.n_stack > 1:
                    for i in range(self.n_stack):
                        obs_queue.append(obs)
                        obs_queue_flip.append(flip_obs)

            pygame.event.pump()
            steer = pygame.joystick.Joystick(0).get_axis(0) #/3
            throttle = -pygame.joystick.Joystick(0).get_axis(2)
            # throttle = -pygame.joystick.Joystick(0).get_axis(2)*2 -1
            # if throttle < 1e-10:
            #     throttle = 0.
            brake = -pygame.joystick.Joystick(0).get_axis(3)
            # if brake < 1e-10:
            #     brake = -1.


            print('steer: ', steer, 'throttle: ', throttle, 'brake: ', brake)
            # image = env.render(only_return=True)
            env.render()
            next_obs, flip_obs_next, reward, done, _ = env.step([steer, throttle, brake], mirror=True)

            if self.n_stack is not None and self.n_stack > 1:
                obs_queue.append(next_obs)
                obs_queue_flip.append(flip_obs_next)
                obs_next_stack = np.array(obs_queue).reshape(-1, self.n_stack)
                obs_next_stack_flip = np.array(obs_queue_flip).reshape(-1, self.n_stack)
                cb.remember_callback(obs, obs_next_stack, [steer, throttle, brake], reward, done)
                cb.remember_callback(flip_obs, obs_next_stack_flip, [-steer, throttle, brake], reward, done)
            else:
                cb.remember_callback(obs, next_obs, [steer, throttle, brake], reward, done)
                # cb.remember_callback(flip_obs, flip_obs_next, [-steer, throttle, brake], reward, done)

            obs = next_obs
            flip_obs = flip_obs_next
            # images_list.append(obs)
            # Drawing
            # image = pygame.surfarray.make_surface(obs)
            # # self.screen.fill((0, 0, 0))
            # self.screen.blit(image, (0, 0))
            # pygame.display.flip()
            pygame.display.update()
            self.clock.tick(self.ticks)

        print('fin')
        # save to npy file
        # save('expert_demonstrations/human_expert_CarRacing_images_3.npy', images_list)
        cb.memory_to_csv('expert_demonstrations/', 'human_expert_CarRacing_v2')

def main_2():
    game = game_loop()
    game.run()


main_2()

# from numpy import load
# # load array
# data_1 = load('/home/serch/AIRL/expert_demonstrations/human_expert_CarRacing_images.npy')
# data_2 = load('/home/serch/AIRL/expert_demonstrations/human_expert_CarRacing_images_2.npy')
# data_3 = load('/home/serch/AIRL/expert_demonstrations/human_expert_CarRacing_images_3.npy')
# # print the array
#
# from matplotlib import pyplot as plt
# array = np.concatenate((data_1, data_2, data_3))
#
# # for a in array:
# #     plt.clf()
# #     fig = plt.figure(1)
# #     plt.imshow(a)
# #     plt.draw()
# #     plt.pause(10e-50)
# save('expert_demonstrations/human_expert_CarRacing_images_full.npy', array)