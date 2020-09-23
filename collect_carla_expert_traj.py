from environments import carlaenv_continuous_stop
from IRL_Problem.base.utils.callbacks import Callbacks
import pygame
from collections import deque
import numpy as np


class game_loop():
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Car tutorial")
        width = 512 #1280
        height = 288 #720
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.ticks = 60
        self.exit = False

    def run(self):
        pygame.display.init()
        pygame.joystick.init()
        pygame.joystick.Joystick(0).init()

        #
        # # Prints the joystick's name
        # JoyName = pygame.joystick.Joystick(0).get_name()
        # print
        # "Name of the joystick:"
        # print
        # JoyName
        # # Gets the number of axes
        # JoyAx = pygame.joystick.Joystick(0).get_numaxes()
        # print
        # "Number of axis:"
        # print
        # JoyAx


        cb = Callbacks()
        env = carlaenv_continuous_stop.env()
        obs = env.reset()

        recording = False
        for e in range(100):

            done = False
            episodic_reward = 0
            n_experiences = 0
            rew_mean_list = deque(maxlen=10)
            # Prints the values for axis0
            obs = env.reset()

            while not self.exit and not done:

                # # Event queue
                # for event in pygame.event.get():
                #     if event.type == pygame.QUIT:
                #         pass
                        # self.exit = True
                # pygame.event.pump()

                # User input
                pressed = pygame.key.get_pressed()
                if pressed[pygame.K_ESCAPE]:
                    self.exit = True
                if pressed[pygame.K_SPACE]:
                    obs = env.reset()

                # Pause
                if pressed[pygame.K_F1]:
                    print('pause')
                    pause = True
                    while pause:
                        pygame.event.pump()
                        pressed = pygame.key.get_pressed()
                        if pressed[pygame.K_SPACE]:
                            obs = env.reset()
                            pause = False

                if pressed[pygame.K_F2]:
                    recording = not recording
                    print('recording = ', recording)


                # x, y = pygame.mouse.get_pos()
                # steer = ((x/1280.)-0.5)/2  # pygame.joystick.Joystick(0).get_axis(0)
                # throttle = ((720 - y)/720 * 2) - 1 # -pygame.joystick.Joystick(0).get_axis(2)

                pygame.event.pump()
                steer = pygame.joystick.Joystick(0).get_axis(0) #/3
                throttle = -pygame.joystick.Joystick(0).get_axis(2) #(-pygame.joystick.Joystick(0).get_axis(1)*1.35)*2 -1
                # if throttle < -0.9:
                #     throttle = -1.
                brake = -pygame.joystick.Joystick(0).get_axis(3)
                # if brake < 0.1:
                #     brake = -1.

                # print('steer: ', steer, 'throttle: ', throttle, 'brake: ', brake)  #, ' mice: ', x, y)
                image = env.render(only_return=True)
                next_obs, reward, done, _ = env.step([throttle, steer, brake])

                if recording:
                    cb.remember_callback(obs, next_obs, [throttle, steer, brake], reward, done)

                obs = next_obs

                # Drawing
                image = pygame.surfarray.make_surface(image)
                # self.screen.fill((0, 0, 0))
                self.screen.blit(image, (0, 0))
                # pygame.display.flip()
                pygame.display.update()
                self.clock.tick(self.ticks)
                n_experiences += 1
                episodic_reward += reward
                # if n_experiences % 100 == 0:
                #     print('Numero experiencias: ', n_experiences)

            rew_mean_list.append(episodic_reward)
            _feedback_print(e, episodic_reward, n_experiences, 1, rew_mean_list)
            if self.exit:
                break
        # list = env.save_img_list
        # for img in list:
        #     cv2.imshow('rgb', cv2.cvtColor(img[0], cv2.COLOR_RGB2BGR) )
        #     cv2.imshow('segmentation', cv2.cvtColor(img[1], cv2.COLOR_RGB2BGR))
        #     cv2.waitKey(1)
        # cb.memory_to_csv('expert_demonstrations/ultimos/', 'human_expert_carla_road_steer')
        print('fin')
    # save('rgb_seg.npy', list)

def _feedback_print(e, episodic_reward, epochs, verbose, epi_rew_list):
    rew_mean = np.sum(epi_rew_list) / len(epi_rew_list)

    if verbose == 1:
        if (e + 1) % 1 == 0:
            print('Episode ', e + 1, 'Epochs ', epochs, ' Reward {:.1f}'.format(episodic_reward),
                  'Smooth Reward {:.1f}'.format(rew_mean), ' Epsilon {:.4f}'.format(1.00))

def main_2():
    game = game_loop()
    game.run()


main_2()

