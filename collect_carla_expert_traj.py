import cv2
from numpy import save
from CAPORL.environments import carlaenv_collect_img, carlaenv_cont_no_decoder, carlaenv_continuous
from src.IRL.utils.callbacks import Callbacks, load_expert_memories
import pygame

class game_loop():
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Car tutorial")
        width = 1280
        height = 720
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
        env = carlaenv_continuous.env()
        obs = env.reset()


        n_experiences = 0

        done = False
        recording = False
        # Prints the values for axis0
        while not self.exit:

            if done:
                obs = env.reset()

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
            if n_experiences % 100 == 0:
                print('Numero experiencias: ', n_experiences)

        print('fin')
        list = env.save_img_list
        for img in list:
            cv2.imshow('rgb', cv2.cvtColor(img[0], cv2.COLOR_RGB2BGR) )
            cv2.imshow('segmentation', cv2.cvtColor(img[1], cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        cb.memory_to_csv('expert_demonstrations/', 'human_expert_carla_wheel_street')

    # save('rgb_seg.npy', list)
def main_2():
    game = game_loop()
    game.run()


main_2()

