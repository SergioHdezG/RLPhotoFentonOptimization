from CAPORL.environments.env_interfaz import EnvInterface, ActionSpaceInterface
import gym
import CAPORL.environments.carla.vae_1 as vae
import os
import numpy as np
import cv2
from CAPORL.utils.custom_networks import custom_nets

class action_space:
    def __init__(self):
        """
        Actions: Permutaciones posibles
        """
        self.low = -1
        self.high = 1
        self.n = 3

class env(EnvInterface):
    """ Problema del viajante
    """
    def __init__(self):
        super().__init__()
        self.gym_env = gym.make("CarRacing-v0")

        self.iterations = 0

        self.action_space = action_space()  # self.gym_env.action_space
        self.observation_space = np.zeros((256,))

        # self.encoder, self.decoder, _vae = vae.load_model(os.path.abspath('saved_models/Carracing_vae/vae_encoder_32_64_128_128'),
        #                 os.path.abspath('saved_models/Carracing_vae/vae_decoder_32_64_128_128'), self.gym_env.observation_space.shape)
        # self.decoder = None
        self.encoder = custom_nets.encoder
        self.decoder = custom_nets.decoder

        self.cum_reward = 0
        self.max_it_epi = 500
        self.episodes_count = 0

    def reset(self):
        self.episodes_count += 1
        obs = self.gym_env.reset()
        for i in range(45):
            obs, reward, done, info = self.gym_env.step([0, 0.0, 0.0])
        obs = np.reshape(obs / 255, (1, obs.shape[0], obs.shape[1], obs.shape[2]))
        z_mean, z_std, z = self.encoder.predict(obs)
        obs = z_mean[0]  # ¿o z[0]?
        self.iterations = 0
        self.cum_reward = 0
        return obs

    def _act(self, action):
        act = np.array([a for a in action])
        # action[1] = (action[1]*1.2 + 0.8)/2
        # action[2] = (action[2]*1.2 + 0.8)/2
        act[1] = np.clip((action[1]+1)/2, 0., 1.)

        # if self.episodes_count < 50:
        #     act[2] = 0
        # else:
        #     act[2] = np.clip(action[2], 0., np.clip(float(self.episodes_count)/200., 0, 1))
        # act[2] = 0
        act[2] = np.clip(action[2], 0., 1.0)

        return self.gym_env.step(act)

    def step(self, action, mirror=False):
        obs, reward, done, info = self._act(action)

        obs = cv2.resize(obs[:-12, :], (96, 96))

        img = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (400, 400))

        ctr_img = self._controls_img(action[0], (action[1]+1)/2, action[2])
        img = cv2.putText(img, "Steer:      {:.4f} ".format(action[0]), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        img = cv2.putText(img, "Throttle:   {:.4f}".format(action[1]), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        img = cv2.putText(img, "brake:      {:.4f}".format(action[2]), (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        img = cv2.putText(img, 'Reward:     {:.4f}'.format(reward), (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        img[75:75+ctr_img.shape[0], 5:5+ctr_img.shape[1], :] = ctr_img
        cv2.imshow("obs", img)

        if mirror:
            flip = np.fliplr(obs)
            img_flip = cv2.cvtColor(flip, cv2.COLOR_BGR2RGB)
            cv2.imshow("flip", img_flip)

        if np.max(obs) > 1:
            normalize = 255
        else:
            normalize = 1

        obs = np.reshape(obs / normalize, (1, obs.shape[0], obs.shape[1], obs.shape[2]))
        z_mean, z_std, z = self.encoder.predict(obs)
        obs = z_mean[0]  # ¿o z[0]?

        if mirror:
            flip = np.reshape(flip / normalize, (1, flip.shape[0], flip.shape[1], flip.shape[2]))
            z_mean, z_std, z = self.encoder.predict(flip)
            flip = z_mean[0]  # ¿o z[0]?

        img = self.decoder.predict(np.reshape(obs, (1, -1)))[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (400, 400))
        cv2.imshow("decoder", img)

        if mirror:
            img_flip = self.decoder.predict(np.reshape(flip, (1, -1)))[0]
            img_flip = cv2.cvtColor(img_flip, cv2.COLOR_BGR2RGB)
            cv2.imshow("flip_decoder", img_flip)
        cv2.waitKey(1)

        self.cum_reward += reward
        self.iterations += 1


        if self.iterations > self.max_it_epi:
            done = True

            if self.cum_reward * 2 > self.max_it_epi:
                if self.cum_reward > self.max_it_epi:
                    self.max_it_epi = self.cum_reward
                else:
                    self.max_it_epi += 10
            elif self.cum_reward < self.max_it_epi and self.max_it_epi > 500:
                self.max_it_epi -= 10

        if mirror:
            return obs, flip, reward, done, info
        return obs, reward, done, info

    def _controls_img(self, steer, throttle, brake):
        ste_img = np.ones((5, 41, 3), dtype=np.uint8) * 255
        thr_img = np.ones((5, 41, 3), dtype=np.uint8) * 255
        brk_img = np.ones((5, 41, 3), dtype=np.uint8) * 255
        ctr_img = np.zeros((28, 47, 3), dtype=np.uint8)

        steer = np.int((steer+1)/2 * 41)
        throttle = np.int(np.clip(throttle, 0.0, 1.0) * 41)
        brake = np.int(np.clip(brake, 0.0, 1.0) * 41)

        ste_img[:, steer:steer + 1, 1:3] = np.zeros((5, 1, 2), dtype=np.uint8)
        thr_img[:, :throttle, 1] = thr_img[:, :throttle, 1] * 0
        brk_img[:, :brake, 2] = brk_img[:, :brake, 2] * 0

        ctr_img[3:8, 3:44, :] = ste_img
        ctr_img[12:17, 3:44, :] = thr_img
        ctr_img[20:25, 3:44, :] = brk_img

        ctr_img = cv2.resize(ctr_img, (100, 50))

        return ctr_img



    def render(self):
        self.gym_env.render()


    def close(self):
        self.gym_env.close()