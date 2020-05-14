import glob
import math
import os
import random
import sys
import time
import cv2
import numpy as np
import weakref
import collections

import matplotlib.pyplot as plt
import CAPORL.environments.carla.vae_1 as vae

try:
    sys.path.append(glob.glob("/home/shernandez/PycharmProjects/AIRL/CAPORL/environments/carla/dist/carla-0.9.7-py2.7-linux-x86_64.egg")[0])
except IndexError:
    pass

import carla
from carla import ColorConverter as cc

class action_space:
    def __init__(self):
        """
        Actions: Permutaciones posibles
        """
        self.action_space = [[0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 0, 1]]

        self.n = len(self.action_space)
        self.action_space = np.array(self.action_space)

SECONDS_PER_EPISODE = 40

class env:


    def __init__(self):
        self.im_width = 1280  # 640
        self.im_height = 720  # 480
        self.steer_amt = 0.1
        self.actor_list = []
        self.collision_hist = []
        self.lane_invasion_hist = []
        self.imu_data_hist = []
        self.action_space = action_space()
        self.observation_space = np.zeros((129,))

        self.render_image = None
        self.front_camera = None
        self.latent_data = None
        self.im_to_save = None
        self._list_low_speed = []
        self.encoder, self.decoder, _vae = vae.load_model('/home/shernandez/CARLA_0.9.7/PythonAPI/CarlaTutorial/encoder_64_64_128_256_l128',
                        '/home/shernandez/CARLA_0.9.7/PythonAPI/CarlaTutorial/decoder_64_64_128_256_l128')
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(2.0)

        # Once we have a client we can retrieve the world that is currently
        # running.
        self.client.load_world('Town04')
        self.world = self.client.get_world()
        weather = carla.WeatherParameters.ClearNoon
        self.world.set_weather(weather)

        # The world contains the list blueprints that we can use for adding new
        # actors into the simulation.
        blueprint_library = self.world.get_blueprint_library()

        # Now let's filter all the blueprints of type 'vehicle' and choose one
        # at random.
        vehicle_list = blueprint_library.filter('vehicle')
        # self.model_3 = random.choice(vehicle_list)
        self.model_3 = blueprint_library.filter('model3')[0]
        # self.model_3.set_attribute("sticky_control", "False")
        self.stating_points = np.loadtxt('/home/shernandez/CARLA_0.9.7/PythonAPI/CarlaTutorial/start_point.txt', )
        self.img_counter = 0
        self.timer_for_recording = time.time()

        self.test_flag = False
        self._last_action = collections.deque(maxlen=30)
        self.first_step = True

    def reset(self):
        self.collision_hist = []
        self.lane_invasion_hist = []
        self.imu_data_hist = []
        self._list_low_speed = []

        for actor in self.actor_list:
            actor.destroy()
        self.actor_list = []

        self.transform = random.choice(self.world.get_map().get_spawn_points())
        index = random.choice(range(self.stating_points.shape[0]))

        # self.transform.location.x = np.random.normal(self.stating_points[index][0], 2)  # -20.6
        # self.transform.location.y = np.random.normal(self.stating_points[index][1], 2)  # -259.5
        self.transform.location.x = self.stating_points[index][0]  #-20.6
        self.transform.location.y = self.stating_points[index][1]  #-259.5
        self.transform.rotation.yaw = self.stating_points[index][2]  #120.
        # self.transform = self.world.get_map().get_spawn_points()[1]
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)

        self.rgb_cam = self.world.get_blueprint_library().find('sensor.camera.rgb')

        self.rgb_cam.set_attribute('image_size_x', str(self.im_width))
        self.rgb_cam.set_attribute('image_size_y', str(self.im_height))
        # self.rgb_cam.set_attribute('fov', '110')

        # transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        transform = carla.Transform(carla.Location(x=1.6, z=1.7))
        self.sensor_rgb = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)

        self.actor_list.append(self.sensor_rgb)
        self.sensor_rgb.listen(lambda data: self.process_img(data, 'rgb'))


        self.rgb_render_cam = self.world.get_blueprint_library().find('sensor.camera.rgb')

        self.rgb_render_cam.set_attribute('image_size_x', str(self.im_width))
        self.rgb_render_cam.set_attribute('image_size_y', str(self.im_height))
        Attachment = carla.AttachmentType
        transform_rgb_render_cam = carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0))
        self.sensor_rgb_render = self.world.spawn_actor(self.rgb_render_cam, transform_rgb_render_cam,
                                                        attach_to=self.vehicle,
                                                        attachment_type=Attachment.SpringArm)

        self.actor_list.append(self.sensor_rgb_render)
        self.sensor_rgb_render.listen(lambda data: self.process_render(data))

        #################################################################
        #          Depth and semantic segmentation
        #################################################################
        # self.segment_cam = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        # self.segment_cam.set_attribute('image_size_x', str(self.im_width))
        # self.segment_cam.set_attribute('image_size_y', str(self.im_height))
        #
        # self.sensor_segment = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        #
        # self.actor_list.append(self.sensor_segment)
        # self.sensor_segment.listen(lambda data: self.process_img(data, 'segmentation'))
        #
        # self.depth_cam = self.world.get_blueprint_library().find('sensor.camera.depth')
        # self.depth_cam.set_attribute('image_size_x', str(self.im_width))
        # self.depth_cam.set_attribute('image_size_y', str(self.im_height))
        #
        # self.sensor_depth = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        #
        # self.actor_list.append(self.sensor_depth)
        # self.sensor_depth.listen(lambda data: self.process_img(data, 'depth'))


        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        time.sleep(4)  # sleep to get things started and to not detect a collision when the car spawns/falls from sky.

        colsensor = self.world.get_blueprint_library().find('sensor.other.collision')
        self.colsensor = self.world.spawn_actor(colsensor, carla.Transform(), attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda col_event: self.collision_data(col_event))

        # self.colsensor1 = CollisionSensor(self.vehicle)

        self.gnssensor = GnssSensor(self.vehicle)
        self.actor_list.append(self.gnssensor)

        lane_sensor = self.world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.lane_sensor = self.world.spawn_actor(lane_sensor, carla.Transform(), attach_to=self.vehicle)
        self.actor_list.append(self.lane_sensor)
        self.lane_sensor.listen(lambda lane_event: self.lane_invasion_data(lane_event))
        #
        imu_sensor = self.world.get_blueprint_library().find('sensor.other.imu')
        self.imu_sensor = self.world.spawn_actor(imu_sensor, carla.Transform(), attach_to=self.vehicle)
        self.actor_list.append(self.imu_sensor)
        self.imu_sensor.listen(lambda event: self.imu_data(event))


        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()

        self.vehicle.apply_control(carla.VehicleControl(brake=0.0, throttle=0.0))

        self.timer_for_recording = time.time()

        obs = self.extract_latent_data(self.front_camera)
        self._last_action = collections.deque(maxlen=30)
        self.first_step = True

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))
        obs = np.concatenate((obs, [kmh]))
        return obs

    def collision_data(self, event):
        print('collision')
        self.collision_hist.append(event)

    def lane_invasion_data(self, event):
        # print(event)
        self.lane_invasion_hist.append(event)

    def imu_data(self, event):
        # print(event)
        self.imu_data_hist.append(event)

    def process_render(self, image):
        i = np.array(image.raw_data)
        # np.save("iout.npy", i)
        image = i.reshape((self.im_height, self.im_width, 4))
        image = image[:, :, :3]

        self.render_image = image

    def process_img(self, image, sensor):
        if sensor == 'segmentation':
            image.convert(cc.CityScapesPalette)
        elif sensor == 'depth':
            image.convert(cc.LogarithmicDepth)

        # array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        i = np.array(image.raw_data)
        #np.save("iout.npy", i)
        image = i.reshape((self.im_height, self.im_width, 4))
        image = image[:, :, :3]

        # cv2.imshow('ventana', image)
        # cv2.waitKey(1)
        # plt.clf()
        # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # plt.draw()
        # plt.pause(10e-50)
        # plt.show()
        self.im_to_save = image
        image = image[370:, :, :]
        image = cv2.resize(image, (128, 128))

        if sensor == 'segmentation':
            image.convert(cc.CityScapesPalette)
        elif sensor == 'depth':
            image.convert(cc.LogarithmicDepth)
        else:
            self.front_camera = image
            # self.latent_data = self.encoder.predict(np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2])))[0]


    def _act(self, action, kmh):
        # if kmh > 30:
        #     throttle = 0.
        # else:
        #     throttle = 0.7

        if action == 0:
            throttle = 0
            self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=0))
        if action == 1:
            throttle = 0
            self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=-1*self.steer_amt))
        if action == 2:
            throttle = 0
            self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=1*self.steer_amt))
        if action == 3:
            throttle = 1
            self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=0))
        if action == 4:
            throttle = 1
            self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=-1*self.steer_amt))
        if action == 5:
            throttle = 1
            self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=1*self.steer_amt))

        return throttle

    def step(self, action):
        '''
        For now let's just pass steer left, center, right?
        0, 1, 2
        '''

        if self.first_step:  # para lanzar el coche al comienzo del episodio
            # for i in range(10):
            self.vehicle.apply_control(carla.VehicleControl(brake=0.0, throttle=1.0))
            time.sleep(1.5)
            self.first_step = False

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))

        throttle = self._act(action, kmh)

        reward, done = self.reward(kmh)

        if kmh < 1:
            self._list_low_speed.append(kmh)

        if self.episode_start + SECONDS_PER_EPISODE < time.time() or len(self._list_low_speed) > 100:
            done = True

        # if done:
        #     self.vehicle.destroy()

        obs = self.extract_latent_data(self.front_camera)
        obs = np.concatenate((obs, [kmh]))
        return obs, reward, done, None

    def reward(self, kmh):
        # if (len(self.collision_hist) > 0 or len(self.lane_invasion_hist) > 0 or self._check_done()) and not self.test_flag:
        if self._check_done():
            # if len(self.lane_invasion_hist) > 2:
            #
            #     frame_diff = np.abs(self.lane_invasion_hist[-1].frame - self.lane_invasion_hist[0].frame)
            #
            #     if frame_diff < 70:
            #         done = True
            #         reward = -1
            #     else:
            #         done = False
            #         reward = -0.5
            #         self.lane_invasion_hist = [self.lane_invasion_hist[-1]]
            # else:
            #     done = False
            #     reward = -0.25
            # done = True
            reward = -10. - 0.1 * kmh
            done = not self.test_flag

        # elif kmh < 1:
        #     self._list_low_speed.append(kmh)
        #     done = False
        #     reward = -0.1
        else:
            if kmh < 50:
                reward = 1. + 0.05 * kmh
            elif kmh < 2:
                reward = -1.
            else:
                reward = 0.5
            done = False

        # print('reward: ', reward)
        return reward, done

    def _check_done(self):
        inside_bounds = False
        # for point in self.stating_points:
        #     distance = [np.sqrt(np.square(self.gnssensor.x - point[0]) + np.square(self.gnssensor.y - point[1])) for point in self.stating_points]
        #     if distance < 2.5:
        #         inside_bounds = True
        #         break
        distance = [np.sqrt(np.square(self.gnssensor.x - point[0]) + np.square(self.gnssensor.y - point[1])) < 1.5 for point
                    in self.stating_points]

        return not True in distance

    def render(self):
        # cv2.imshow('ventana', self.render_image)
        # cv2.waitKey(1)
        # plt.clf()
        # plt.imshow(cv2.cvtColor(self.front_camera, cv2.COLOR_BGR2RGB))
        # plt.draw()
        # plt.pause(10e-50)
        # plt.show()
        current_time = time.time()
        # print(current_time - self.timer_for_recording, self.timer_for_recording, current_time)
        save = False
        if save and current_time - self.timer_for_recording > 1.:
            cv2.imwrite('/home/shernandez/CARLA_0.9.7/PythonAPI/CarlaTutorial/_out/img_' + str(self.img_counter)
                        + '.png', self.im_to_save)

            self.img_counter += 1
        # image = cv2.cvtColor(self.front_camera, cv2.COLOR_BGR2RGB)
        # cv2.imshow('VAE input', self.front_camera)  #cv2.cvtColor(self.front_camera, cv2.COLOR_BGR2RGB)
        # cv2.waitKey(1)

        train_img = cv2.cvtColor(self.front_camera, cv2.COLOR_BGR2RGB)
        train_img = np.reshape(train_img/255, (1, self.front_camera.shape[0],
                                 self.front_camera.shape[1], self.front_camera.shape[2]))

        # z_mean, z_std, z = self.encoder.predict(train_img)
        z_mean = self.extract_latent_data(self.front_camera)
        z_mean = np.reshape(z_mean, (-1, len(z_mean)))
        images = self.decoder.predict(z_mean)[0]
        self.latent_data = z_mean[0]
        images = np.uint8(images * 255)
        images = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)
        # cv2.imshow('VAE output', images)
        # cv2.waitKey(1)

        render_img = self.render_image
        render_img[10:10 + self.front_camera.shape[0], 10:10 + self.front_camera.shape[0], :] = self.front_camera
        render_img[10:10 + images.shape[0], 20 + self.front_camera.shape[0]:20 + self.front_camera.shape[0] + images.shape[0], :] = images

        for i in range(len(self._last_action)):
            render_img = cv2.putText(render_img, 'Action: ' + str(self._last_action[i]), (1000, 20 + i * 20),
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        cv2.imshow('ventana', render_img)
        cv2.waitKey(1)
        pass

    def close(self):
        pass

    def extract_latent_data(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.reshape(img / 255, (1, img.shape[0], img.shape[1], img.shape[2]))
        z_mean, z_std, z = self.encoder.predict(img)
        self.latent_data = z_mean[0]
        return self.latent_data

class CollisionSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        # self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

        def get_collision_history(self):
            history = collections.defaultdict(int)
            for frame, intensity in self.history:
                history[frame] += intensity
            return history

        @staticmethod
        def _on_collision(weak_self, event):
            self = weak_self()
            if not self:
                return
            print('collision')
            self.history.append(event)


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        self.x = 0.0
        self.y = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))
        self.position_list = []

    def destroy(self):
        self.sensor.destroy()

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude
        self.x = event.transform.location.x
        self.y = event.transform.location.y
        # self.position_list.append([event.transform.location.x, event.transform.location.y, event.transform.rotation.yaw])
