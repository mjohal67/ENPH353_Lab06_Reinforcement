
import cv2
import gym
import math
#from asyncio.timeouts import timeout
import rospy
import roslaunch
import time
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import Image
from time import sleep

from gym.utils import seeding


class Gazebo_Lab06_Env(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        LAUNCH_FILE = '/home/fizzer/enph353_gym-gazebo-noetic/gym_gazebo/envs/ros_ws/src/enph353_lab06/launch/lab06_world.launch'
        gazebo_env.GazeboEnv.__init__(self, LAUNCH_FILE)
        
        # Setup pub/sub for state/action
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        # Gazebo specific services to start/stop its behavior and
        # facilitate the overall RL environment
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)

        # Setup the environment
        self.action_space = spaces.Discrete(3)  # 0=F,1=L,2=R
        self.reward_range = (-np.inf, np.inf)
        self.episode_history = []

        self._seed() 

        #State
        self.data = None
        self.max_timeout = 30 #number of consecutive times we let the robot not see a line before we reset
        self.Q = {} #Q matrix

        #Image processing
        self.bridge = CvBridge()
        self.timeout = 0  # Used to keep track of images with no line detected
        self.lower_blue = np.array([97,  0,   0])
        self.upper_blue = np.array([150, 255, 255])
        self.road_threshold = 120 #used for binary threshold
        self.crop_bound = 250 #used to crop out everything above cv_frame.shape[0] - CROP_BOUND (above refers to smaller y values)
        #note: in robot.xacro, defined camera's output image dimensions to be 800x800 (constant)
        
    def process_image(self, data):
        '''
            @brief Coverts data into a opencv image and displays it
            @param data : Image data from ROS

            @retval (state, done)
        '''
        print("ENTERED PROCESS IMAGE FUNCTION")
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        (x_width, y_height, channels) = cv_image.shape #y top is 0, x left is 0
        frame_cropped = cv_image[-self.crop_bound:-1, :]
        frame_gray = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2GRAY)
        _, frame_binary_thresh_inv = cv2.threshold(frame_gray, self.road_threshold, 255, cv2.THRESH_BINARY_INV) #road white (255), rest black (0)
         
        # https://stackoverflow.com/questions/54388832/calculating-center-of-an-object-in-an-image
        # moment = distribution of matter about a point/axis
        # for an image, it's sum of (x,y) of entire image * intensity
        # because we're looking at binary image, needed to invert so road is 1 and everything else is 0
        # https://www.youtube.com/watch?v=AAbUfZD_09s
        # to get centroid (x_bar, y_bar), do x_bar=M10/M00, y_bar=M01/M00
        # M00 is 0th order moment in x and y, area of non-zero pixels (road)

        moment = cv2.moments(frame_binary_thresh_inv)
        x_centroid = moment["m10"] / (moment["m00"])

        state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        done = False
        
        # The state array is a list of 10 elements indicating where in the
        # image the line is:
        # i.e.
        #    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] indicates line is on the left
        #    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] indicates line is in the center

        #take width of frame, divide by 10, start from zero and step by width/10 (0-->80, 80-->160, 160-->240, ...)
        step = x_width/10 #800/10 = 80
        state_index = 0
        for i in range(0, x_width, step):
            if (x_centroid > i) or (x_centroid < i+step): #if centroid between two thresholds
                state[state_index] = 1
                break #found location of line, stop looping
            state_index+=1

        if(1 not in state): #no line detected
            self.timeout+=1
        else:
            self.timeout = 0 #saw line, reset timeout

        if(self.timeout == self.max_timeout):
            done = True

        #output feed (cropped, grayscale image with circle for centroid)
        frame_out = frame_gray
        frame_out = cv2.circle(frame_binary_thresh_inv, (int(x_centroid), 150), 10, (0, 0, 255), -1)
        frame_out = cv2.putText(frame_out, "["+(",".join([str(i) for i in state]))+"]", (10, 10), cv2.FONT_HERSHEY_SIMPLEX,color=(255, 255, 255))
        cv2.imshow("Feed", frame_out)

        return state, done

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action): #call after you've chosen an action

        #unpause physics
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        #add action we're about to take to the history of actions
        self.episode_history.append(action)

        vel_cmd = Twist()

        if action == 0:  # FORWARD
            vel_cmd.linear.x = 0.4
            vel_cmd.angular.z = 0.0
        elif action == 1:  # LEFT
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = 0.5
        elif action == 2:  # RIGHT
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = -0.5

        self.vel_pub.publish(vel_cmd) #take action (move robot)

        #wait for image data from camera
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/rrbot/camera1/image_raw', Image, timeout=5)
            except:
                pass

        #got message from camera, pause physics
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause() #pause physics
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state, done = self.process_image(data) #process image for state we've transitioned into

        # Set the rewards for our action
        if not done:
            if action == 0:  # FORWARD
                reward = 4
            elif action == 1:  # LEFT
                reward = 2
            else:
                reward = 2  # RIGHT
        else:
            reward = -200

        return state, reward, done, {}

    def reset(self):

        print("Episode history: {}".format(self.episode_history))
        self.episode_history = []
        print("Resetting simulation...")
        # Resets the state of the environment and returns an initial
        # observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            # reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            # resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        # read image data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/rrbot/camera1/image_raw', Image, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        self.timeout = 0
        state, done = self.process_image(data)

        return state
