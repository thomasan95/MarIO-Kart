from skimage.transform import resize
import config
import math
from inputs import get_gamepad
import threading
import numpy as np
from skimage.io import imread
import sys
conf = config.Config()


def get_batches(x_train, y_train, batch_size):
    """
    Grab batches for supervised training
    :param x_train: x training data
    :type x_train: numpy.ndarray
    :param y_train: labels for training data
    :type y_train: numpy.ndarray:
    :param batch_size:
    :returns: batch_x, batch_y
    """
    for batch_i in range(len(x_train) // batch_size):
        start = batch_i * batch_size
        end = start + batch_size
        batch_x = x_train[start:end]
        labels = y_train[start:end]
        yield batch_x, labels


def get_4d_batches(x_train, y_train, batch_size):
    for batch_i in range(len(x_train) // batch_size):
        start = batch_i * batch_size
        end = start + batch_size
        batch_x = np.zeros((batch_size, conf.img_h, conf.img_w, conf.img_d*conf.num_frames))
        counter = 0
        batch_y = np.zeros((batch_size, conf.OUTPUT_SIZE))
        for d_i in range(start, end-3):
            one = x_train[d_i]
            two = x_train[d_i+1]
            three = x_train[d_i+2]
            four = x_train[d_i+3]
            batch_x[counter] = np.dstack((one, two, three, four))
            batch_y[counter] = y_train[d_i+3]
            counter += 1
        yield batch_x, batch_y


def resize_img(img):
    """
    Image to pass in
    :param img: numpy array image
    :return: resized image
    """
    im = resize(img, (conf.img_h, conf.img_w, conf.img_d))
    im_arr = im.reshape((conf.img_h, conf.img_w, conf.img_d))
    return im_arr


class XboxController(object):
    MAX_TRIG_VAL = math.pow(2, 8)
    MAX_JOY_VAL = math.pow(2, 15)

    def __init__(self):

        self.LeftJoystickY = 0
        self.LeftJoystickX = 0
        self.RightJoystickY = 0
        self.RightJoystickX = 0
        self.LeftTrigger = 0
        self.RightTrigger = 0
        self.LeftBumper = 0
        self.RightBumper = 0
        self.A = 0
        self.X = 0
        self.Y = 0
        self.B = 0
        self.LeftThumb = 0
        self.RightThumb = 0
        self.Back = 0
        self.Start = 0
        self.LeftDPad = 0
        self.RightDPad = 0
        self.UpDPad = 0
        self.DownDPad = 0

        self._monitor_thread = threading.Thread(target=self._monitor_controller, args=())
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

    def read(self):
        x = self.LeftJoystickX
        y = self.LeftJoystickY
        a = self.A
        b = self.X # b=1, x=2
        rb = self.RightBumper
        return [x, y, a, b, rb]

    def _monitor_controller(self):
        while True:
            events = get_gamepad()
            for event in events:
                if event.code == 'ABS_Y':
                    self.LeftJoystickY = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_X':
                    self.LeftJoystickX = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_RY':
                    self.RightJoystickY = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_RX':
                    self.RightJoystickX = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_Z':
                    self.LeftTrigger = event.state / XboxController.MAX_TRIG_VAL # normalize between 0 and 1
                elif event.code == 'ABS_RZ':
                    self.RightTrigger = event.state / XboxController.MAX_TRIG_VAL # normalize between 0 and 1
                elif event.code == 'BTN_TL':
                    self.LeftBumper = event.state
                elif event.code == 'BTN_TR':
                    self.RightBumper = event.state
                elif event.code == 'BTN_SOUTH':
                    self.A = event.state
                elif event.code == 'BTN_NORTH':
                    self.X = event.state
                elif event.code == 'BTN_WEST':
                    self.Y = event.state
                elif event.code == 'BTN_EAST':
                    self.B = event.state
                elif event.code == 'BTN_THUMBL':
                    self.LeftThumb = event.state
                elif event.code == 'BTN_THUMBR':
                    self.RightThumb = event.state
                elif event.code == 'BTN_SELECT':
                    self.Back = event.state
                elif event.code == 'BTN_START':
                    self.Start = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY1':
                    self.LeftDPad = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY2':
                    self.RightDPad = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY3':
                    self.UpDPad = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY4':
                    self.DownDPad = event.state


def load_sample(sample):
    """
    Loads a specific sample from csv
    :param sample: folder where csv files are found
    :return: loaded images and loaded joystick outputs
    :rtype: numpy.ndarray, numpy.ndarray
    """
    image_files = np.loadtxt(sample + '/data.csv', delimiter=',', dtype=str, usecols=(0,))
    joystick_values = np.loadtxt(sample + '/data.csv', delimiter=',', usecols=(1, 2, 3, 4, 5))
    return image_files, joystick_values


def prepare(samples):
    """
    Processes data into .npy files so the network can train on them

    :param samples: all folders to be read from
    :type samples: list
    :return: None
    """
    print("Preparing data")
    x = []
    y = []
    for sample in samples:
        print(sample)
        # load sample
        image_files, joystick_values = load_sample(sample)
        # add joystick values to y
        y.append(joystick_values)
        # load, prepare and add images to X
        for image_file in image_files:
            image = imread(image_file)
            vec = resize_img(image)
            x.append(vec)
    print("Saving to file...")
    x = np.asarray(x)
    y = np.concatenate(y)
    np.save("data/X", x)
    np.save("data/y", y)
    print("Done!")
    return


if __name__ == "__main__":
    if sys.argv[1] == 'prepare':
        prepare(sys.argv[2:])
