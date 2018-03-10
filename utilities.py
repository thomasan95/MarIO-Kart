from skimage.transform import resize
import config
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


def resize_img(img):
    """
    Image to pass in
    :param img: numpy array image
    :return: resized image
    """
    im = resize(img, (conf.img_h, conf.img_w, conf.img_d))
    im_arr = im.reshape((conf.img_h, conf.img_w, conf.img_d))
    return im_arr
