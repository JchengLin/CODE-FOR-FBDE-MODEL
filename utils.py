import tensorflow as tf
from tensorflow.contrib import slim
import cv2
import os, random
import numpy as np
import scipy.stats as st
from scipy.ndimage import filters
from skimage import segmentation, color
from joblib import Parallel, delayed
from PIL import Image
import sys, math

class ImageData:

    def __init__(self, load_size, channels, augment_flag):
        self.load_size = load_size
        self.channels = channels
        self.augment_flag = augment_flag

    def image_processing(self, filename):
        x = tf.read_file(filename)
        x_decode = tf.image.decode_jpeg(x, channels=self.channels)
        img = tf.image.resize_images(x_decode, [self.load_size, self.load_size])
        img = tf.cast(img, tf.float32) / 127.5 - 1

        if self.augment_flag :
            augment_size = self.load_size + (30 if self.load_size == 256 else 15)
            p = random.random()
            if p > 0.5:
                img = data_augmentation(img, augment_size)

        return img

def data_augmentation(image, augment_size):
    seed = random.randint(0, 2 ** 31 - 1)
    ori_image_shape = tf.shape(image)
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.resize_images(image, [augment_size, augment_size])
    image = tf.random_crop(image, ori_image_shape, seed=seed)
    
    return image
      
def load_test_data(image_path, size=256):

    img = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dsize=(size, size))
    img = np.expand_dims(img, axis=0)
    img = img/127.5 - 1

    return img

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def inverse_transform(images):
    return ((images+1.) / 2) * 255.0

def imsave(images, size, path):
    images = merge(images, size)
    images = cv2.cvtColor(images.astype('uint8'), cv2.COLOR_RGB2BGR)

    return cv2.imwrite(path, images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h*j:h*(j+1), w*i:w*(i+1), :] = image

    return img

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def str2bool(x):

    return x.lower() in ('true')
    
##########################################
# Image Augmentation.
##########################################

def image_augmentation(image):

    _realA = random_brightness(image)
    _realA = random_contrast(_realA)
    _realA = random_color_transform(_realA)
    _realA = additive_shade(_realA)
    _realA = guided_filter(_realA, image, 5, eps=2e-1)
    _realA = random_distortion(_realA)
    
    return _realA

def additive_gaussian_noise(image, stddev_range=[5, 95]):
    stddev = tf.random_uniform((), *stddev_range)
    p = random.random()
    noise = p * tf.random_normal(tf.shape(image), stddev=stddev)
    return image + noise

def random_brightness(image, max_abs_change=50):
    return tf.image.random_brightness(image, max_abs_change)

def random_contrast(image, strength_range=[0.5, 1.5]):
    return tf.image.random_contrast(image, *strength_range)

def random_color_transform(image, color_matrix=None):
    # color_matrix is 3x3
    if color_matrix is None:
        color_matrix = tf.random_uniform((3,3), 0, 1.0, dtype=tf.float32)
        color_matrix_norm = tf.reduce_sum(color_matrix, axis=0, keepdims=True)
        color_matrix = color_matrix / (color_matrix_norm + 1e-6)
    elif isinstance(color_matrix, np.ndarray):
        color_matrix = tf.convert_to_tensor(color_matrix, dtype=tf.float32)
    im_shp = tf.shape(image)
    C = im_shp[-1]
    image = tf.reshape(image, [-1, C])
    image = tf.matmul(image, color_matrix)
    image = tf.reshape(image, im_shp)
    return image

def additive_shade(image, nb_ellipses=20, transparency_range=[-0.5, 0.8],
                   kernel_size_range=[250, 350]):

    def _py_additive_shade(img):
        min_dim = min(img.shape[:2]) / 4
        mask = np.zeros(img.shape[:2], np.uint8)
        for i in range(nb_ellipses):
            ax = int(max(np.random.rand() * min_dim, min_dim / 5))
            ay = int(max(np.random.rand() * min_dim, min_dim / 5))
            max_rad = max(ax, ay)
            x = np.random.randint(max_rad, img.shape[1] - max_rad)  # center
            y = np.random.randint(max_rad, img.shape[0] - max_rad)
            angle = np.random.rand() * 90
            cv2.ellipse(mask, (x, y), (ax, ay), angle, 0, 360, 255, -1)

        transparency = np.random.uniform(*transparency_range)
        kernel_size = np.random.randint(*kernel_size_range)
        if (kernel_size % 2) == 0:  # kernel_size has to be odd
            kernel_size += 1
        mask = cv2.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), 0)
        shaded = img * (1 - transparency * mask[..., np.newaxis]/255.)
        return shaded

    shaded = tf.py_func(_py_additive_shade, [image], tf.float32)
    res = tf.reshape(shaded, tf.shape(image))
    return res

def tf_box_filter(x, r):
    
    ch = x.get_shape().as_list()[-1]
    weight = 1/((2*r+1)**2)
    box_kernel = weight*np.ones((2*r+1, 2*r+1, ch, 1))
    box_kernel = np.array(box_kernel).astype(np.float32)
    output = tf.nn.depthwise_conv2d(x, box_kernel, [1, 1, 1, 1], 'SAME')

    return output
    
def guided_filter(x, y, r, eps=1e-2):

    x_shape = tf.shape(x)
    N = tf_box_filter(tf.ones((1, x_shape[1], x_shape[2], 1), dtype=x.dtype), r)

    mean_x = tf_box_filter(x, r) / N
    mean_y = tf_box_filter(y, r) / N
    cov_xy = tf_box_filter(x * y, r) / N - mean_x * mean_y
    var_x  = tf_box_filter(x * x, r) / N - mean_x * mean_x

    A = cov_xy / (var_x + eps)
    b = mean_y - A * mean_x

    mean_A = tf_box_filter(A, r) / N
    mean_b = tf_box_filter(b, r) / N

    output = mean_A * x + mean_b

    return output

def random_distortion(images, num_anchors=10, perturb_sigma=5.0, disable_border=True):
    # Similar results to elastic deformation (a bit complex transformation)
    # However, the transformation is much faster that elastic deformation and have a straightforward arguments
    # TODO: Need to adapt reflect padding and eliminate out-of-frame
    # images is 4D tensor [B,H,W,C]
    # num_anchors : the number of base position to make distortion, total anchors in a image = num_anchors**2
    # perturb_sigma : the displacement sigma of each anchor

    src_shp_list = images.get_shape().as_list()
    batch_size, src_height, src_width = tf.unstack(tf.shape(images))[:3]

    if disable_border:
        pad_size = tf.to_int32(tf.to_float(tf.maximum(src_height, src_width)) *  (np.sqrt(2)-1.0) / 2 + 0.5)
        images = tf.pad(images, [[0,0], [pad_size]*2, [pad_size]*2, [0,0]], 'REFLECT')
    height, width = tf.unstack(tf.shape(images))[1:3]

    mapx_base = tf.matmul(tf.ones(shape=tf.stack([num_anchors, 1])),
                    tf.transpose(tf.expand_dims(tf.linspace(0., tf.to_float(width), num_anchors), 1), [1, 0]))
    mapy_base = tf.matmul(tf.expand_dims(tf.linspace(0., tf.to_float(height), num_anchors), 1),
                    tf.ones(shape=tf.stack([1, num_anchors])))

    mapx_base = tf.tile(mapx_base[None,...,None], [batch_size,1,1,1]) # [batch_size, N, N, 1]
    mapy_base = tf.tile(mapy_base[None,...,None], [batch_size,1,1,1])
    distortion_x = tf.random_normal((batch_size,num_anchors,num_anchors,1), stddev=perturb_sigma)
    distortion_y = tf.random_normal((batch_size,num_anchors,num_anchors,1), stddev=perturb_sigma)
    mapx = mapx_base + distortion_x
    mapy = mapy_base + distortion_y
    mapx_inv = mapx_base - distortion_x
    mapy_inv = mapy_base - distortion_y

    interp_mapx_base = tf.image.resize_images(mapx_base, size=(height, width), method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
    interp_mapy_base = tf.image.resize_images(mapy_base, size=(height, width), method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
    coord_maps_base = tf.concat([interp_mapx_base, interp_mapy_base], axis=-1)

    interp_mapx = tf.image.resize_images(mapx, size=(height, width), method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
    interp_mapy = tf.image.resize_images(mapy, size=(height, width), method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
    coord_maps = tf.concat([interp_mapx, interp_mapy], axis=-1) # [batch_size, height, width, 2]

    # interp_mapx_inv = tf.image.resize_images(mapx_inv, size=(height, width), method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
    # interp_mapy_inv = tf.image.resize_images(mapy_inv, size=(height, width), method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
    # coord_maps_inv = tf.concat([interp_mapx_inv, interp_mapy_inv], axis=-1) # [batch_size, height, width, 2]
    coord_maps_inv = coord_maps_base + (coord_maps_base-coord_maps)

    warp_images = bilinear_sampling(images, coord_maps)

    if disable_border:
        warp_images = tf.slice(warp_images, [0, pad_size, pad_size, 0], [-1, src_height, src_width, -1])

    warp_images.set_shape(src_shp_list)
    # shp_list[-1] = 2
    # coord_maps.set_shape(shp_list)
    # coord_maps_inv.set_shape(shp_list)

    return warp_images
    # return warp_images, coord_maps, coord_maps_inv
 
#
# Image processing
# Some codes come from  https://github.com/rpautrat/SuperPoint
# input image is supposed to be 3D tensor [H,W,C] and floating 0~255 values
# 

def get_rank(inputs):
    return len(inputs.get_shape())
    
def bilinear_sampling(photos, coords):
    """Construct a new image by bilinear sampling from the input image.

    Points falling outside the source image boundary have value 0.

    Args:
        photos: source image to be sampled from [batch, height_s, width_s, channels]
        coords: coordinates of source pixels to sample from [batch, height_t,
          width_t, 2]. height_t/width_t correspond to the dimensions of the output
          image (don't need to be the same as height_s/width_s). The two channels
          correspond to x and y coordinates respectively.
    Returns:
        A new sampled image [batch, height_t, width_t, channels]
    """ 
    # photos: [batch_size, height2, width2, C]
    # coords: [batch_size, height1, width1, C]
    def _repeat(x, n_repeats):
        rep = tf.transpose(
            tf.expand_dims(tf.ones(shape=tf.stack([
                n_repeats,
            ])), 1), [1, 0])
        rep = tf.cast(rep, 'float32')
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])

    with tf.name_scope('image_sampling'):
        coords_x, coords_y = tf.split(coords, [1, 1], axis=3)
        inp_size = tf.shape(photos)
        coord_size = tf.shape(coords)

        out_size = tf.stack([coord_size[0], 
                             coord_size[1],
                             coord_size[2],
                             inp_size[3],
                            ]) 

        coords_x = tf.cast(coords_x, 'float32')
        coords_y = tf.cast(coords_y, 'float32')

        x0 = tf.floor(coords_x)
        x1 = x0 + 1
        y0 = tf.floor(coords_y)
        y1 = y0 + 1

        y_max = tf.cast(tf.shape(photos)[1] - 1, 'float32')
        x_max = tf.cast(tf.shape(photos)[2] - 1, 'float32')
        zero = tf.zeros([1], dtype='float32')

        x0_safe = tf.clip_by_value(x0, zero, x_max)
        y0_safe = tf.clip_by_value(y0, zero, y_max)
        x1_safe = tf.clip_by_value(x1, zero, x_max)
        y1_safe = tf.clip_by_value(y1, zero, y_max)

        ## bilinear interp weights, with points outside the grid having weight 0
        # wt_x0 = (x1 - coords_x) * tf.cast(tf.equal(x0, x0_safe), 'float32')
        # wt_x1 = (coords_x - x0) * tf.cast(tf.equal(x1, x1_safe), 'float32')
        # wt_y0 = (y1 - coords_y) * tf.cast(tf.equal(y0, y0_safe), 'float32')
        # wt_y1 = (coords_y - y0) * tf.cast(tf.equal(y1, y1_safe), 'float32')

        wt_x0 = x1_safe - coords_x
        wt_x1 = coords_x - x0_safe
        wt_y0 = y1_safe - coords_y
        wt_y1 = coords_y - y0_safe

        ## indices in the flat image to sample from
        dim2 = tf.cast(inp_size[2], 'float32')
        dim1 = tf.cast(inp_size[2] * inp_size[1], 'float32')
        base = tf.reshape(
            _repeat(
                tf.cast(tf.range(coord_size[0]), 'float32') * dim1,
                coord_size[1] * coord_size[2]),
            [out_size[0], out_size[1], out_size[2], 1])

        base_y0 = base + y0_safe * dim2
        base_y1 = base + y1_safe * dim2
        idx00 = tf.reshape(x0_safe + base_y0, [-1])
        idx01 = x0_safe + base_y1
        idx10 = x1_safe + base_y0
        idx11 = x1_safe + base_y1

        
        ## sample from photos
        photos_flat = tf.reshape(photos, tf.stack([-1, inp_size[3]]))
        photos_flat = tf.cast(photos_flat, 'float32')

        im00 = tf.reshape(tf.gather(photos_flat, tf.cast(idx00, 'int32')), out_size)
        im01 = tf.reshape(tf.gather(photos_flat, tf.cast(idx01, 'int32')), out_size)
        im10 = tf.reshape(tf.gather(photos_flat, tf.cast(idx10, 'int32')), out_size)
        im11 = tf.reshape(tf.gather(photos_flat, tf.cast(idx11, 'int32')), out_size)

        w00 = wt_x0 * wt_y0
        w01 = wt_x0 * wt_y1
        w10 = wt_x1 * wt_y0
        w11 = wt_x1 * wt_y1

        out_photos = tf.add_n([
            w00 * im00, w01 * im01,
            w10 * im10, w11 * im11])
        
        return out_photos