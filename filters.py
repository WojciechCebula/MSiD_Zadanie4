import numpy as np
import cv2
import multiprocessing


def gabor(image_set, name):
    num = 1
    kernels = []
    queue = multiprocessing.Queue(maxsize=0)
    phi = 0
    for theta in range(8):
        theta = theta / 4. * np.pi
        for sigma in (1, 3, 5, 7):
            for lamda in np.arange(np.pi / 4, np.pi, np.pi / 4):
                for gamma in (0.05, 0.5):
                    for ksize in (3, 5):
                        save_name = name + str(num)
                        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
                        kernels.append(kernel)
                        queue.put((apply_kernel, image_set, kernel, save_name))
                        num += 1
    return queue


def apply_kernel(image_set, kernel):
    filtered_images = []
    for image in image_set:
        filtered_images.append(cv2.filter2D(image, cv2.CV_8U, kernel))
    return filtered_images
