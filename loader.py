import torch
from torchvision import datasets
from torchvision import transforms
import numpy as np
import os
import cv2

"""
=================================================
        LOADING FRAMES AND GROUND TRUTH
=================================================
"""

# suitable extensions
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

# get current path
CWD = os.getcwd()


# height and width for normalization
def height_and_width(video_root):
    for file in os.listdir(path=video_root):
        filename, file_extension = os.path.splitext(file)
        if file_extension in IMG_EXTENSIONS:
            img = cv2.imread(video_root + '/' + file, cv2.IMREAD_UNCHANGED)
            return img.shape[0], img.shape[1]
    return None, None


# normalization by image height and weight
def gt_normalize(gt, h, w):
    gt[:, 0] /= h
    gt[:, 1] /= w
    gt[:, 2] /= h
    gt[:, 3] /= w
    return gt


# class for store each video
class VideoBuffer(object):

    def __init__(self, title, height, width, gt_load, frames_load, complete_sequences, set_len):
        self.title = title
        self.height = height
        self.width = width
        self.ground_truth_loader = gt_load
        self.frames_loader = frames_load
        self.complete_sequences = complete_sequences
        self.set_len = set_len
        self.test_fails = 0
        self.test_predictions = []

    def sample_frames(self, start_ind, left_frames):
        pass

    def sample_gt(self, start_ind, left_frames):
        pass


def load_videos(titles_list, T, img_size):
    # all roots to all videos is list
    roots_list = [CWD + "/videos/" + x for x in titles_list]

    # Resize to image_size
    # Transform to torch tensor
    # Normalize : mean and std for 3 channels
    transform = transforms.Compose([transforms.Resize(img_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # set of videos, could be Training or Test
    videos_list = []

    for video_title, root in zip(titles_list, roots_list):
        # load frames
        frame_set = datasets.ImageFolder(root=root, transform=transform)
        frame_loader = torch.utils.data.DataLoader(frame_set, batch_size=T, shuffle=False)

        # load ground truth
        gt_path = root + "/frames/groundtruth.txt"
        gt_txt = np.loadtxt(gt_path, delimiter=',', dtype=np.float32)
        gt_tens = torch.from_numpy(gt_txt)
        height, width = height_and_width(root + '/frames/')
        gt_tens = gt_normalize(gt_tens, height, width)
        gt_loader = torch.utils.data.DataLoader(gt_tens, batch_size=T, shuffle=False)

        # add to training set
        vid = VideoBuffer(title=video_title, height=height,
                          width=width, gt_load=gt_loader, frames_load=frame_loader,
                          complete_sequences=frame_set.__len__() // T,
                          set_len=frame_set.__len__())
        videos_list.append(vid)

    return videos_list
