from glob import glob
import os
import cv2
import imageio

from utils.vis import pad_frames_to_same_size

exp_dir = 'dump/gradient_realworld/objectnav/vis/stair'
img_dir = os.path.join(exp_dir, 'vis')
video_save_path = os.path.join(exp_dir, 'video.mp4')

total_num = len(glob(os.path.join(img_dir, '*.png')))
imgs = []
for i in range(total_num):
    img_path = os.path.join(img_dir, f'vis_{i}.png')
    if os.path.exists(img_path):
        imgs.append(img_path)
print(imgs)

img_list = []
for img_path in imgs:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img_list.append(img)

same_size_frames = pad_frames_to_same_size(img_list)
imageio.mimsave(video_save_path, same_size_frames, fps=2)