import random
import shutil
import os

def split(path, mask_path, lane_path):
    os.mkdir(path + 'train')
    os.mkdir(path + 'val')
    os.mkdir(mask_path + 'train')
    os.mkdir(mask_path + 'val')
    os.mkdir(lane_path + 'train')
    os.mkdir(lane_path + 'val')
    val_index = random.sample(range(660), 200)
    for i in range(660):
        if i in val_index:
            shutil.move(path+'{}.png'.format(i), path + 'val')
            shutil.move(mask_path+'{}.png'.format(i), mask_path + 'val')
            shutil.move(lane_path+'{}.png'.format(i), lane_path + 'val')
        else:
            shutil.move(path+'{}.png'.format(i), path + 'train')
            shutil.move(mask_path+'{}.png'.format(i), mask_path + 'train')
            shutil.move(lane_path+'{}.png'.format(i), lane_path + 'train')


if __name__ == '__main__':
    path = "data/images/"
    mask_path = "data/masks/"
    lane_path = "data/lanes/"
    split(path, mask_path, lane_path)


