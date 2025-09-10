import os
import argparse
import rasterio
import cv2
import numpy as np

def make_patches(im_dir, out_dir, folder_name="RGB_1m", 
                 txt_name="train.txt", patch_size=512, stride=400, out_format='tif'):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    image_dir = os.path.join(im_dir, folder_name)
    txt_path = os.path.join(im_dir, txt_name)

    with open(os.path.join(txt_path), "r") as f:
        file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
    img_dirs = [os.path.join(image_dir, x + "." + out_format) for x in file_names]

    # image_dataset = []  
    # for path, subdirs, files in os.walk(im_dir):
        # subdirs.sort()
        # dirname = path.split(os.path.sep)[-1]
        # images = sorted(os.listdir(path))  #List of all image names in this subdirectory
    
    index = 0
    for image_name in img_dirs:
        if image_name.endswith(".jpg") or \
            image_name.endswith(".png") or \
            image_name.endswith(".tif"):   # Only read jpg images...

            with rasterio.open(os.path.join(image_name)) as src:
                # image = cv2.imread(os.path.join(path, image_name), 1)  # Read each image as BGR
                image = src.read().transpose(1, 2, 0)
                profile = src.profile
                # print(os.path.join(path, image_name))
                # print(image)
                print(image.shape)
            
            '''
            如果 x + patch_size > image.shape[1]，则 x = image.shape[1] - patch_size
            x = image.shape[1] - patch_size 这句话的意思是，在最后一个patch的时候通过向后重叠的方式来生成patch，以保证不会有遗漏
            '''
            for x in range(0, image.shape[0], stride):
                for y in range(0, image.shape[1], stride):
                    
                    if (y + patch_size > image.shape[0]): # check if x is out of bound
                        y = image.shape[0] - patch_size
                    if (x + patch_size > image.shape[1]): # check if y is out of bounds
                        x = image.shape[1] - patch_size
                    tile = image[y : y+patch_size, x : x+patch_size]

                    channel_0 = tile[:, :, 0:1] # 使用切片保持维度
                    tile = np.concatenate((tile, channel_0), axis=2)

                    im_name = txt_name.split('.')[0] + "_" + str(index) + '.'+ out_format
                    index = index + 1
                    # print(im_name)

                    cv2.imwrite(os.path.join(out_dir, im_name), tile)
            # print("tile", tile.shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, default=None, help='directory containing images to be patchified', required=True)
    parser.add_argument('--out_dir', type=str, default=None, help='directory to save the patches to', required=True)
    parser.add_argument('--folder_name', type=str, default="RGB_1m", help='name of the folder to read images from', required=True)
    parser.add_argument('--txt_name', type=str, default="train.txt", help='name of the text file containing image names', required=True)
    parser.add_argument('--patch_size', type=int, default=512, help='size of the square patches to be created in pixels per side') # 256
    parser.add_argument('--stride', type=int, default=400, help='number of pixels to move the window creating the patches') # 400
    parser.add_argument('--output_format', choices=['png', 'tif', 'jpg'], default='png')
    opt = parser.parse_args()

    make_patches(opt.in_dir, opt.out_dir, opt.folder_name, opt.txt_name, opt.patch_size, opt.stride, out_format=opt.output_format)



# ls -l | grep "^-" | wc -l
# ls -lR | grep "^-" | wc -l
# 1600 - 400
# 160 - 40

# python patchify.py --in_dir /home/icclab/Documents/lqw/DatasetMMF/ISASeg --out_dir /home/icclab/Documents/lqw/DatasetMMF/ISASeg_train/RGB_1m --folder_name RGB_1m --txt_name train.txt --output_format tif --patch_size 400 --stride 320
# python patchify.py --in_dir /home/icclab/Documents/lqw/DatasetMMF/ISASeg --out_dir /home/icclab/Documents/lqw/DatasetMMF/ISASeg_train/Label_train --folder_name Label_train --txt_name train.txt --output_format png --patch_size 400 --stride 320


# python patchify.py --in_dir /home/icclab/Documents/lqw/DatasetMMF/ISASeg --out_dir /home/icclab/Documents/lqw/DatasetMMF/ISASeg_train/Sentinel1 --folder_name Sentinel1 --txt_name train.txt --output_format tif --patch_size 40 --stride 32





# python patchify.py --in_dir /home/icclab/Documents/lqw/DatasetMMF/ISASeg --out_dir /home/icclab/Documents/lqw/DatasetMMF/ISASeg_train/RGB_1m --folder_name RGB_1m --txt_name train.txt --output_format tif --patch_size 800 --stride 640
# python patchify.py --in_dir /home/icclab/Documents/lqw/DatasetMMF/ISASeg --out_dir /home/icclab/Documents/lqw/DatasetMMF/ISASeg_train/Label_train --folder_name Label_train --txt_name train.txt --output_format png --patch_size 800 --stride 640


# python patchify.py --in_dir /home/icclab/Documents/lqw/DatasetMMF/ISASeg --out_dir /home/icclab/Documents/lqw/DatasetMMF/ISASeg_train/Sentinel1 --folder_name Sentinel1 --txt_name train.txt --output_format tif --patch_size 80 --stride 64
