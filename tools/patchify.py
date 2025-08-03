import os
import cv2
import argparse
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

def make_patches(im_dir, out_dir, patch_size=256, stride=200, format='map'):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # image_dataset = []  
    for path, subdirs, files in os.walk(im_dir):
        subdirs.sort()
        dirname = path.split(os.path.sep)[-1]
        images = sorted(os.listdir(path))  #List of all image names in this subdirectory
        # print("images: ", images)
        
        # TODO: create out folder if not exists
        
        index = 0
        for i, image_name in enumerate(images):
            if image_name.endswith(".jpg") \
                or image_name.endswith(".png") \
                or image_name.endswith(".mat") \
                or image_name.endswith(".tif"):   # Only read jpg images...

                # image = cv2.imread(path + "/" + image_name, 1)  # Read each image as BGR
                if format == 'img' or format == 'map':
                    image = sio.loadmat(path + "/" + image_name)[format]
                else:
                    image = cv2.imread(path + "/" + image_name)
                # print(image.shape)
                
                '''
                如果 x + patch_size > image.shape[1]，则 x = image.shape[1] - patch_size
                x = image.shape[1] - patch_size 这句话的意思是，在最后一个patch的时候通过向后重叠的方式来生成patch，以保证不会有遗漏
                '''
                print("image.shape: ", image.shape[0], image.shape[1])
                for x in range(0, image.shape[0], stride): 
                    for y in range(0, image.shape[1], stride):

                        # print("im_name 1", image_name)
                        im_name = image_name[:-4] + '_' + str(x) + '_' + str(y)
                        # print("im_name ", image_name[:-4], str(x), str(y))

                        if (x + patch_size > image.shape[0]): # check if x is out of bound
                            x = image.shape[0] - patch_size
                        if (y + patch_size > image.shape[1]): # check if y is out of bounds
                            y = image.shape[1] - patch_size
                        tile = image[x : x + patch_size, y : y + patch_size]
                        
                        print("tile.shape: ", tile.shape)
                        # print("tile.range: ", x, x + patch_size, y, y + patch_size)
                        # cv2.imwrite(os.path.join(out_dir, im_name + ".png"), tile)
                        

                        out_img_path = os.path.join(out_dir, "{}.mat".format(index))
                        sio.savemat(out_img_path, {"img": tile}, do_compression=True)

                        # out_sar_path = os.path.join(out_dir, "{}.mat".format(index))
                        # sio.savemat(out_sar_path, {"sar": tile}, do_compression=True)
                        # out_sar_path = os.path.join(out_dir, "{}.png".format(str(index)))
                        # cv2.imwrite(out_sar_path, tile)
                        
                        # out_mask_path = os.path.join(out_dir, "{}.mat".format(index))
                        # sio.savemat(out_mask_path, {"map": tile}, do_compression=True)
                        # out_mask_path = os.path.join(out_dir, "{}.png".format(index))
                        # # # print(np.unique(tile))
                        # tile[tile == 1] = 255
                        # image_3d = np.repeat(tile[:, :, np.newaxis], 3, axis=2)
                        # print("image_3d.shape: ", image_3d.shape)
                        # cv2.imwrite(out_mask_path, image_3d)
                        # # plt.imsave(out_mask_path, image_3d)
                        index += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, default=None, help='directory containing images to be patchified', required=True)
    parser.add_argument('--out_dir', type=str, default=None, help='directory to save the patches to', required=True)
    parser.add_argument('--patch_size', type=int, default=256, help='size of the square patches to be created in pixels per side')
    parser.add_argument('--stride', type=int, default=200, help='number of pixels to move the window creating the patches')
    parser.add_argument('--format', type=str, choices=['img', 'map', 'sar'], default='img')
    opt = parser.parse_args()

    make_patches(opt.in_dir, opt.out_dir, opt.patch_size, opt.stride, format=opt.format)


# python tools/patchify.py --format img --in_dir /home/leo/MMF/OSD/TestCopyPost --out_dir /home/leo/MMF/OSDT/test2/imagesPng128 --patch_size 128 --stride 64
# python tools/patchify.py --format map --in_dir /home/leo/MMF/OSD/TestCopyPost --out_dir /home/leo/MMF/OSDT/test2/masksPng128 --patch_size 128 --stride 64
# python tools/patchify.py --format sar --in_dir /home/leo/MMF/OSD/TestSAR --out_dir /home/leo/MMF/OSDT/test2/sarPng128 --patch_size 128 --stride 64


# python tools/patchify.py --format img --in_dir /home/leo/MMF/OSD/TestCopyPost --out_dir /home/leo/MMF/OSDT/test2/imagesOrder128 --patch_size 128 --stride 64
# python tools/patchify.py --format map --in_dir /home/leo/MMF/OSD/TestCopyPost --out_dir /home/leo/MMF/OSDT/test2/masksOrder128 --patch_size 128 --stride 64
# python tools/patchify.py --format sar --in_dir /home/leo/MMF/OSD/TestSAR --out_dir /home/leo/MMF/OSDT/test2/sarOrder128 --patch_size 128 --stride 64

