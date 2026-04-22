import os
from collections import Counter

import numpy as np
import h5py 
from scipy.io import loadmat
import matplotlib.pyplot as plt
import spectral as spl

def draw(label, name: str = "default", scale: float = 4.0, dpi: int = 200, save_img=None):
    '''
    get classification map , then save to given path
    :param label: classification label, 2D
    :param name: saving path and file's name
    :param scale: scale of image. If equals to 1, then saving-size is just the label-size
    :param dpi: default is OK
    :return: null
    '''
    fig, ax = plt.subplots()
    numlabel = np.array(label)
    v = spl.imshow(classes=numlabel.astype(np.int16), fignum=fig.number)
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    print(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
    fig.set_size_inches(label.shape[1] * scale * 0.5 / dpi, label.shape[0] * scale * 0.5 / dpi)
    foo_fig = plt.gcf()  # 'get current figure'
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    if save_img:
        foo_fig.savefig(name + '.png', format='png', transparent=True, dpi=dpi, pad_inches=0)
    print(name + '.png')

# same to split_data
def data_info(train_label=None, val_label=None, test_label=None, start=0):
    class_num = np.max(train_label)

    if train_label is not None and val_label is not None and test_label is not None:
        total_train_pixel = 0
        total_val_pixel = 0
        total_test_pixel = 0
        train_mat_num = Counter(train_label.flatten())
        val_mat_num = Counter(val_label.flatten())
        test_mat_num = Counter(test_label.flatten())

        for i in range(start, class_num+1):
            print("class", i, "\t", train_mat_num[i],"\t", val_mat_num[i],"\t", test_mat_num[i])
            total_train_pixel += train_mat_num[i]
            total_val_pixel += val_mat_num[i]
            total_test_pixel += test_mat_num[i]
        print("total", "    \t", total_train_pixel, "\t", total_val_pixel, "\t", total_test_pixel)
    
    elif train_label is not None and val_label is not None:
        total_train_pixel = 0
        total_val_pixel = 0
        train_mat_num = Counter(train_label.flatten())
        val_mat_num = Counter(val_label.flatten())

        for i in range(start, class_num+1):
            print("class", i, "\t", train_mat_num[i],"\t", val_mat_num[i])
            total_train_pixel += train_mat_num[i]
            total_val_pixel += val_mat_num[i]
        print("total", "    \t", total_train_pixel, "\t", total_val_pixel)
    
    elif train_label is not None:
        total_pixel = 0
        data_mat_num = Counter(train_label.flatten())

        for i in range(start, class_num+1):
            if data_mat_num[i] == 0:
                continue
            print("class", i, "\t", data_mat_num[i])
            total_pixel += data_mat_num[i]
        print("total:   ", total_pixel)
        
    else:
        raise ValueError("labels are None")
    

def min_max_data(savepath):
    image_data = loadmat(r"C:\Users\jc962911\Project\datasets\OSD\HSI_Gulf_of_Mexico_dataset\GM01.mat") 
    print("GM01", image_data.keys())
    image_hsi = image_data['img']
    print(image_hsi.shape, np.max(image_hsi), np.min(image_hsi))
    label_hsi = image_data['map']
    print(np.unique(label_hsi))
    draw(label_hsi, name="GM01_label", save_img=True)
    # plt.axis('off')
    # plt.savefig(os.path.join(savepath, "GM01_label.png"))


    image_data = loadmat(r"C:\Users\jc962911\Project\datasets\OSD\HSI_Gulf_of_Mexico_dataset\GM02.mat") 
    print("GM02", image_data.keys())
    image_hsi = image_data['img']
    print(image_hsi.shape, np.max(image_hsi), np.min(image_hsi))
    label_hsi = image_data['map']
    print(np.unique(label_hsi))
    draw(label_hsi, name="GM02_label", save_img=True)
    # # plt.axis('off')
    # # plt.savefig(os.path.join(savepath, "GM02_label.png"))

    # image_data = loadmat(r"C:\Users\jc962911\Project\datasets\OSD\HSI_Gulf_of_Mexico_dataset\GM03.mat") 
    # print("GM03", image_data.keys())
    # image_hsi = image_data['img']
    # print(image_hsi.shape, np.max(image_hsi), np.min(image_hsi))
    # label_hsi = image_data['map']
    # print(np.unique(label_hsi))
    # draw(label_hsi, name="GM03_label", save_img=True)
    # # plt.axis('off')
    # # plt.savefig(os.path.join(savepath, "GM03_label.png"))
    
    # image_data = loadmat(r"C:\Users\jc962911\Project\datasets\OSD\HSI_Gulf_of_Mexico_dataset\GM04.mat") 
    # print("GM04", image_data.keys())
    # image_hsi = image_data['img']
    # print(image_hsi.shape, np.max(image_hsi), np.min(image_hsi))
    # label_hsi = image_data['map']
    # print(np.unique(label_hsi))
    # draw(label_hsi, name="GM04_label", save_img=True)
    # # plt.axis('off')
    # # plt.savefig(os.path.join(savepath, "GM04_label.png"))
    
    # image_data = loadmat(r"C:\Users\jc962911\Project\datasets\OSD\HSI_Gulf_of_Mexico_dataset\GM05.mat") 
    # print("GM05", image_data.keys())
    # image_hsi = image_data['img']
    # print(image_hsi.shape, np.max(image_hsi), np.min(image_hsi))
    # label_hsi = image_data['map']
    # print(np.unique(label_hsi))
    # draw(label_hsi, name="GM05_label", save_img=True)
    # # plt.axis('off')
    # # plt.savefig(os.path.join(savepath, "GM05_label.png"))
    
    # image_data = loadmat(r"C:\Users\jc962911\Project\datasets\OSD\HSI_Gulf_of_Mexico_dataset\GM06.mat") 
    # print("GM06", image_data.keys())
    # image_hsi = image_data['img']
    # print(image_hsi.shape, np.max(image_hsi), np.min(image_hsi))
    # label_hsi = image_data['map']
    # print(np.unique(label_hsi))
    # draw(label_hsi, name="GM06_label", save_img=True)
    
    # image_data = loadmat(r"C:\Users\jc962911\Project\datasets\OSD\HSI_Gulf_of_Mexico_dataset\GM07.mat") 
    # print("GM07", image_data.keys())
    # image_hsi = image_data['img']
    # print(image_hsi.shape, np.max(image_hsi), np.min(image_hsi))
    # label_hsi = image_data['map']
    # print(np.unique(label_hsi))
    # draw(label_hsi, name="GM07_label", save_img=True)
    
    # image_data = loadmat(r"C:\Users\jc962911\Project\datasets\OSD\HSI_Gulf_of_Mexico_dataset\GM08.mat") 
    # print("GM08", image_data.keys())
    # image_hsi = image_data['img']
    # print(image_hsi.shape, np.max(image_hsi), np.min(image_hsi))
    # label_hsi = image_data['map']
    # print(np.unique(label_hsi))
    # draw(label_hsi, name="GM08_label", save_img=True)
    
    # image_data = loadmat(r"C:\Users\jc962911\Project\datasets\OSD\HSI_Gulf_of_Mexico_dataset\GM09.mat") 
    # print("GM09", image_data.keys())
    # image_hsi = image_data['img']
    # print(image_hsi.shape, np.max(image_hsi), np.min(image_hsi))
    # label_hsi = image_data['map']
    # print(np.unique(label_hsi))
    # draw(label_hsi, name="GM09_label", save_img=True)
    
    # image_data = loadmat(r"C:\Users\jc962911\Project\datasets\OSD\HSI_Gulf_of_Mexico_dataset\GM10.mat") 
    # print("GM010", image_data.keys())
    # image_hsi = image_data['img']
    # print(image_hsi.shape, np.max(image_hsi), np.min(image_hsi))
    # label_hsi = image_data['map']
    # print(np.unique(label_hsi))
    # draw(label_hsi, name="GM10_label", save_img=True)
    
    # image_data = loadmat(r"C:\Users\jc962911\Project\datasets\OSD\HSI_Gulf_of_Mexico_dataset\GM11.mat") 
    # print("GM011", image_data.keys())
    # image_hsi = image_data['img']
    # print(image_hsi.shape, np.max(image_hsi), np.min(image_hsi))
    # label_hsi = image_data['map']
    # print(np.unique(label_hsi))
    # draw(label_hsi, name="GM11_label", save_img=True)
    
    # image_data = loadmat(r"C:\Users\jc962911\Project\datasets\OSD\HSI_Gulf_of_Mexico_dataset\GM12.mat") 
    # print("GM012", image_data.keys())
    # image_hsi = image_data['img']
    # print(image_hsi.shape, np.max(image_hsi), np.min(image_hsi))
    # label_hsi = image_data['map']
    # print(np.unique(label_hsi))
    # draw(label_hsi, name="GM12_label", save_img=True)
    
    # image_data = loadmat(r"C:\Users\jc962911\Project\datasets\OSD\HSI_Gulf_of_Mexico_dataset\GM13.mat") 
    # print("GM013", image_data.keys())
    # image_hsi = image_data['img']
    # print(image_hsi.shape, np.max(image_hsi), np.min(image_hsi))
    # label_hsi = image_data['map']
    # print(np.unique(label_hsi))
    # draw(label_hsi, name="GM13_label", save_img=True)
    
    # image_data = loadmat(r"C:\Users\jc962911\Project\datasets\OSD\HSI_Gulf_of_Mexico_dataset\GM14.mat") 
    # print("GM014", image_data.keys())
    # image_hsi = image_data['img']
    # print(image_hsi.shape, np.max(image_hsi), np.min(image_hsi))
    # label_hsi = image_data['map']
    # print(np.unique(label_hsi))
    # draw(label_hsi, name="GM14_label", save_img=True)
    
    # image_data = loadmat(r"C:\Users\jc962911\Project\datasets\OSD\HSI_Gulf_of_Mexico_dataset\GM15.mat") 
    # print("GM015", image_data.keys())
    # image_hsi = image_data['img']
    # print(image_hsi.shape, np.max(image_hsi), np.min(image_hsi))
    # label_hsi = image_data['map']
    # print(np.unique(label_hsi))
    # draw(label_hsi, name="GM15_label", save_img=True)
    
    # image_data = loadmat(r"C:\Users\jc962911\Project\datasets\OSD\HSI_Gulf_of_Mexico_dataset\GM16.mat") 
    # print("GM016", image_data.keys())
    # image_hsi = image_data['img']
    # print(image_hsi.shape, np.max(image_hsi), np.min(image_hsi))
    # label_hsi = image_data['map']
    # print(np.unique(label_hsi))
    # draw(label_hsi, name="GM16_label", save_img=True)
    
    # image_data = loadmat(r"C:\Users\jc962911\Project\datasets\OSD\HSI_Gulf_of_Mexico_dataset\GM17.mat") 
    # print("GM017", image_data.keys())
    # image_hsi = image_data['img']
    # print(image_hsi.shape, np.max(image_hsi), np.min(image_hsi))
    # label_hsi = image_data['map']
    # print(np.unique(label_hsi))
    # draw(label_hsi, name="GM17_label", save_img=True)
    
    # image_data = loadmat(r"C:\Users\jc962911\Project\datasets\OSD\HSI_Gulf_of_Mexico_dataset\GM18.mat") 
    # print("GM018", image_data.keys())
    # image_hsi = image_data['img']
    # print(image_hsi.shape, np.max(image_hsi), np.min(image_hsi))
    # label_hsi = image_data['map']
    # print(np.unique(label_hsi))
    # draw(label_hsi, name="GM18_label", save_img=True)
    
    # h5py_data = h5py.File(r"C:\Users\jc962911\Project\datasets\OSD\HSI_Penglai_dataset\penglai.mat") 
    # print("ploil", h5py_data.keys())
    # image_hsi = h5py_data['img']
    # print(image_hsi.shape, np.max(image_hsi), np.min(image_hsi))
    # label_hsi = h5py_data['gt']
    # print(np.unique(label_hsi))
    # draw(label_hsi, name="penglai_label", save_img=True)

    # image_data = loadmat(r"C:\Users\jc962911\Project\datasets\OSD\HSI_Deep_water_horizon_dataset\dwh_1.mat") 
    # print("dwh_1", image_data.keys())
    # image_hsi = image_data['img']
    # print(image_hsi.shape, np.max(image_hsi), np.min(image_hsi))

    # image_data = loadmat(r"C:\Users\jc962911\Project\datasets\OSD\HSI_Deep_water_horizon_dataset\dwh_2.mat") 
    # print("dwh_2", image_data.keys())
    # image_hsi = image_data['img']
    # print(image_hsi.shape, np.max(image_hsi), np.min(image_hsi))

if __name__ == "__main__":
    save_path = r"C:\Users\jc962911\Project\Oil_Spill_Detection\data_process\ground truth"
    min_max_data(save_path)
