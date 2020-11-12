import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold

def read_imgs_labels(root_path, label_path):
    label_data = pd.read_excel(label_path, dtype={'Subject': 'str'})
    # print(label_data.head())
    emotion_map = {'happiness': 0, 'disgust': 1, 'repression': 2, 'surprise': 3, 'others': 4}
    # print(label_data.columns)
    imgs = []
    labels = []
    cnt = 0
    for index, row in label_data.iterrows():
        # print(row['Subject'])
        img_fold_path = root_path + 'sub' + str(row['Subject']) + '\\' + row['Filename']
        if row['Estimated Emotion'] not in emotion_map:
            continue
        label = emotion_map[row['Estimated Emotion']]
        # print(label)
        # print(img_fold_path)
        # if cnt == 50:
        #     break
        cnt += 1
        print(cnt, img_fold_path)
        
        with os.scandir(img_fold_path) as folder:
            f_list = list(folder)
            len_frames = len(f_list)
            frames = []
            for i in range(15):
                if i == 0:
                    idx = 0
                elif i == 14:
                    idx = len_frames - 1
                else:
                    idx = len_frames // 15 * i
                print('read', f_list[idx].path)
                img_array = cv2.imread(f_list[idx].path)
                # img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                img_array = cv2.resize(img_array, (224, 244))
                frames.append(img_array)
                # print(img_array.shape)
                # break
        imgs.append(frames)
        labels.append(label)

        # break
    return np.array(imgs), np.array(labels)









if __name__ == '__main__':
    root_path = 'G:\\vscode_workspace\\MER\\data\\CASME2_preprocessed_Li Xiaobai\\Cropped\\'
    label_path = 'G:\\vscode_workspace\\MER\\data\\CASME2_preprocessed_Li Xiaobai\\CASME2-coding-20190701.xlsx'
    save_path = 'G:\\vscode_workspace\\MER\\dataset\\CASME2_BGR_224_15frames.pickle'

    imgs, labels = read_imgs_labels(root_path, label_path)
    # exit()
    sfolder = StratifiedKFold(n_splits=10)
    save_data = []
    for train, test in sfolder.split(imgs, labels):
        train_img_fold = imgs[train]
        trian_label_fold = labels[train]
        test_img_fold = imgs[test]
        test_label_fold = labels[test]
        save_data.append({'train_img': train_img_fold,
                        'train_label': trian_label_fold,
                        'test_img': test_img_fold,
                        'test_label': test_label_fold})
        print(train_img_fold.shape, trian_label_fold.shape, test_img_fold.shape, test_label_fold.shape)
    with open(save_path, 'wb') as pfile:
        pickle.dump(save_data, pfile, pickle.HIGHEST_PROTOCOL)
