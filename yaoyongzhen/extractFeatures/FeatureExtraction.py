"""
@date: Created on 2018/5/23
@author: yaoyongzhen
@notes: 利用OpenSmile提取IS10特征集
"""
import os

#config
data_dir = "/Users/yaoyongzhen/Desktop/情感计算/IEMOCAP_full_release/"
list_file = 'cat.txt'
feature_dir= 'data/'
z_norm = False
config_path = '/Users/yaoyongzhen/Desktop/情感计算/extrafeature/IS10_paraling.conf'
file_list_file = open(list_file,'r')
file_list = []
label_list = []

for line in file_list_file.readlines():
            line=line.strip('\r\n')
            line_array = line.split("\t")
            file_dir = line_array[0]
            file_list.append(file_dir)
            label_list.append(line_array[1])

print (len(file_list),"files in total\n")

for i in range(len(file_list)):
    file_dir_full = data_dir + file_list[i]
    print ("No.",i,"extracting")
    print('file_dir_full',file_dir_full)

    file_name = file_list[i].strip(".wav")
    feature_name = feature_dir+file_name+".npy"
    print('feature_name',feature_name)
    save_dir = feature_name.strip(feature_name.split('/')[-1])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 利用OpenSmile提取IS10特征集
    order = 'SMILExtract -C '+config_path+' -I '+file_dir_full+' -O '+feature_name
    os.system(order)
    print ("No.",i,"saved\n")
