#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/12/14 18:00
# @Author  : zhouhonghong
# @Email   : zhouhonghong@bupt.edu.cn
import os
def download():
    Set01= "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s01s02.zip"
    Set02= "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s01s03.zip"
    Set03= "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s01s07.zip"
    Set04= "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s02s01.zip"
    Set05= "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s02s03.zip"
    Set06= "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s02s06.zip"
    Set07= "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s02s07.zip"
    Set08= "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s03s02.zip"
    Set09= "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s03s04.zip"
    Set10= "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s03s05.zip"
    Set11= "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s03s06.zip"
    Set12= "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s04s02.zip"
    Set13= "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s04s03.zip"
    Set14= "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s04s06.zip"
    Set15= "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s05s02.zip"
    Set16= "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s05s03.zip"
    Set17= "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s06s02.zip"
    Set18= "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s06s03.zip"
    Set19= "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s06s04.zip"
    Set20= "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s07s01.zip"
    Set21= "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s07s03.zip"
    import urllib.request as Req

    for i in range(1,22):
        s = '%02d' % i
        #print(s)
        name = "Set"+s
        print(name)
        url = locals()[name]
        Req.urlretrieve(url,"./data/demo_{}.zip".format(i))

def unzip_data():
    '''
    unzip the code
    :return:
    '''

    import zipfile
    def unzip(sourceFile, targetPath):
        '''
        :param sourceFile: 待解压zip路径
        :param targetPath: 目标文件目录
        :return:
        '''
        file = zipfile.ZipFile(sourceFile, 'r')
        file.extractall(targetPath)
        print('success to unzip file!')

    source = "./data"
    target = "./unzip"
    if not os.path.exists(source):
        os.mkdir(source)
    if not os.path.exists(target):
        os.mkdir(target)
    for i in range(1, 22):
        filename = "demo_" + str(i)
        path = os.path.join(source, filename + ".zip")
        print(path)
        target_path = os.path.join(target, str(i))
        unzip(path, target_path)


def copy_txt():
    """
    copy the skeleton_pos.txt files to another place.
    dataset root: PROJECT_PATH/datasets/SBU
    and the SBU dataset has the following structure
    - set(01-21)
        - interaction(01-08)
            - skeleton_pos_num.txt
    eg. 20/07/skeleton_pos_001.txt
    :return:
    """
    for i in range(1, 22):
        filename = str(i)
        mainFile = os.path.join("./unzip", filename)
        check = os.listdir(mainFile)
        subFile = check[0] if check[1] == '__MACOSX' else check[1]
        print(subFile)
        sub_path = os.path.join(mainFile, subFile)
        cats = sorted(os.listdir(sub_path))
        cats = cats[1::]
        for cat in cats:
            cat_path = os.path.join(sub_path, cat)
            nums = sorted(os.listdir(cat_path))
            if nums[0] == '.DS_Store':
                nums = nums[1::]
            for num in nums:
                num_path = os.path.join(cat_path, num)
                source_txt_path = os.path.join(num_path, "skeleton_pos.txt")
                # print(source_txt_path)
                target_txt_path = os.path.join('%02d' % i, cat)
                target_name = 'skeleton_pos_' + num + '.txt'
                if not os.path.exists(target_txt_path):
                    os.makedirs(target_txt_path)
                with open(source_txt_path) as f:
                    data = f.read()
                with open(os.path.join(target_txt_path, target_name), 'w') as f:
                    f.write(data)

if __name__ == '__main__':
    copy_txt()