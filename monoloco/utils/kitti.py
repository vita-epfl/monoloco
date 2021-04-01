
import os
import glob

import numpy as np


def get_calibration(path_txt):
    """Read calibration parameters from txt file:
    For the left color camera we use P2 which is K * [I|t]

    P = [fu, 0, x0, fu*t1-x0*t3
         0, fv, y0, fv*t2-y0*t3
         0, 0,  1,          t3]

    check also http://ksimek.github.io/2013/08/13/intrinsic/

    Simple case test:
    xyz = np.array([2, 3, 30, 1]).reshape(4, 1)
    xyz_2 = xyz[0:-1] + tt
    uv_temp = np.dot(kk, xyz_2)
    uv_1 = uv_temp / uv_temp[-1]
    kk_1 = np.linalg.inv(kk)
    xyz_temp2 = np.dot(kk_1, uv_1)
    xyz_new_2 = xyz_temp2 * xyz_2[2]
    xyz_fin_2 = xyz_new_2 - tt
    """

    with open(path_txt, "r") as ff:
        file = ff.readlines()
    p2_str = file[2].split()[1:]
    p2_list = [float(xx) for xx in p2_str]
    p2 = np.array(p2_list).reshape(3, 4)

    p3_str = file[3].split()[1:]
    p3_list = [float(xx) for xx in p3_str]
    p3 = np.array(p3_list).reshape(3, 4)

    kk, tt = get_translation(p2)
    kk_right, tt_right = get_translation(p3)

    return [kk, tt], [kk_right, tt_right]


def get_translation(pp):
    """Separate intrinsic matrix from translation and convert in lists"""

    kk = pp[:, :-1]
    f_x = kk[0, 0]
    f_y = kk[1, 1]
    x0, y0 = kk[2, 0:2]
    aa, bb, t3 = pp[0:3, 3]
    t1 = float((aa - x0*t3) / f_x)
    t2 = float((bb - y0*t3) / f_y)
    tt = [t1, t2, float(t3)]
    return kk.tolist(), tt


def get_simplified_calibration(path_txt):

    with open(path_txt, "r") as ff:
        file = ff.readlines()

    for line in file:
        if line[:4] == 'K_02':
            kk_str = line[4:].split()[1:]
            kk_list = [float(xx) for xx in kk_str]
            kk = np.array(kk_list).reshape(3, 3).tolist()
            return kk

    raise ValueError('Matrix K_02 not found in the file')


def check_conditions(line, category, method, thresh=0.3):
    """Check conditions of our or m3d txt file"""

    check = False
    assert category in ['pedestrian', 'cyclist', 'all']

    if category == 'all':
        category = ['pedestrian', 'person_sitting', 'cyclist']

    if method == 'gt':
        if line.split()[0].lower() in category:
            check = True

    else:
        conf = float(line[15])
        if line[0].lower() in category and conf >= thresh:
            check = True
    return check


def get_difficulty(box, trunc, occ):

    hh = box[3] - box[1]
    if hh >= 40 and trunc <= 0.15 and occ <= 0:
        cat = 'easy'
    elif trunc <= 0.3 and occ <= 1 and hh >= 25:
        cat = 'moderate'
    elif trunc <= 0.5 and occ <= 2 and hh >= 25:
        cat = 'hard'
    else:
        cat = 'excluded'
    return cat


def split_training(names_gt, path_train, path_val):
    """Split training and validation images"""
    set_gt = set(names_gt)
    set_train = set()
    set_val = set()

    with open(path_train, "r") as f_train:
        for line in f_train:
            set_train.add(line[:-1] + '.txt')
    with open(path_val, "r") as f_val:
        for line in f_val:
            set_val.add(line[:-1] + '.txt')

    set_train = set_gt.intersection(set_train)
    set_train.remove('000518.txt')
    set_train.remove('005692.txt')
    set_train.remove('003009.txt')
    set_train = tuple(set_train)
    set_val = tuple(set_gt.intersection(set_val))
    assert set_train and set_val, "No validation or training annotations"
    return set_train, set_val


def factory_basename(dir_ann, dir_gt):
    """ Return all the basenames in the annotations folder corresponding to validation images"""

    # Extract ground truth validation images
    names_gt = tuple(os.listdir(dir_gt))
    path_train = os.path.join('splits', 'kitti_train.txt')
    path_val = os.path.join('splits', 'kitti_val.txt')
    _, set_val_gt = split_training(names_gt, path_train, path_val)
    set_val_gt = {os.path.basename(x).split('.')[0] for x in set_val_gt}

    # Extract pifpaf files corresponding to validation images
    list_ann = glob.glob(os.path.join(dir_ann, '*.json'))
    set_basename = {os.path.basename(x).split('.')[0] for x in list_ann}
    set_val = set_basename.intersection(set_val_gt)
    assert set_val, " Missing json annotations file to create txt files for KITTI datasets"
    return set_val


def read_and_rewrite(path_orig, path_new):
    """Read and write same txt file. If file not found, create open file"""
    try:
        with open(path_orig, "r") as f_gt:
            with open(path_new, "w+") as ff:
                for line_gt in f_gt:
                    # if check_conditions(line_gt, category='all', method='gt'):
                    line = line_gt.split()
                    hwl = [float(x) for x in line[8:11]]
                    hwl = " ".join([str(i)[0:4] for i in hwl])
                    temp_1 = " ".join([str(i) for i in line[0: 8]])
                    temp_2 = " ".join([str(i) for i in line[11:]])
                    line_new = temp_1 + ' ' + hwl + ' ' + temp_2 + '\n'
                    ff.write("%s" % line_new)
    except FileNotFoundError:
        ff = open(path_new, "a+")
        ff.close()


def find_cluster(dd, clusters):
    """Find the correct cluster. Above the last cluster goes into "excluded (together with the ones from kitti cat"""

    for idx, clst in enumerate(clusters[:-1]):
        if int(clst) < dd <= int(clusters[idx+1]):
            return clst
    return 'excluded'
