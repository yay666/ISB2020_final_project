#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sys
from scipy.spatial.distance import euclidean, cosine
import math
from math import pi
import scipy
from skimage.io import imread
from skimage.measure import label, regionprops
from skimage.color import label2rgb
import os
from os import listdir
from PIL import Image, ImageDraw, ImageFont

from resnet50 import res_model
from skimage import morphology
from scipy import ndimage
import cv2
import time
from skimage.segmentation import clear_border
import pickle
from matplotlib import pyplot as plt
import itertools
from datetime import datetime

from scipy.optimize import linear_sum_assignment
import sqlite3
from track_module import compute_overlap_matrix, compute_overlap_pair, compute_overlap_single, generate_traj_seri, relabel_traj, record_traj_start_end, judge_mol_type, search_false_link, judge_border, find_border_obj, break_link, connect_link, false_seg_mark, judge_traj_am, judge_apoptosis_tracklet, traj_start_end_info, am_obj_info, compute_specific_overlap, compute_cost, calculate_area_penalty, find_am_sisters, cal_cell_fusion, cal_cell_split, find_mitosis_pairs_to_break, find_fuse_pairs_to_break, find_split_pairs_to_break, judge_fuse_type, judge_split_type
from cnn_prep_data import generate_single_cell_img_edt
import glob
import pipe_util2

# In[2]:


# ----parameter setting -----------
# depend on: cell type, time interval-------
# 1/2 max distance two traj end for selecting possible mitosis: cell type
# and time interval
mitosis_max_dist = 50
# size_similarity=1-abs(size1-size2)/(size1+size2) size similarity between
# two possible sister cells
simi_thres = 0.7
# for judging traj beginning or end is mitosis or apoptosi or not: time
# interval
t_search_range = 3
traj_len_thres = 6  # for judging single cell, if trajectory length is larger than this value, it is probably single cell:time  interval


# In[3]:


# main_path='/home/zoro/Desktop/experiment_data/2019-02-22_HK2_3d/'
# output_path=main_path+'/output/'


def icnn_seg_load_weight(icnn_seg_weights):
    obj_h=128
    obj_w=128
    input_shape=(obj_h,obj_w,1)
    nb_class=3
    icnn_seg=res_model(input_shape,nb_class)
    icnn_seg.load_weights(icnn_seg_weights)
    return icnn_seg


def traj_reconganize1(img_path, output_path, icnn_seg_weights, DIC_chan_label, obj_h = 128, obj_w = 128):

    """

    :param img_path: the original image folder
    :param output_path: the _output folder.
    :return: generating several files under the _output folder
    """
    
    icnn = icnn_seg_load_weight(icnn_seg_weights)
    
    # preparing paths
    print("processing %s" % (img_path), flush=True)
    img_path = pipe_util2.folder_verify(img_path)
    img_list = sorted(glob.glob(img_path + "*" + DIC_chan_label + "*"))
    for i in range(len(img_list)):
        img_list[i] = os.path.basename(img_list[i])

    seg_path = pipe_util2.folder_verify(output_path) + 'seg/'
    seg_img_list = sorted(listdir(seg_path))

    dir_path = pipe_util2.folder_verify(output_path)
#     df = pd.read_csv(dir_path + 'Per_Object.csv')
#     relation_df=pd.read_csv(dir_path + 'Per_Relationships.csv')

    # reading cell_track.db and am_record.csv
    conn = sqlite3.connect(dir_path + 'cell_track.db')
    df = pd.read_sql_query('SELECT * FROM Per_Object', conn)
    relation_df = pd.read_sql_query('SELECT * FROM Per_Relationships', conn)
    am_record = pd.read_csv(dir_path + 'am_record.csv')
    t_span = max(df['ImageNumber'])

    # ---------------------------------------
    pipe_util2.print_time()

    df = relabel_traj(df)
    traj_start, traj_end = record_traj_start_end(df)
    # ---------------------------------------
    #hj_util.print_time()

    # find candidate mitosis pairs from apoptosis and mitosis cells
    am_arr, am_xy, am_area = am_obj_info(am_record, df)

    # ---------------------------------------
    #hj_util.print_time()

    # F is a n by 5 matrix
    # including obj_x, obj_y, img_num, obj_num, area
    F = np.column_stack((am_xy, am_arr, np.expand_dims(am_area[:], axis=1)))

    # ---------------------------------------
    #hj_util.print_time()

    candi_am_sisters = find_am_sisters(F,
                                       mitosis_max_distance=mitosis_max_dist,
                                       size_simi_thres=simi_thres).tolist()
    # ---------------------------------------
    #hj_util.print_time()

    # find fuse and split segmentation, border obj and false link
    prefuse_cells = []
    prefuse_group = []  # each element is a list include all prefuse cells in a fuse event, corresponding to postfuse_cells
    postfuse_cells = []  # include: img_num,obj_num

    presplit_cells = []  # include: img_num,obj_num
    postsplit_group = []  # each element is a list include all postsplit cells in a split event
    postsplit_cells = []

    false_link = []
    border_obj = []

    for img_num in range(1, len(img_list) + 1):
        print("processing %d/%d images" % (img_num, len(img_list)), flush=True)
        # -------------find obj on border------------------
        border_obj.extend(find_border_obj(seg_path, seg_img_list, img_num))

        if img_num == t_span:
            break
        img_num_1 = img_num
        img_num_2 = img_num + 1
        frame_overlap = compute_overlap_matrix(
            seg_path, seg_img_list, img_num_1, img_num_2)
        #print(frame_overlap)

        # -----------find false link with max_overlap relation-----
        target_idx_list = df[df['ImageNumber'] == img_num_2].index.tolist()
        for target_idx in target_idx_list:
            if df.iloc[target_idx]['Cell_TrackObjects_ParentImageNumber'] == img_num_1:
                target_o_n = int(df.iloc[target_idx]['ObjectNumber'])
                source_o_n = int(
                    df.iloc[target_idx]['Cell_TrackObjects_ParentObjectNumber'])
                #print(source_o_n)
                rel_flag = judge_mol_type(
                    frame_overlap, source_o_n, target_o_n)
                false_pair = search_false_link(
                    df,
                    relation_df,
                    frame_overlap,
                    img_num_1,
                    source_o_n,
                    img_num_2,
                    target_o_n,
                    rel_flag)

                if len(false_pair) > 0:
                    false_link.append(false_pair)

    # ----------------find split and merge------------------------------------
        area_arr = df.loc[(df['ImageNumber'] == img_num_1),
                          'Cell_AreaShape_Area'].values
        area_arr_R = df.loc[(df['ImageNumber'] == img_num_2),
                            'Cell_AreaShape_Area'].values  # area array in img2

        nb_cell_1 = frame_overlap.shape[0]
        nb_cell_2 = frame_overlap.shape[1]

        postf_cells, pref_group = cal_cell_fusion(
            frame_overlap, img_num_1, img_num_2, nb_cell_1, nb_cell_2)

        pres_cells, posts_group = cal_cell_split(
            frame_overlap, img_num_1, img_num_2, nb_cell_1, nb_cell_2)

        postfuse_cells.extend(postf_cells)
        prefuse_group.extend(pref_group)
        presplit_cells.extend(pres_cells)
        postsplit_group.extend(posts_group)

    np.save(dir_path + 'border_obj.npy', np.array(border_obj))
    np.save(dir_path + 'false_link.npy', np.array(false_link))

    # break false link and relabel traj
    df, relation_df = break_link(df, relation_df, false_link)

    df = relabel_traj(df)
    traj_start, traj_end = record_traj_start_end(df)

    # find mitosis_pairs_to_break,fuse_pairs_to_break,split_pairs_to break
    candi_am_sisters, mitosis_pairs_to_break = find_mitosis_pairs_to_break(
        relation_df, candi_am_sisters, false_link)

    postfuse_cells, prefuse_group, fuse_pairs, fuse_pairs_to_break = find_fuse_pairs_to_break(
        relation_df, postfuse_cells, prefuse_group, false_link, border_obj)

    presplit_cells, postsplit_group, split_pairs, split_pairs_to_break = find_split_pairs_to_break(
        relation_df, presplit_cells, postsplit_group, false_link, border_obj)

    for f_g in prefuse_group:
        prefuse_cells.extend(f_g)
    for s_g in postsplit_group:
        postsplit_cells.extend(s_g)

    np.save(dir_path + 'prefuse_cells.npy', np.array(prefuse_cells))
    np.save(dir_path + 'postfuse_cells.npy', np.array(postfuse_cells))
    np.save(dir_path + 'fuse_pairs.npy', np.array(fuse_pairs))
    np.save(
        dir_path +
        'fuse_pairs_to_break.npy',
        np.array(fuse_pairs_to_break))

    np.save(dir_path + 'presplit_cells.npy', np.array(presplit_cells))
    np.save(dir_path + 'postsplit_cells.npy', np.array(postsplit_cells))
    np.save(dir_path + 'split_pairs.npy', np.array(split_pairs))
    np.save(
        dir_path +
        'split_pairs_to_break.npy',
        np.array(split_pairs_to_break))

    with open(dir_path + 'prefuse_group', 'wb') as fp:
        pickle.dump(prefuse_group, fp)
    with open(dir_path + 'postsplit_group', 'wb') as fp:
        pickle.dump(postsplit_group, fp)

    # break pairs to break
    pairs_to_break = []
    pairs_to_break.extend(fuse_pairs_to_break)
    pairs_to_break.extend(split_pairs_to_break)
    pairs_to_break.extend(mitosis_pairs_to_break)
    if len(pairs_to_break) > 0:
        # there are same pairs in split_pairs_to_break and
        # mitosis_pairs_to_break
        pairs_to_break = np.unique(np.asarray(pairs_to_break), axis=0).tolist()
    print(len(pairs_to_break))

    df, relation_df = break_link(df, relation_df, pairs_to_break)
    df = relabel_traj(df)
    df.to_csv(dir_path + 'Per_Object_break.csv', index=False, encoding='utf-8')
    relation_df.to_csv(
        dir_path +
        'Per_Relationships_break.csv',
        index=False,
        encoding='utf-8')
    traj_start, traj_end = record_traj_start_end(df)

    # judge fuse,split type and find candi_mitosis
    false_traj_label = []
    candi_mitosis_label = []
    false_mitosis_obj = []
    # ------for dealing with mitosis and then fuse cells--------
    candi_mitosis_fc_label = []
    candi_mitosis_fp_label = []
    candi_mitosis_fp_group = []
    candi_mitosis_fp_group_xy = []

    mitosis_fuse_fp_label = []
    mitosis_fuse_sc_label = []
    mitosis_fuse_link_pairs = []

    for i in range(len(postfuse_cells)):
        fc_cell = postfuse_cells[i]
        fc_i_n, fc_o_n = fc_cell[0], fc_cell[1]
        fc_img = generate_single_cell_img_edt(
            img_path,
            seg_path,
            img_list,
            seg_img_list,
            obj_h,
            obj_w,
            fc_i_n,
            fc_o_n)
        fc_prob = icnn.predict(fc_img)[0]
        fc_am_flag = judge_traj_am(
            df,
            am_record,
            fc_i_n,
            fc_o_n,
            judge_later=True,
            t_range=t_search_range)

        fp_group = prefuse_group[i]
        fp_group_prob = []
        fp_group_am_flag = []
        for [fp_i_n, fp_o_n] in fp_group:
            fp_img = generate_single_cell_img_edt(
                img_path, seg_path, img_list, seg_img_list, obj_h, obj_w, fp_i_n, fp_o_n)
            fp_prob = icnn.predict(fp_img)[0]
            fp_group_prob.append(fp_prob.tolist())
            fp_am_flag = judge_traj_am(
                df,
                am_record,
                fp_i_n,
                fp_o_n,
                judge_later=False,
                t_range=t_search_range)
            fp_group_am_flag.append(fp_am_flag)

        f_label, m_fc_label, m_fp_group_label, m_fp_group, m_fp_group_xy, fc_type, fp_group_type = judge_fuse_type(
            df, am_record, fc_cell, fp_group, fc_prob, fp_group_prob, tracklet_len_thres=traj_len_thres)
        false_traj_label.extend(f_label)

        if len(m_fc_label) > 0:
            candi_mitosis_fc_label.extend(m_fc_label)
            candi_mitosis_fp_label.append(m_fp_group_label)
            candi_mitosis_fp_group.append(m_fp_group)
            candi_mitosis_fp_group_xy.append(m_fp_group_xy)

    for i in range(len(presplit_cells)):
        sp_cell = presplit_cells[i]
        sp_i_n, sp_o_n = sp_cell[0], sp_cell[1]
        sp_label = np.asscalar(df.loc[(df['ImageNumber'] == sp_i_n) & (
            df['ObjectNumber'] == sp_o_n), 'Cell_TrackObjects_Label'].values)
        sp_img = generate_single_cell_img_edt(
            img_path,
            seg_path,
            img_list,
            seg_img_list,
            obj_h,
            obj_w,
            sp_i_n,
            sp_o_n)
        sp_prob = icnn.predict(sp_img)[0]

        mitosis_fuse_flag = 0

        sc_group = postsplit_group[i]
        if sp_label in candi_mitosis_fc_label and len(sc_group) == 2:
            mitosis_fuse_flag = 1
            ind = candi_mitosis_fc_label.index(sp_label)
            mitosis_fuse_fp_label.extend(candi_mitosis_fp_label[ind])

        if mitosis_fuse_flag == 1:
            if sp_label not in false_traj_label:
                false_traj_label.append(sp_label)

            sc_group_xy = []
            for [sc_i_n, sc_o_n] in sc_group:
                sc_label = np.asscalar(df.loc[(df['ImageNumber'] == sc_i_n) & (
                    df['ObjectNumber'] == sc_o_n), 'Cell_TrackObjects_Label'].values)
                sc_group_xy.append(df.loc[(df['ImageNumber'] == sc_i_n) & (df['ObjectNumber'] == sc_o_n), [
                                   'Cell_AreaShape_Center_X', 'Cell_AreaShape_Center_Y']].values[0].tolist())
                mitosis_fuse_sc_label.append(sc_label)

            d_matrix = np.zeros((2, 2))
            for m in range(2):
                for n in range(2):
                    x1, y1 = candi_mitosis_fp_group_xy[ind][m][0], candi_mitosis_fp_group_xy[ind][m][1]
                    x2, y2 = sc_group_xy[n][0], sc_group_xy[n][1]
                    d_matrix[m, n] = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            r_ind, c_ind = linear_sum_assignment(d_matrix)
            for r, c in zip(r_ind, c_ind):
                mitosis_fuse_link_pairs.append(
                    [
                        candi_mitosis_fp_group[ind][r][0],
                        candi_mitosis_fp_group[ind][r][1],
                        sc_group[c][0],
                        sc_group[c][1]])

        else:
            sc_group_prob = []

            for [sc_i_n, sc_o_n] in sc_group:
                sc_img = generate_single_cell_img_edt(
                    img_path, seg_path, img_list, seg_img_list, obj_h, obj_w, sc_i_n, sc_o_n)
                sc_prob = icnn.predict(sc_img)[0]
                sc_group_prob.append(sc_prob.tolist())

            candi_mitosis_flag, f_label, cm_label, fm_obj, sp_type, sc_group_type = judge_split_type(
                df, am_record, sp_cell, sc_group, sp_prob, sc_group_prob, tracklet_len_thres=traj_len_thres)

            if candi_mitosis_flag == 0:
                if len(f_label) > 0:
                    false_traj_label.extend(f_label)

            else:
                candi_mitosis_label.extend(cm_label)
                if len(fm_obj) > 0:
                    false_mitosis_obj.extend(fm_obj)

    with open(dir_path + 'false_traj_label', 'wb') as fp:
        pickle.dump(false_traj_label, fp)
    with open(dir_path + 'candi_mitosis_label', 'wb') as fp:
        pickle.dump(candi_mitosis_label, fp)
    with open(dir_path + 'false_mitosis_obj', 'wb') as fp:
        pickle.dump(false_mitosis_obj, fp)

    for tlabel in false_traj_label:
        if tlabel not in candi_mitosis_label and tlabel not in mitosis_fuse_fp_label and tlabel not in mitosis_fuse_sc_label:
            df.loc[df['Cell_TrackObjects_Label'] ==
                   tlabel, 'Cell_TrackObjects_Label'] = -1

    df, relation_df = connect_link(df, relation_df, mitosis_fuse_link_pairs)
    df = relabel_traj(df)
    df.to_csv(
        dir_path +
        'Per_Object_modify.csv',
        index=False,
        encoding='utf-8')
    traj_start, traj_end = record_traj_start_end(df)



# -------------judge fuse type-----------
# two fuse types:
# 1 undersegmentation:two or more cells fuse together:
# (1) two or more normal cells
# (2) one or more mitosis cell with one or more normal cell
# (3) one or more apoptosis cell with one or more normal cell
# (4) two or more mitosis cells
# 2 oversegmentation:several pieces of a cell join together


# -------------judge split type-----------
# three split types:
# 1 undersegmentation before split: two or more cells fuse and split:
# (1) two or more normal cells
# (2) one or more mitosis cell with one or more normal cell
# (3) one or more apoptosis cell with one or more normal cell
# (4) two or more mitosis cells
# 2 oversegmentation of a cell
# 3 cell mitosis


# In[ ]:


#     np.save(dir_path+'/postfuse_cells.npy',np.array(postfuse_cells))
#     np.save(dir_path+'/presplit_cells.npy',np.array(presplit_cells))
#     with open(dir_path+'/prefuse_group', 'wb') as fp:
#         pickle.dump(prefuse_group, fp)
#     with open(dir_path+'/postsplit_group', 'wb') as fp:
#         pickle.dump(postsplit_group, fp)

#     postfuse_cells=np.load(dir_path+'/postfuse_cells.npy').tolist()
#     presplit_cells=np.load(dir_path+'/presplit_cells.npy').tolist()
#     with open (dir_path+'/prefuse_group', 'rb') as fp:
#         prefuse_group = pickle.load(fp)
#     with open (dir_path+'/postsplit_group', 'rb') as fp:
#         postsplit_group = pickle.load(fp)

#     border_obj=np.load(dir_path+'/border_obj.npy').tolist()
#     false_link=np.load(dir_path+'/false_link.npy').tolist()

#     prefuse_cells=np.load(dir_path+'/prefuse_cells.npy').tolist()
#     postfuse_cells=np.load(dir_path+'/postfuse_cells.npy').tolist()
#     fuse_pairs=np.load(dir_path+'/fuse_pairs.npy')
#     split_pairs=np.load(dir_path+'/split_pairs.npy')

#     presplit_cells=np.load(dir_path+'/presplit_cells.npy').tolist()
#     postsplit_cells=np.load(dir_path+'/postsplit_cells.npy').tolist()

#     fuse_pairs_to_break=np.load(dir_path+'/fuse_pairs_to_break.npy').tolist()
#     split_pairs_to_break=np.load(dir_path+'/split_pairs_to_break.npy').tolist()

#     with open (dir_path+'/prefuse_group', 'rb') as fp:
#         prefuse_group = pickle.load(fp)
#     with open (dir_path+'/postsplit_group', 'rb') as fp:
#         postsplit_group = pickle.load(fp)

#     with open (dir_path+'/false_traj_label', 'rb') as fp:
#         false_traj_label = pickle.load(fp)
#     with open (dir_path+'/candi_mitosis_label', 'rb') as fp:
#         candi_mitosis_label = pickle.load(fp)
#     with open (dir_path+'/candi_mitosis_sp', 'rb') as fp:
#         candi_mitosis_sp = pickle.load(fp)
#     with open (dir_path+'/candi_mitosis_sc_group', 'rb') as fp:
#         candi_mitosis_sc_group = pickle.load(fp)
#     with open (dir_path+'/candi_mitosis_stype', 'rb') as fp:
#         candi_mitosis_stype = pickle.load(fp)
#     with open (dir_path+'/candi_mitosis_scomb', 'rb') as fp:
#         candi_mitosis_scomb = pickle.load(fp)

if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    img_path = sys.argv[1]
    output_path = sys.argv[2]
    icnn_seg_wts = sys.argv[3]
    DIC_chan_label = sys.argv[4]
    traj_reconganize1(img_path, output_path, icnn_seg_wts, DIC_chan_label)
