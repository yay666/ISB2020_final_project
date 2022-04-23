import numpy as np
import numpy.ma
from scipy.ndimage import distance_transform_edt
import scipy.ndimage
import scipy.sparse
from index import Indexes
import pandas as pd
from os import listdir
from skimage.io import imread
from scipy.optimize import linear_sum_assignment
import pickle
import sys

from track_module import compute_overlap_matrix, compute_overlap_pair, compute_overlap_single, generate_traj_seri, relabel_traj, record_traj_start_end, judge_mol_type, search_false_link, judge_border, find_border_obj, break_link, connect_link, false_seg_mark, judge_traj_am, judge_apoptosis_tracklet, traj_start_end_info, am_obj_info, compute_specific_overlap, compute_cost, calculate_area_penalty, find_am_sisters, cal_cell_fusion, cal_cell_split, find_mitosis_pairs_to_break, find_fuse_pairs_to_break, find_split_pairs_to_break, judge_fuse_type, judge_split_type, find_uni, get_mitotic_triple_scores, cal_size_correlation, search_wrong_mitosis

import pipe_util2


# ----parameter setting -----------
# depend on: cell type, time interval-------
# 1/2 max distance between two trajectory ends: cell type and time interval
mitosis_max_dist = 75
# size_similarity=1-abs(size1-size2)/(size1+size2) size similarity between
# two possible sister cells
simi_thres = 0.7
# for judging possible mitosis, if exceed this value, probably a single
# cell: time interval
traj_len_thres = 6
# size correlation between two sister cells, if exceed this value
# ,probably mitosis
size_corr_thres = 0.5
mature_time = 60  # time from the cell born to divide: depend on cell type and time interval


# In[13]:


# main_path='/home/zoro/Desktop/experiment_data/2018-12-21_HK2_2d_traj/'
# input_path=main_path+'/img/'
# output_path=main_path+'/output/'

def traj_reconganize2(output_path):
    dir_path = pipe_util2.folder_verify(output_path)
    seg_path = dir_path + 'seg/'
    seg_img_list = sorted(listdir(seg_path))
    df = pd.read_csv(dir_path + 'Per_Object_modify.csv')
    am_record = pd.read_csv(dir_path + 'am_record.csv')

    mother_cells = []
    daughter_cells = []
    mitosis_labels = []

    # find all traj start and traj end, calculate the possible mitosis group
    traj_start, traj_end, traj_start_xy, traj_end_xy, traj_start_area, traj_end_area = traj_start_end_info(
        df)
    F = np.column_stack(
        (traj_start_xy,
         traj_start,
         np.expand_dims(
             traj_start_area[:],
             axis=1)))
    L = np.column_stack(
        (traj_end_xy,
         traj_end,
         np.expand_dims(
             traj_end_area[:],
             axis=1)))

    mitoses, m_dist, size_simi = get_mitotic_triple_scores(
        F, L, mitosis_max_distance=mitosis_max_dist, size_simi_thres=simi_thres)
    n_mitoses = len(m_dist)

    if n_mitoses > 0:

        # sort with the mother cell's img num
        order = np.argsort(mitoses[:, 0])

        mitoses, m_dist, size_simi = mitoses[order], m_dist[order], size_simi[order]

        # -------------extract mitoses_features
        mother_am_flag = []
        mother_traj_len = []
        # mother_ff=[]#form factor
        for i_n, o_n in mitoses[:, :2]:
            obj_am_flag = 0
            if ((am_record['ImageNumber'] == i_n) & (
                    am_record['ObjectNumber'] == o_n)).any():
                obj_am_flag = np.asscalar(am_record.loc[(am_record['ImageNumber'] == i_n) & (
                    am_record['ObjectNumber'] == o_n), 'am_flag'].values)
            mother_am_flag.append(obj_am_flag)

            cur_traj_label = np.asscalar(df.loc[(df['ImageNumber'] == i_n) & (
                df['ObjectNumber'] == o_n)]['Cell_TrackObjects_Label'].values)
            traj_len = len(df.loc[df['Cell_TrackObjects_Label']
                                  == cur_traj_label, 'Cell_AreaShape_Area'].values)
            mother_traj_len.append(traj_len)
            # mother_ff.append(np.asscalar(df.loc[(df['ImageNumber']==i_n)&(df['ObjectNumber']==o_n)]['Cell_AreaShape_FormFactor'].values))

        sis1_am_flag = []
        sis1_traj_len = []
        # sis1_ff=[]
        for i_n, o_n in mitoses[:, 2:4]:
            obj_am_flag = 0
            if ((am_record['ImageNumber'] == i_n) & (
                    am_record['ObjectNumber'] == o_n)).any():
                obj_am_flag = np.asscalar(am_record.loc[(am_record['ImageNumber'] == i_n) & (
                    am_record['ObjectNumber'] == o_n), 'am_flag'].values)
            sis1_am_flag.append(obj_am_flag)
            cur_traj_label = np.asscalar(df.loc[(df['ImageNumber'] == i_n) & (
                df['ObjectNumber'] == o_n)]['Cell_TrackObjects_Label'].values)
            traj_len = len(df.loc[df['Cell_TrackObjects_Label']
                                  == cur_traj_label, 'Cell_AreaShape_Area'].values)
            sis1_traj_len.append(traj_len)
            # sis1_ff.append(np.asscalar(df.loc[(df['ImageNumber']==i_n)&(df['ObjectNumber']==o_n)]['Cell_AreaShape_FormFactor'].values))

        sis2_am_flag = []
        sis2_traj_len = []
        # sis2_ff=[]
        for i_n, o_n in mitoses[:, 4:]:
            obj_am_flag = 0
            if ((am_record['ImageNumber'] == i_n) & (
                    am_record['ObjectNumber'] == o_n)).any():
                obj_am_flag = np.asscalar(am_record.loc[(am_record['ImageNumber'] == i_n) & (
                    am_record['ObjectNumber'] == o_n), 'am_flag'].values)
            sis2_am_flag.append(obj_am_flag)
            cur_traj_label = np.asscalar(df.loc[(df['ImageNumber'] == i_n) & (
                df['ObjectNumber'] == o_n)]['Cell_TrackObjects_Label'].values)
            traj_len = len(df.loc[df['Cell_TrackObjects_Label']
                                  == cur_traj_label, 'Cell_AreaShape_Area'].values)
            sis2_traj_len.append(traj_len)
            # sis2_ff.append(np.asscalar(df.loc[(df['ImageNumber']==i_n)&(df['ObjectNumber']==o_n)]['Cell_AreaShape_FormFactor'].values))

        mother_am_flag, sis1_am_flag, sis2_am_flag = np.array(
            mother_am_flag), np.array(sis1_am_flag), np.array(sis2_am_flag)
        mother_traj_len, sis1_traj_len, sis2_traj_len = np.array(
            mother_traj_len), np.array(sis1_traj_len), np.array(sis2_traj_len)
        # mother_ff,sis1_ff,sis2_ff=np.array(mother_ff),np.array(sis1_ff),np.array(sis2_ff)
        # mitoses_features=np.column_stack((mother_am_flag,sis1_am_flag,sis2_am_flag,m_dist,size_simi,mother_traj_len,sis1_traj_len,sis2_traj_len))
        # print(mitoses_features)

        # calculate size correlation of the begining part of sister cells
        mitoses_size_corr = []
        for sis1_i_n, sis1_o_n, sis2_i_n, sis2_o_n in mitoses[:, 2:]:
            size_corr = cal_size_correlation(
                df,
                sis1_i_n,
                sis1_o_n,
                sis2_o_n,
                high_traj_len_thres=4,
                low_traj_len_thres=3)
            mitoses_size_corr.append(size_corr)
        mitoses_size_corr = np.array(mitoses_size_corr)

        # based on mitosis features,find true mitosis

        mask = np.zeros((mitoses.shape[0],), dtype=bool)
        for i in range(mitoses.shape[0]):
            flag = 0
            if (
                mother_am_flag[i] == 2 and (
                    sis1_am_flag[i] > 0 or sis2_am_flag[i] > 0)) or (
                mother_am_flag[i] == 2 and sis1_traj_len[i] > traj_len_thres and sis2_traj_len[i] > traj_len_thres) or (
                sis1_am_flag[i] == 2 and sis1_am_flag[i] == 2 and sis1_traj_len[i] > traj_len_thres and sis2_traj_len[i] > traj_len_thres) or (
                    mother_traj_len[i] > traj_len_thres and sis1_traj_len[i] > traj_len_thres and sis2_traj_len[i] > traj_len_thres) or mitoses_size_corr[i] > size_corr_thres:
                flag = 1
                mother_cells.append([mitoses[i][0], mitoses[i][1]])
                mitosis_labels.append(np.asscalar(df.loc[(df['ImageNumber'] == mitoses[i][0]) & (
                    df['ObjectNumber'] == mitoses[i][1]), 'Cell_TrackObjects_Label'].values))
                daughter_cells.append([mitoses[i][2], mitoses[i][3]])
                mitosis_labels.append(np.asscalar(df.loc[(df['ImageNumber'] == mitoses[i][2]) & (
                    df['ObjectNumber'] == mitoses[i][3]), 'Cell_TrackObjects_Label'].values))
                daughter_cells.append([mitoses[i][4], mitoses[i][5]])
                mitosis_labels.append(np.asscalar(df.loc[(df['ImageNumber'] == mitoses[i][4]) & (
                    df['ObjectNumber'] == mitoses[i][5]), 'Cell_TrackObjects_Label'].values))
                mask[i] = True
        mitoses = mitoses[mask]
    # ------------mitosis record-------------------
    mitosis_record = pd.DataFrame(
        columns=[
            'mother_traj_label',
            'mother_i_n',
            'mother_o_n',
            'sis1_traj_label',
            'sis1_i_n',
            'sis1_o_n',
            'sis2_traj_label',
            'sis2_i_n',
            'sis2_o_n'])
    for i in range(mitoses.shape[0]):
        mitosis_count = i + 1

        mother_i_n = mitoses[i][0]
        mother_o_n = mitoses[i][1]
        mother_traj_label = np.asscalar(df.loc[(df['ImageNumber'] == mother_i_n) & (
            df['ObjectNumber'] == mother_o_n), 'Cell_TrackObjects_Label'].values)
        sis1_i_n = mitoses[i][2]
        sis1_o_n = mitoses[i][3]
        sis1_traj_label = np.asscalar(df.loc[(df['ImageNumber'] == sis1_i_n) & (
            df['ObjectNumber'] == sis1_o_n), 'Cell_TrackObjects_Label'].values)
        sis2_i_n = mitoses[i][4]
        sis2_o_n = mitoses[i][5]
        sis2_traj_label = np.asscalar(df.loc[(df['ImageNumber'] == sis2_i_n) & (
            df['ObjectNumber'] == sis2_o_n), 'Cell_TrackObjects_Label'].values)

        mitosis_record.loc[mitosis_count] = [
            mother_traj_label,
            mother_i_n,
            mother_o_n,
            sis1_traj_label,
            sis1_i_n,
            sis1_o_n,
            sis2_traj_label,
            sis2_i_n,
            sis2_o_n]
#     print(mitosis_record)

    mitosis_record = search_wrong_mitosis(mitosis_record, mature_time)
    mitoses = mitosis_record[['mother_i_n',
                              'mother_o_n',
                              'sis1_i_n',
                              'sis1_o_n',
                              'sis2_i_n',
                              'sis2_o_n']].values
    mitosis_labels = mitosis_record[[
        'mother_traj_label', 'sis1_traj_label', 'sis2_traj_label']].values.flatten().tolist()
    mother_cells = mitosis_record[['mother_i_n', 'mother_o_n']].values.tolist()

    daughter_cells = []
    daughter_cells.extend(
        mitosis_record[['sis1_i_n', 'sis1_o_n']].values.tolist())
    daughter_cells.extend(
        mitosis_record[['sis2_i_n', 'sis2_o_n']].values.tolist())
    print(mitoses.shape)

    np.save(dir_path + 'mitoses.npy', mitoses)
    with open(dir_path + 'mother_cells', 'wb') as fp:
        pickle.dump(mother_cells, fp)
    with open(dir_path + 'daughter_cells', 'wb') as fp:
        pickle.dump(daughter_cells, fp)
    with open(dir_path + 'mitosis_labels', 'wb') as fp:
        pickle.dump(mitosis_labels, fp)

    with open(dir_path + 'false_mitosis_obj', 'rb') as fp:
        false_mitosis_obj = pickle.load(fp)

    # remove the false segmentation in candi_mitosis that are not real mitosis
    false_traj_label = []
    for i in range(len(false_mitosis_obj)):
        fm_i_n, fm_o_n = false_mitosis_obj[i][0], false_mitosis_obj[i][1]
        fm_label = np.asscalar(df.loc[(df['ImageNumber'] == fm_i_n) & (
            df['ObjectNumber'] == fm_o_n), 'Cell_TrackObjects_Label'].values)
        if fm_label not in mitosis_labels:
            false_traj_label.append(fm_label)

    for tlabel in false_traj_label:
        df.loc[df['Cell_TrackObjects_Label'] ==
               tlabel, 'Cell_TrackObjects_Label'] = -1
    # relabel df and save
    df = relabel_traj(df)
    df.to_csv(
        dir_path +
        'Per_Object_mitosis.csv',
        index=False,
        encoding='utf-8')
    traj_start, traj_end = record_traj_start_end(df)

if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    output_path = sys.argv[1]
    traj_reconganize2(output_path)

