#!/usr/bin/env python
# coding: utf-8

# In[3]:


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
from track_module import compute_overlap_matrix, compute_overlap_pair, compute_overlap_single, generate_traj_seri, relabel_traj, record_traj_start_end, judge_mol_type, search_false_link, judge_border, find_border_obj, break_link, connect_link, false_seg_mark, judge_traj_am, judge_apoptosis_tracklet, traj_start_end_info, am_obj_info, compute_specific_overlap, compute_cost, calculate_area_penalty, find_am_sisters, cal_cell_fusion, cal_cell_split, find_mitosis_pairs_to_break, find_fuse_pairs_to_break, find_split_pairs_to_break, judge_fuse_type, judge_split_type, find_uni, get_mitotic_triple_scores, traj_start_end_info, get_gap_pair_scores
import pipe_util2
import sys

# ----parameter setting -----------
# depend on: cell type, time interval-------
# the maximum allowed # of frames between the traj end and traj start:
# time interval
max_frame_difference = 5
# distance*proporation between large cell area/small cell area of traj end
# and start : time interval
max_gap_score = 100


# In[18]:'

# main_path='/home/zoro/Desktop/experiment_data/2019-02-22_HK2_3d/'
# main_path = '/mnt/data0/Ke/weikang_exp_process/2019-05-05_A549_vim/'
# # input_path=main_path+'/img/'
# input_path=main_path+'a549_tif/vimentin/' # 'img/'
# output_path=main_path+'output/'


def traj_reconganize3(output_path):

    output_path = pipe_util2.folder_verify(output_path)
    dir_path = output_path
    seg_path = pipe_util2.folder_verify(output_path + 'seg')
    seg_img_list = sorted(listdir(seg_path))
    df = pd.read_csv(dir_path + 'Per_Object_mitosis.csv')
    am_record = pd.read_csv(dir_path + 'am_record' + '.csv')
    relation_df = pd.read_csv(dir_path + 'Per_Relationships_break.csv')

    with open(dir_path + '/mother_cells', 'rb') as fp:
        mother_cells = pickle.load(fp)
    with open(dir_path + '/daughter_cells', 'rb') as fp:
        daughter_cells = pickle.load(fp)
    mitoses = np.load(dir_path + '/mitoses.npy')
    mother_cells = np.array(mother_cells)
    daughter_cells = np.array(daughter_cells)

    # find all traj start and end
    traj_start, traj_end, traj_start_xy, traj_end_xy, traj_start_area, traj_end_area = traj_start_end_info(
        df)
    # find those traj start or end that are not belong to mitosis
    if len(daughter_cells) > 0:
        ts_mask = np.all(
            np.any((traj_start - daughter_cells[:, None]), axis=2), axis=0)

    else:
        ts_mask = np.ones((traj_start.shape[0],), dtype=bool)

    if len(mother_cells) > 0:
        te_mask = np.all(
            np.any((traj_end - mother_cells[:, None]), axis=2), axis=0)
    else:
        te_mask = np.ones((traj_end.shape[0],), dtype=bool)

    mitosis_ts_ind = np.where(~ts_mask)
    mitosis_te_ind = np.where(~te_mask)

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

    # close gap between traj start and traj end

    link_obj_pairs = []

    a, gap_scores = get_gap_pair_scores(
        F, L, max_frame_difference)  # a is index pairs of L and F

    # filter by max gap score
    mask = gap_scores <= max_gap_score
    if np.sum(mask) > 0:
        a, gap_scores = a[mask], gap_scores[mask]
        end_nodes = []
        start_nodes = []
        scores = []
        end_nodes.append(a[:, 0])
        start_nodes.append(a[:, 1])
        scores.append(gap_scores)

        i = np.hstack(end_nodes)

        j = np.hstack(start_nodes)

        c = np.hstack(scores)

        score_matrix = scipy.sparse.coo_matrix(
            (c, (i, j)), shape=(L.shape[0], F.shape[0])).tocsr()

        score_matrix = score_matrix.toarray()
        score_matrix[score_matrix == 0] = 1e100
        # set the score_matrix element to 1e100 when row index or column index
        # belong to mitosis
        score_matrix[mitosis_te_ind, :] = 1e100
        score_matrix[:, mitosis_ts_ind] = 1e100

        row_ind, col_ind = linear_sum_assignment(score_matrix)
        for r, c in zip(row_ind, col_ind):
            if score_matrix[r, c] != 1e100:
                link_obj_pairs.append([L[r, 2], L[r, 3], F[c, 2], F[c, 3]])

    np.save(dir_path + 'link_obj_pairs.npy', np.array(link_obj_pairs))

    # remove all border object and relabel all traj
    border_obj = np.load(dir_path + 'border_obj.npy').tolist()

    df, relation_df = connect_link(df, relation_df, link_obj_pairs)
    df = false_seg_mark(df, border_obj)
    df = relabel_traj(df)
    df.to_csv(
        dir_path +
        'Per_Object_relink' + '.csv',
        index=False,
        encoding='utf-8')
    traj_seri_self, traj_idx_seri_self = generate_traj_seri(df)
    traj_start, traj_end = record_traj_start_end(df)
    traj_label = df['Cell_TrackObjects_Label'].values
    traj_label = np.unique(traj_label[traj_label > 0])

    traj_seri_self.insert(loc=0, column='traj_label', value=traj_label)
    traj_seri_self.to_csv(
        dir_path +
        'traj_object_num' + '.csv',
        index=False,
        encoding='utf-8')

    # ------------mitosis record because the traj labels probably change------
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
    mitosis_record.to_csv(
        dir_path +
        'mitosis_record' + '.csv',
        index=False,
        encoding='utf-8')

if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    output_path = sys.argv[1]
    traj_reconganize3(output_path)

