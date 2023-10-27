"""
Source: https://github.com/zhunzhong07/person-re-ranking

Created on Mon Jun 26 14:46:56 2017
@author: luohao
Modified by Houjing Huang, 2017-12-22.
- This version accepts distance matrix instead of raw features.
- The difference of `/` division between python 2 and 3 is handled.
- numpy.float16 is replaced by numpy.float32 for numerical precision.

CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking

API
q_g_dist: query-gallery distance matrix, numpy array, shape [num_query, num_gallery]
q_q_dist: query-query distance matrix, numpy array, shape [num_query, num_query]
g_g_dist: gallery-gallery distance matrix, numpy array, shape [num_gallery, num_gallery]
k1, k2, lambda_value: parameters, the original paper is (k1=20, k2=6, lambda_value=0.3)
Returns:
  final_dist: re-ranked distance, numpy array, shape [num_query, num_gallery]
"""
from __future__ import division, print_function, absolute_import
import numpy as np

__all__ = ['re_ranking']


def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.7):

    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.
    # q_g_dist(50, 910) q_q_dist(50, 50), g_g_dist(910, 910)
    original_dist = np.concatenate(
        [
            np.concatenate([q_q_dist, q_g_dist], axis=1), # (50, 50 + 910)
            np.concatenate([q_g_dist.T, g_g_dist], axis=1) # (910, 50 + 910)
        ],
        axis=0 # (50 + 910, 50 + 910)
    )
    original_dist = np.power(original_dist, 2).astype(np.float32) # 对original_dist平方
    original_dist = np.transpose(
        1. * original_dist / np.max(original_dist, axis=0) # 每一个query和gallery的960个距离都除以这960个里面的最大值
    )
    V = np.zeros_like(original_dist).astype(np.float32)
    initial_rank = np.argsort(original_dist).astype(np.int32) # (960, 960)

    query_num = q_g_dist.shape[0]
    gallery_num = q_g_dist.shape[0] + q_g_dist.shape[1]
    all_num = gallery_num

    for i in range(all_num): # 遍历每一个query和gallery
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1] # 找到对第i个图片前k1+1相似的索引 (k1+1, )
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1] # 再找到forward_k_neigh_index里图片前k+1相似的索引 (k1+1, k1+1)
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi] # 找到与i互为前k1+1最相似的索引
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)): # 遍历i的每个k_reciprocal_index
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[ 
                candidate, :int(np.around(k1 / 2.)) + 1]
            candidate_backward_k_neigh_index = initial_rank[
                candidate_forward_k_neigh_index, :int(np.around(k1 / 2.)) + 1]
            fi_candidate = np.where(
                candidate_backward_k_neigh_index == candidate
            )[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[
                fi_candidate]   # 61-69和前面一样
            if len(
                np.
                intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)
            ) > 2. / 3 * len(candidate_k_reciprocal_index): # 如果j的k_reciprocal_index(k1 / 2 + 1)与i的交集大于j的2/3
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index, candidate_k_reciprocal_index
                )

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index) # 去重
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index]) 
        V[i, k_reciprocal_expansion_index] = 1. * weight / np.sum(weight) # 79、80是论文里的smooth 但他还归一化了
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe # 82-86 对所有qg作了expansion
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num): # 对于第i个图片，把所有和它匹配的图片索引加进来
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float32)
        indNonZero = np.where(V[i, :] != 0)[0] # 返回的是列索引, 与query_i所有匹配的索引
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero] # 与query_i匹配的图片的各自匹配的图片
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(
                V[i, indNonZero[j]], V[indImages[j], indNonZero[j]] # query_i与j在V中的结果，与j所有匹配的图片与j在V中的结果
            )
        jaccard_dist[i] = 1 - temp_min / (2.-temp_min) # temp_min <= 1

    final_dist = jaccard_dist * lambda_value + original_dist*(1-lambda_value)
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist
