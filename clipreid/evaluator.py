"""
Source: https://github.com/DeepSportRadar/player-reidentification-challenge

for: pairwise_distance, compute_scores, write_mat_csv

"""

from collections import OrderedDict
import torch
from .metrics import cmc, mean_ap
from tqdm import tqdm
from .rerank import re_ranking
import numpy as np
import time
from torch.nn import functional as F
import pickle

def predict(model,
            dataloader,
            device,
            normalize_features=False,
            verbose=True):
    
    # wait a second bevor starting progress bar
    time.sleep(1)

    model.eval()

    if verbose:
        bar = tqdm(dataloader,
                   total=len(dataloader),
                   ascii=True,
                   bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                   desc="Test ")
    else:
        bar = dataloader
        
    features = OrderedDict()

    file_names = []
    players = []
    img_types = []

    for img, file_name, player, img_type in bar:
        
        img = img.to(device)
        
        file_names.extend(file_name)
        players.extend(player)
        img_types.extend(img_type)
        
        with torch.no_grad():
            output = model(img)
            # output, _ = model(img)
            if normalize_features:
                output = F.normalize(output, p=2, dim=1)
            output = output.cpu()
            
        for i in range(len(output)):
            features[file_name[i]] = output[i]
        
    if verbose:
        bar.close()
    
    return features


def pairwise_distance(features,
                      query=None,
                      gallery=None):
    
    
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist = dist.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist

    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    
    # ------------------------------
    # with open('ckp/query_fea.pkl', 'wb') as f:
    #     pickle.dump(x, f)
    # with open('ckp/gallery_fea.pkl', 'wb') as f:
    #     pickle.dump(y, f)
    tmp = torch.cat([features[key].unsqueeze(0) for key in features], 0)
    with open('ckp/test_fea.pkl', 'wb') as f:
        pickle.dump(tmp, f)
    
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)

    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    
    return dist


def compute_dist_matrix(features_dict,
                        query,
                        gallery,
                        rerank=False,
                        k1=20,
                        k2=6,
                        lambda_value=0.3):
    

    dist_matrix = pairwise_distance(features_dict, query, gallery)
    
    if rerank:
        distmat_qq = pairwise_distance(features_dict, query, query)
        distmat_gg = pairwise_distance(features_dict, gallery, gallery)
        dist_matrix_rerank = re_ranking(dist_matrix, distmat_qq, distmat_gg, k1=k1, k2=k2, lambda_value=lambda_value)
        return dist_matrix.numpy(), dist_matrix_rerank
    else:
        return dist_matrix

def postprocess_distance(features, query=None, gallery=None, k1=20, k2=6, lamda=0.5, expan_thr=2/3, metric=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist = dist.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist
    # 根据dataset.query和dataset.gallery里的顺序，分别把query和gallery的特征cat在一起
    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    # 把query和gallery的特征cat在一起  (query_num + gallery_num, feat_dim)
    z = torch.cat([x, y], dim=0)

    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)


    # 计算query_gallery和query_gallery之间的欧氏距离
    dist_orig = pairwise_distance(features, query, gallery)
     
    #  # -------------对齐22top1
    # dist_orig = torch.pow(dist_orig, 2).float()
    # dist_orig = dist_orig / torch.max(dist_orig, dim=0)[0]
    # # dist_orig = dist_orig.t()


    # 计算query_gallery和query_gallery之间的欧氏距离
    t = z.size(0)
    z = z.view(t, -1)
    dist_orig_qg = torch.pow(z, 2).sum(dim=1, keepdim=True).expand(t, t) + \
           torch.pow(z, 2).sum(dim=1, keepdim=True).expand(t, t).t()
    dist_orig_qg.addmm_(z, z.t(), beta=1, alpha=-2) # 维度(query_num, gallery_num)

    # # -------------对齐22top1
    # dist_orig_qg = torch.pow(dist_orig_qg, 2).float()
    # dist_orig_qg = dist_orig_qg / torch.max(dist_orig_qg, dim=0)[0]
    # # dist_orig_qg = dist_orig_qg.t()

    # 计算R(p, k1)
    R_pk1_list = []

    _, indices_qg = torch.topk(dist_orig_qg[:m, :], k1, dim=1, largest=False) # (50, 20)
    _, indices_gq = torch.topk(dist_orig_qg[m:, :], k1, dim=1, largest=False) # (910, 20)

    for q_idx in range(m):
        tmp_list = []
        qg_top_k = indices_qg[q_idx] # qg_top_k中元素取值范围[0, 959]
        for g_idx in qg_top_k:
            # 如果是query则跳过，因为query不应该在R(p, k1)中
            if g_idx < m:
                continue

            gq_topk = indices_gq[g_idx - m, :] # gq_topk中元素取值范围[0, 959]
            has_element = torch.any(torch.eq(gq_topk, q_idx))
            if has_element:
                tmp_list.append(g_idx - m)
        R_pk1_list.append(tmp_list) # 这里R_pk1_list中的gallery索引取值范围是[0, 909]
    

    # 计算R(g, k1)
    R_gk1_list = []

    _, indices_gg = torch.topk(dist_orig_qg[m:, m:], k1, dim=1, largest=False) # (910, 20)
    # _, indices_gg_inv = torch.topk(dist_orig_qg[m:, m:], k1, dim=0, largest=False) # (20, 910)

    for g_idx in range(n):
        tmp_list = []
        gg_top_k = indices_gg[g_idx] # gg_top_k中元素取值范围[0, 909]
        for gg_idx in gg_top_k:
            # gg_topk_inv = indices_gg_inv[:, gg_idx] # gg_topk_inv中元素取值范围[0, 909]
            gg_topk_inv = indices_gg[gg_idx, :] # gg_topk_inv中元素取值范围[0, 909]
            has_element = torch.any(torch.eq(gg_topk_inv, g_idx))
            if has_element:
                tmp_list.append(gg_idx)
        R_gk1_list.append(tmp_list) # 这里R_gk1_list中的gallery索引取值范围是[0, 909]


    # R(p, k1) -> R*(p, k1)
    R_gk1_list_2 = []
    indices_gg_2 = indices_gg[:, :k1 // 2]
    for g_idx in range(n):
        tmp_list = []
        gg_top_k = indices_gg_2[g_idx] # gg_top_k中元素取值范围[0, 909]
        for gg_idx in gg_top_k:
            # gg_topk_inv = indices_gg_inv[:, gg_idx] # gg_topk_inv中元素取值范围[0, 909]
            gg_topk_inv = indices_gg_2[gg_idx, :] # gg_topk_inv中元素取值范围[0, 909]
            has_element = torch.any(torch.eq(gg_topk_inv, g_idx))
            if has_element:
                tmp_list.append(gg_idx)
        R_gk1_list_2.append(tmp_list) # 这里R_gk1_list_2中的gallery索引取值范围是[0, 909]

    cnt1 = 0
    for i, pk_list in enumerate(R_pk1_list):
        pk_array = np.array([item.item() for item in pk_list])
        for g_idx in pk_list:
            gk_array = np.array([item.item() for item in R_gk1_list_2[g_idx]])
            intersection_count = len(np.intersect1d(pk_array, gk_array))
            # if intersection_count >= len(gk_array) * 2 // 3:
            if intersection_count >= len(gk_array) * expan_thr:
                R_pk1_list[i] += R_gk1_list_2[g_idx]
                # 去重
                unique_tensors = torch.unique(torch.stack(R_pk1_list[i]))
                R_pk1_list[i] = [t for t in unique_tensors]
                cnt1 += 1
    print('生成R*(p, k)时, 成功合并的数量为{}'.format(cnt1))

            
    # R(g, k1) -> R*(g, k1)
    cnt2 = 0
    for i, gk_list in enumerate(R_gk1_list):
        gk_array = np.array([item.item() for item in gk_list])[1:] # 除去自己
        for g_idx in gk_array:
            gk_array_2 = np.array([item.item() for item in R_gk1_list_2[g_idx]])
            intersection_count = len(np.intersect1d(gk_array, gk_array_2))
            # if intersection_count >= len(gk_array_2) * 2 // 3:
            if intersection_count >= len(gk_array_2) * expan_thr:

                R_gk1_list[i] += R_gk1_list_2[g_idx]
                # 去重
                unique_tensors = torch.unique(torch.stack(R_gk1_list[i]))
                R_gk1_list[i] = [t for t in unique_tensors]
                cnt2 += 1
    print('生成R*(g, k)时, 成功合并的数量为{}'.format(cnt2))


    # 计算每一个query和gallery的k-reciprocal feature
    query_k_re_fea = torch.zeros((m, n))
    gallery_k_re_fea = torch.zeros((n, n))

    for i, item in enumerate(R_pk1_list):
        for g_idx in item:
            query_k_re_fea[i, g_idx] = 1
            # smooth--------------------(弃用)
            # query_k_re_fea[i, g_idx] = torch.exp(-dist_orig[i, g_idx])
    
    for i, item in enumerate(R_gk1_list):
        for g_idx in item:
            gallery_k_re_fea[i, g_idx] = 1
            # smooth--------------------(弃用)
            # gallery_k_re_fea[i, g_idx] = torch.exp(-dist_orig_qg[i + m, g_idx + m])
    

    ## Local Query Expansion
    # 对query的特征进行扩展
    for i, q_fea in enumerate(query_k_re_fea):
        tmp_tensor = torch.zeros(n)
        fusion_idxs = indices_qg[i, :k2]
        for fusion_idx in fusion_idxs:
            if fusion_idx < m: # query
                tmp_tensor += query_k_re_fea[fusion_idx]
            else:
                tmp_tensor += gallery_k_re_fea[fusion_idx - m]
        query_k_re_fea[i] = tmp_tensor / len(fusion_idxs)
    
    # 对gallery的特征进行扩展
    for i, g_fea in enumerate(gallery_k_re_fea):
        tmp_tensor = torch.zeros(n)
        fusion_idxs = indices_gg[i, 1 : k2 + 1] # 排除自身
        for fusion_idx in fusion_idxs:
            # if fusion_idx < m: # query
            #     tmp_tensor += query_k_re_fea[fusion_idx]
            # else:
            #     tmp_tensor += gallery_k_re_fea[fusion_idx - m]
            tmp_tensor += gallery_k_re_fea[fusion_idx]
            
        gallery_k_re_fea[i] = tmp_tensor / len(fusion_idxs)
    
    
    
    
    # 计算Jaccard distance
    cnt3 = 0
    jaccard_dist = torch.zeros((m, n))
    for q_idx in range(m):
        q_fea = query_k_re_fea[q_idx]
        for g_idx in range(n):
            g_fea = gallery_k_re_fea[g_idx]
            inter = torch.sum(torch.min(q_fea, g_fea))
            if inter > 0.0:
                cnt3 += 1
            union = torch.sum(torch.max(q_fea, g_fea))
            jaccard_dist[q_idx, g_idx] = 1 - inter / union
    
    print('number of (inter != 0) = {}'.format(cnt3))

    # # linear map
    # _, indices_orig_qg = torch.topk(dist_orig, 20, dim=1, largest=False) # (50, 20)
    # _, indices_orig_gq = torch.topk(dist_orig.T, 20, dim=1, largest=False) # (910, 20)
    # # print(indices_orig_gq)
    # # mean = torch.mean(dist_orig, dim=0)
    # # variance = torch.var(dist_orig, dim=0)
    # # mean1 = torch.mean(jaccard_dist, dim=0)
    # # variance1 = torch.var(jaccard_dist, dim=0)
    # # print("均值：", mean)
    # # print("方差：", variance)
    # # print("均值：", mean1)
    # # print("方差：", variance1)
    # cnt4 = 0
    # cnt5 = 0
    # cnt6 = 0
    # # top_k = 1
    # for g_idx, q_idxs in enumerate(indices_orig_gq):
    #     # for top_idx, q_idx in enumerate(q_idxs[: top_k]):
    #     q_idx = q_idxs[0] # 与这个gallery最相似的query的索引
    #     if indices_orig_qg[q_idx][0] != g_idx: # 如果这个query的最相似gallery不是这个gallery
    #         top1_g_idx = indices_orig_qg[q_idx][0]  # 与这个query的最相似的gallery索引
    #         cnt4 += 1
    #         if indices_orig_gq[top1_g_idx][0] != q_idx: # 如果与这个query的最相似的gallery的最相似query不是这个query
    #             min_dist = dist_orig[q_idx][top1_g_idx]
    #             scale_factor = math.ceil(dist_orig[q_idx][g_idx] / min_dist) 
    #             dist_orig[q_idx][g_idx] /= scale_factor
    #             cnt5 += 1
    #         # if g_idx not in indices_orig_qg[q_idx][: 19]: # 如果这个gallery都不在与它最相似的query的前10相似里, 那么就把这个gallery和query的距离改为第19
    #         #     min_dist = dist_orig[q_idx][18]
    #         #     scale_factor = math.ceil(dist_orig[q_idx][g_idx] / min_dist) 
    #         #     dist_orig[q_idx][g_idx] /= scale_factor
    #         #     cnt6 += 1
    # print("gallery_A的最相似query_A的最相似gallery_B不是这个gallery_A的数量: {}".format(cnt4))
    # print("gallery_A的最相似query_A的最相似gallery_B不是这个gallery_A, 且这个query_A的最相似gallery_B的最相似query_B不是这个query_A的数量: {}".format(cnt5))
    # # print("gallery_A的最相似query_A的最相似gallery_B不是这个gallery_A, 且gallery_A在query_A的相似度矩阵中排不到前5的数量: {}".format(cnt6))
    
    normalized_dist_orig = F.normalize(dist_orig, p=2)
    
    dist = (1 - lamda) * normalized_dist_orig + lamda * jaccard_dist
    # dist = (1 - lamda) * dist_orig + lamda * jaccard_dist

    return dist

def compute_scores(distmat,
                   query=None,
                   gallery=None,
                   query_ids=None,
                   gallery_ids=None,
                   query_cams=None,
                   gallery_cams=None,
                   cmc_topk=(1, 5, 10),
                   cmc_scores=True):
    
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('mAP: {:4.2%}'.format(mAP))
    
    
    if cmc_scores:
        # Compute all kinds of CMC scores
        cmc_configs = {
            'allshots': dict(separate_camera_set=False,
                             single_gallery_shot=False,
                             first_match_break=False),
            'cuhk03': dict(separate_camera_set=True,
                           single_gallery_shot=True,
                           first_match_break=False),
            'market1501': dict(separate_camera_set=False,
                               single_gallery_shot=False,
                               first_match_break=True)}
        cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                                query_cams, gallery_cams, **params)
                      for name, params in cmc_configs.items()}
    
        print('CMC Scores{:>12}{:>12}{:>12}'
              .format('allshots', 'cuhk03', 'market1501'))
        for k in cmc_topk:
            print('  top-{:<4}{:12.2%}{:12.2%}{:12.2%}'
                  .format(k,
                          cmc_scores['allshots'][k - 1],
                          cmc_scores['cuhk03'][k - 1],
                          cmc_scores['market1501'][k - 1]))
            
    return mAP


def write_mat_csv(fpat, dist_matrix, query, gallery):
    gallery_order_list = [pid for _, pid, _ in gallery]
    query_order_list = [pid for _, pid, _ in query]
    data = np.array([0, *gallery_order_list])
    rows = np.array(query_order_list)[:, np.newaxis]
    with open(fpat, 'w') as f:
        np.savetxt(f, data.reshape(1, data.shape[0]), delimiter=',', fmt='%i')
        np.savetxt(f, np.hstack((rows, dist_matrix)), newline='\n', fmt=['%i',
                   *['%10.5f']*dist_matrix.shape[1]], delimiter=',')



