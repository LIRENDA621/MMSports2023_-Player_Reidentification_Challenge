import pickle
import os

q_path = '/home/data1/lrd/mmsport/2022-winners-player-reidentification-challenge-master/data_reid/reid_challenge/query/'
g_path = '/home/data1/lrd/mmsport/2022-winners-player-reidentification-challenge-master/data_reid/reid_challenge/gallery/'

q_list = os.listdir(q_path)
q_list = sorted(q_list, key=lambda x: int(x.split('.')[0]))

q_list_new = []
for i, item in enumerate(q_list):
    q_list_new.append((q_path + item, 0))

g_list = os.listdir(g_path)
g_list = sorted(g_list, key=lambda x: int(x.split('.')[0]))

g_list_new = []
for i, item in enumerate(g_list):
    g_list_new.append((g_path + item, 0))

with open('pkl/challenge_query.pkl', 'wb') as f:
    pickle.dump(q_list_new, f)

with open('pkl/challenge_gallery.pkl', 'wb') as f:
    pickle.dump(g_list_new, f)