import pickle
import os

q_path = 'data_reid/reid_test/query'
g_path = 'data_reid/reid_test/gallery'

q_list = os.listdir(q_path)
q_list = sorted(q_list, key=lambda x: (int(x.split('.')[0].split('_')[0]), int(x.split('.')[0].split('_')[2])))

q_list_new = []
for i, item in enumerate(q_list):
    q_list_new.append(('data_reid/reid_test/query/' + item, 0))

g_list = os.listdir(g_path)
g_list = sorted(g_list, key=lambda x: (int(x.split('.')[0].split('_')[0]), int(x.split('.')[0].split('_')[2])))

g_list_new = []
for i, item in enumerate(g_list):
    g_list_new.append(('data_reid/reid_test/gallery/' + item, 0))

with open('pkl/test_query.pkl', 'wb') as f:
    pickle.dump(q_list_new, f)

with open('pkl/test_gallery.pkl', 'wb') as f:
    pickle.dump(g_list_new, f)