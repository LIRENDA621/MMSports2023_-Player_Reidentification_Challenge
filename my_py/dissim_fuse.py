import pandas as pd
import os
from collections import Counter
import csv

csv_dir = 'model_fuse/result_csvs_dir' # dist_csvs used fuse
save_path = 'dissim.csv'
topk = 15 # 可调

fuse_csv = os.listdir(csv_dir)

datas = []
for file in fuse_csv:
    with open(os.path.join(csv_dir, file), 'r') as f:
        csv_reader = csv.reader(f)
        datas.append(list(csv_reader))

rows, cols, model_num = len(datas[0]), len(datas[0][0]), len(datas)

re_num = model_num // 2 + 1 # 可调

inter0_cnt = 0
cnt2 = 0
rows_fuse_weight = []
for row in range(1, rows):  # 从第二行开始，跳过第一行
    rows_data = []
    for data in datas:
        rows_data.append(list(map(float, data[row][1:])))
    
    rows_topk_indeces = []
    for row_data in rows_data:
        rows_topk_indeces.append([index for index, _ in sorted(enumerate(row_data), key=lambda x: x[1], reverse=False)[:topk]])

  
    flat_list = [element for sublist in rows_topk_indeces for element in sublist]

    # 使用 Counter 进行计数
    counter = Counter(flat_list)

    # 筛选出至少出现两次的元素
    good_idx = [element for element, count in counter.items() if count >= re_num]
    inter_num = []
    for lst in rows_topk_indeces:
        set1 = set(good_idx)
        set2 = set(lst)
        
        inter_num.append(len(set1.intersection(set2)))
    
   
        
    
    total_inter_num = sum(inter_num)

    # 正比
    fuse_weight = []
    for item in inter_num:
        if total_inter_num == 0:
            inter0_cnt += 1
            fuse_weight.append(1 / model_num)
        else:
            if item == 0:
                fuse_weight.append(0.1)
                cnt2 += 1
            else:
                fuse_weight.append(item / total_inter_num)
            
    
    # 反比
    inverse_sum = sum(1/num for num in fuse_weight)
    fuse_weight = [1/(num * inverse_sum) for num in fuse_weight]
            
    rows_fuse_weight.append(fuse_weight)

print(inter0_cnt)
print(cnt2)
result = []

for row in range(1, rows):  # 从第二行开始，跳过第一行
    weighted_sum = []

    for col in range(1, cols):  # 从第二列开始，跳过第一列
        values = [float(datas[i][row][col]) for i in range(model_num)]

        weighted_value = sum(w * v for w, v in zip(rows_fuse_weight[row-1], values))
        weighted_sum.append(round(weighted_value, 5))

    result.append(weighted_sum)

result.insert(0, datas[0][0][:])  # 第一个文件的第一行的数据（跳过第一列）
for i in range(1, len(result)):
    result[i].insert(0, datas[0][i][0])  # 后续文件的第一列的数据

with open(save_path, 'w', newline='') as result_file:
    csv_writer = csv.writer(result_file)

    for row in result:
        csv_writer.writerow(row)

