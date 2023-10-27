import pandas as pd
import os

res1 = 'model_fuse/stage2sub/1.csv'
res2 = 'model_fuse/stage2sub/2.csv'
res3 = 'model_fuse/stage2sub/3.csv'
res4 = 'model_fuse/stage2sub/4.csv'

save_root = 'model_fuse/stage2sub'
# fuse_model = 'paperuse2_1_avg.csv'
fuse_model = '5.csv'


# 读取两个CSV文件
df1 = pd.read_csv(res1)
df2 = pd.read_csv(res2)
df3 = pd.read_csv(res3)
df4 = pd.read_csv(res4)

result = (df1 + df2 + df3 + df4) / 4.0
result = result.round(5)

result.to_csv(os.path.join(save_root, fuse_model), index=False)