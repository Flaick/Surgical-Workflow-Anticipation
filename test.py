import pickle
import numpy as np
import matplotlib.pyplot as plt

root_path = '/home/kun/Desktop/miccai21/TeCNO/Videos/dataframes/cholec_split_250px_25fps.pkl'

df = open(root_path,'rb')
data = pickle.load(df)
df.close()

print(data.keys())
print(data['video_idx'][:1000])
# pre = np.argmax(data[1], 1)
# gt = data[2]
#
# print(pre.shape[0])
# x = np.arange(pre.shape[0])
# l1 = plt.plot(x, pre, 'r--', label='pred')
# l2 = plt.plot(x, gt, 'b--', label='gt')
#
# plt.xlabel('row')
# plt.ylabel('column')
# plt.legend()
# # plt.savefig('./visual_phase/' + str(ID) + '.png')
# plt.show()
# exit()

