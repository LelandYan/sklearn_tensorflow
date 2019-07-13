import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import dok_matrix

# aupr_w = np.loadtxt("only_change_w/CMLDR_precisions.txt")
# aupr_cost = np.loadtxt("add_two_items_to_cost_function/CMLDR_precisions.txt")
# k_mat = k_Mat = [5, 10, 20, 30, 40, 50]
# for i in np.arange(6):
#     plt.plot(aupr_w[:,i],'r',label="precision_w")
#     plt.plot(aupr_cost[:,i],'b',label="precision_cost")
#     plt.legend()
#     plt.title(f"k={k_mat[i]},precisions_vs")
#     plt.savefig(f"k={k_mat[i]},precisions_vs.png")
#     plt.show()


train = dok_matrix((4,5))
train[1,3] = 1
train[2,3] = 1
train[0,3] = 1
tmp = train.tocoo()
print(train.toarray())
print(train[0])
print(list(tmp.col.reshape(-1)))
print(list(tmp.row.reshape(-1)))


