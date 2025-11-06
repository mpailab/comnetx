import sys,os
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FLMIG_root = os.path.join(PROJECT_PATH, "baselines", "FLMIG_algorithm", "FLMIG_algorithm")
if FLMIG_root not in sys.path:
    sys.path.insert(0, FLMIG_root)

import numpy as np

# PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# Data_root = os.path.join(PROJECT_PATH, "test", "graphs", "small")

# A = np.load(Data_root + "/dblp/dblp_adj.npy")

# # Для неориентированного графа
# edges = []
# n = A.shape[0]
# for i in range(n):
#     for j in range(i+1, n): # Для неориентированного графа 
#         if A[i, j] != 0:     
#             edges.append((i, j))

# with open(Data_root + "/dblp/dblp.txt", "w") as f:
#     for u, v in edges:
#         f.write(f"{u} {v}\n")

# print("dblp.txt создан для нового алгоритма")

from FLMIG import de_main

if __name__ == '__main__':

   
    if  sys.argv[4] != 'None':
        NMI_max,Q_Max,Q_avg ,Q_std,time_run = de_main()
        print("NMI_max",NMI_max)
        print("the value of Q_max",Q_Max)
        print("the value of Q_avg",Q_avg)
        print("the value of Q_std",Q_std)
        print("time ",time_run)
    elif sys.argv[4] == 'None' :
        Q_max, Q_avg, Q_std,time_run = de_main()
        print("the value of Q_max",Q_max)
        print("the value of Q_avg",Q_avg)
        print("the value of Q_std",Q_std)
        print("the value of time ",time_run) 