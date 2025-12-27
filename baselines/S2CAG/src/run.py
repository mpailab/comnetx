import tensorflow as tf
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
from time import time
from scipy import sparse
from sklearn.cluster import KMeans
from utils import *
from sklearn.utils.extmath import randomized_svd
from sklearn import cluster


flags = tf.compat.v1.flags
FLAGS = flags.FLAGS


# Parameters
flags.DEFINE_string('dataset', 'acm', 'Name of the graph dataset (`acm`, `dblp`, `arxiv`, `pubmed` or `wiki`).')
flags.DEFINE_integer('T', 10, 'Propagation order.')
flags.DEFINE_integer('runs', 5, 'Number of runs per power.')
flags.DEFINE_float('alpha', '0.9', 'the weight parameter in PowerIteration.')
flags.DEFINE_integer('fdim', '0', 'feature dimension.')
flags.DEFINE_string('method', 'sub', 'choose the specific algorithm mentioned in related paper.')
flags.DEFINE_float('gamma', '0.9', 'weight parameter for the second term in modularity maximization.')
flags.DEFINE_integer('tau', 7, 'the itertate times to get convergence results')

dataset = flags.FLAGS.dataset
T = flags.FLAGS.T
n_runs = flags.FLAGS.runs
alpha= flags.FLAGS.alpha
fdim = flags.FLAGS.fdim
method = flags.FLAGS.method
gamma=flags.FLAGS.gamma
tau=flags.FLAGS.tau
adj, features, labels, n_classes =  datagen(dataset)

if fdim>0 and fdim>n_classes:
    print("reduce attribute dim")
    U, Sigma, _ = randomized_svd(features, n_components=fdim, random_state=42)
    features = sparse.csr_matrix(U).dot(sparse.diags(Sigma))

norm_adj, features = preprocess_dataset(adj, features,
                                      tf_idf=True,
                                      sparse=True)
# print(type(norm_adj), type(features), type(labels), sep='\n')


features = features.toarray()
n, d = features.shape
k = n_classes


metrics = {}
metrics['acc'] = []
metrics['nmi'] = []
metrics['ari'] = []
metrics['time'] = []
adj_pt=sparse.coo_matrix(adj)


x = features


for run in range(n_runs):
    features = x

    t0 = time()

    P, Q = run_SSCAG(features, k, norm_adj, T, alpha,method=method,dataset=dataset,gamma=gamma,tau=tau)



metrics['time'].append(time()-t0)
metrics['acc'].append(clustering_accuracy(labels, P)*100)
# from sklearn.metrics import accuracy_score
# metrics['acc'].append(accuracy_score(labels, P) * 100)
metrics['nmi'].append(nmi(labels, P)*100)
metrics['ari'].append(ari(labels, P)*100)


results = {
      'mean': {k:(np.mean(v)).round(2) for k,v in metrics.items() }, 
      'std': {k:(np.std(v)).round(2) for k,v in metrics.items()}
    }

means = results['mean']
std = results['std']


print(f"{dataset} {T} {alpha} {method} {tau}")
print(f"acc: {means['acc']}±{std['acc']} & nmi: {means['nmi']}±{std['nmi']} & ari: {means['ari']}±{std['ari']} & time: {means['time']}±{std['time']}", sep=',')
def append_text_to_file(text, filename):
    with open(filename,'a') as file:
        file.write(text+'\n')
file_name=f"{dataset}_batch.txt"
append_text_to_file(f"T: {T} & alpha: {alpha} & method: {method} & tau: {tau} & acc: {means['acc']}±{std['acc']} & nmi: {means['nmi']}±{std['nmi']} & ari: {means['ari']}±{std['ari']} & time: {means['time']}±{std['time']}",file_name)

# print(f"{means['time']}±{std['time']} seconds")
