# Spectral Subspace Clustering for Attributed Graphs

<p align="center">
	<a href="https://arxiv.org/abs/2411.11074"><img src="https://img.shields.io/badge/arXiv-pdf-yellowgreen"></a>
	<a href="https://github.com/HKBU-LAGAS/S2CAG/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
</p>

--------------------------------------------------------------------------------

## 📝 Envirorment

tensorflow --2.10.0

numpy --1.24.4

networkx --3.1

scikit-learn --1.3.2

scipy --1.10.1

## Parameters

| Parameter |   Type  | Default | Description                                                             |
| :-------: | :-----: | :-----: | :---------------------------------------------------------------------- |
| `dataset` |  string |  `acm`  | Name of the graph dataset (`acm`, `dblp`, `arxiv`, `pubmed` or `wiki`). |
|  `T`      | integer |   `10`  | Propagation order.                                                      |
|  `alpha`  |  float  |  `0.9`  | the weight parameter in PowerIteration.                                 |
|  `gamma`  |  float  |   `1.0` | weight parameter for the second term in modularity maximization.        |
|  `tau`    | integer |   `7`   | the itertate times to get convergence results.                          |
|  `runs`   | integer |   `5`   | Number of runs.                                                         |

## 🚀 Example
You can get the results in paper by running following instruction.
```bash
$bash run.sh 
```

## 📚 Datasets
You can download all datasets from [HERE](https://www.dropbox.com/scl/fi/9olm295mxf415c5pf8hvy/S2CAG-datasets.zip?rlkey=3iuzartucsdatbolgvqv1gcfr&st=1ucjwnkf&dl=0).

---------


## 😀 Contact

For any questions or feedback, feel free to contact Miss [Xiaoyang LIN](mailto:csxylin@comp.hkbu.edu.hk).


## 🌟 Citation

If you find S2CAG and M-S2CAG useful in your research or applications, please kindly cite:
```tex
@inproceedings{lin2024s2cag,
title={Spectral Subspace Clustering for Attributed Graphs}, 
author={Xiaoyang Lin and Renchi Yang and Haoran Zheng and Xiangyu Ke},
booktitle={Proceedings of the 31th ACM SIGKDD conference on knowledge discovery and data mining},
pages={To Appear},
year={2024}
}
```

## 😊 Acknowledgements
You may refer to related work that serves as foundations for our framework and code repository, [SAGSC](https://github.com/chakib401/SAGSC), etc. Thanks for their wonderful works.


