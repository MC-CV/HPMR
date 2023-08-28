# Hierarchical Projection Enhanced Multi-behavior Recommendation

This repository contains TensorFlow codes and datasets for the paper.

## Environment
The codes of HPMR are implemented and tested under the following development environment:
* python=3.7.12
* tensorflow=1.15.0
* numpy=1.19.5
* scipy=1.7.3
* tensorflow-determinism=0.3.0

## Datasets
We utilized two datasets to evaluate HPMR: <i>Beibei</i> and <i>Taobao</i>. The <i>purchase</i> behavior is taken as the target behavior for all datasets. The last target behavior for the test users are left out to compose the testing set. We filtered out users and items with too few interactions.

## How to run

* Beibei
```
python HPMR.py --gpu_id 6 --gnn_layer 3 --transfer_gnn_layer 1 --encoder gccf --alpha '[0,1,3]' --re_mult 2 
```
* Taobao
```
python HPMR.py --gpu_id 7 --gnn_layer 4 --transfer_gnn_layer 1 --encoder gccf --alpha '[0,1,3]' --re_mult 2 --dataset Taobao --mess_drop '[0.1]'
```

## Citation
If you want to use our codes and datasets in your research, please cite:
```
@inproceedings{meng2023hierarchical,
  title={Hierarchical Projection Enhanced Multi-behavior Recommendation},
  author={Meng, Chang and Zhang, Hengyu and Guo, Wei and Guo, Huifeng and Liu, Haotian and Zhang, Yingxue and Zheng, Hongkun and Tang, Ruiming and Li, Xiu and Zhang, Rui},
  booktitle={Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={4649--4660},
  year={2023}
}
```

