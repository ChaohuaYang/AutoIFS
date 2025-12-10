## WSDM26_AutoIFS

Experiments codes for the paper:

Chaohua Yang, Dugang Liu, Shiwei Li, Yuwen Fu, Xing Tang, Weihong Luo, Xiangyu Zhao, Xiuqiang He, and Zhong Ming. Automated information flow selection for multi-scenario multi-task recommendation. In Proceedings of WSDM '24.

**Please cite our WSDM '26 paper if you use our codes. Thanks!**


##  Requirement
See the contents of requirements.txt

## Data Preprocessing

Please download the original data ([MovieLens-1M](https://grouplens.org/datasets/movielens/) and [KuaiRand-Pure](https://kuairand.com/)) and place them in the corresponding directory of data.

You can prepare the MovieLens1M data in the following code.
```
# process origin data
python datatransform/preprocess_ml1m.py --dataset_path ../data/ml-1m/raw_data/
# datatransform
python datatransform/ml1m2tf.py --dataset_path ../data/ml-1m/raw_data/ml-1m-processed.csv
```

You can prepare the KuaiRand-Pure data in the following code.

```
# process origin data
python datatransform/preprocess_kuairand-pure.py --dataset_path ../data/kuairand-pure/raw_data/
# datatransform
python datatransform/kuairand-pure2tf.py --dataset ../data/kuairand-pure/raw_data/kuairand-pure-processed.csv
```

## Usage


An example of running AutoIFS:
```
# For MovieLens-!M
python autoifs_trainer.py --dataset ml-1m

# For Kuairand-Pure
python autoifs_trainer.py --dataset kuairand-pure
```


## 

If you have any issues or ideas, feel free to contact us ([chaohua.ych@gmail.com](mailto:chaohua.ych@gmail.com)).

