from transform_ml1m import DataTransform
from datetime import datetime, date
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
parser = argparse.ArgumentParser(description='Transfrom original data to TFRecord')

parser.add_argument('--label', type=str, default=["click", "like"])
parser.add_argument("--store_stat", action="store_true", default=True)
parser.add_argument("--threshold", type=int, default=0)
parser.add_argument("--dataset", type=Path, default='ml-1m-processed.csv')
parser.add_argument("--stats", type=Path, default='./data/ml-1m/stats_0')
parser.add_argument("--record", type=Path, default='./data/ml-1m/threshold_0')
parser.add_argument("--ratio", nargs='+', type=float, default=[0.8, 0.1, 0.1])

args = parser.parse_args()


class Transform(DataTransform):
    def __init__(self, dataset_path,  path, stats_path, min_threshold, label_index, ratio, store_stat=False, seed=2021):
        super(Transform, self).__init__(dataset_path, stats_path, store_stat=store_stat, seed=seed)
        self.threshold = min_threshold
        self.label = label_index
        self.domain = 'domain_id'
        self.split = ratio
        self.path = path
        self.stats_path = stats_path
        self.name = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip', 'cate_id', 'domain_indicator', "click", "like", 'domain_id']

    def process(self):
        self._read(name=self.name, header=True, sep=",", label_index=self.label, domain_id=self.domain)
        if self.store_stat:
            white_list = []
            print('white_list len:', len(white_list))
            self.generate_and_filter(threshold=self.threshold, label_index=self.label, domain_id=self.domain, white_list=white_list)

        tr, te, val = self.random_split(ratio=self.split)
        self.transform_tfrecord(tr, self.path, "train", label_index=self.label, domain_id=self.domain)
        self.transform_tfrecord(val, self.path, "validation", label_index=self.label, domain_id=self.domain)
        self.transform_tfrecord(te, self.path, "test", label_index=self.label, domain_id=self.domain)


    def _process_x(self):
        print(self.data[self.data[self.label[0]] == 1].shape)
        print(self.data[self.data[self.label[1]] == 1].shape)


    def _process_y(self):
        pass

if __name__ == "__main__":
    tranformer = Transform(args.dataset, args.record, args.stats,
                                 args.threshold, args.label,
                                 args.ratio, store_stat=args.store_stat)
    tranformer.process()
