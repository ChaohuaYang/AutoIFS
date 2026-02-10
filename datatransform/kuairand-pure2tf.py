from transform_kuairand_pure import DataTransform
from datetime import datetime, date
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
parser = argparse.ArgumentParser(description='Transfrom original data to TFRecord')

#parser.add_argument('avazu', type=string)
parser.add_argument('--label', type=str, default=["click", "long_view"])
parser.add_argument("--store_stat", action="store_true", default=True)
parser.add_argument("--threshold", type=int, default=2)
parser.add_argument("--dataset", type=Path, default='kuairand-pure-processed.csv')
parser.add_argument("--stats", type=Path, default='./data/kuairand-pure/stats_2')
parser.add_argument("--record", type=Path, default='./data/kuairand-pure/threshold_2')
parser.add_argument("--ratio", nargs='+', type=float, default=[0.8, 0.1, 0.1])

args = parser.parse_args()


class KuairandTransform(DataTransform):
    def __init__(self, dataset_path, path, stats_path, min_threshold, label_index, ratio, store_stat=False, seed=2021):
        super(KuairandTransform, self).__init__(dataset_path, stats_path, store_stat=store_stat, seed=seed)
        self.threshold = min_threshold
        self.label = label_index
        self.domain = 'domain_id'
        self.split = ratio
        self.path = path
        self.stats_path = stats_path
        self.name = ['Uc1', 'Uc2', 'Uc3', 'Uc4', 'Uc5', 'Uc6', 'Uc7', 'Uc8', 'Uc9', 'Uc10', 'Uc11', 'Uc12', 'Uc13', 'Uc14', 'Uc15', 'Uc16',
                     'Uc17', 'Uc18', 'Uc19', 'Uc20', 'Uc21', 'Uc22', 'Uc23', 'Uc24', 'Uc25', 'Uc26', 'Uc27', 'Ui28', 'Ui29', 'Ui30', 'Ui31',
                     'Ic32', 'Ic33', 'Ic34', 'Ic35', 'Ii36', 'Ii37', 'Ii38', 'Ii39', 'Ii40', 'Ii41', 'Ii42', 'Ii43', 'Ii44', 'Ii45', 'Ii46',
                     'Ii47', 'Ii48', 'Ii49', 'Ii50', 'Ii51', 'Ii52', 'Ii53', 'Ii54', 'Ii55', 'Ii56', 'Ii57', 'Ii58', 'Ii59', 'Ii60', 'Ii61',
                     'Ii62', 'Ii63', 'Ii64', 'Ii65', 'Ii66', 'Ii67', 'Ii68', 'Ii69', 'Ii70', 'Ii71', 'Ii72', 'Ii73', 'Ii74', 'Ii75', 'Ii76',
                     'Ii77', 'Ii78', 'Ii79', 'Ii80', 'Ii81', 'Ii82', 'Ii83', 'Ii84', 'Ii85', 'Ii86', 'Ii87', 'Ii88', 'd89', 'click', 'long_view', 'domain_id']
        # Uc*: user_category_feature, Ui*: user_numeric_feature, Ic*: item_category_feature, Ii*: item_numeric_feature, 'd*: doamin feature'
    def process(self):
        self._read(name=self.name, header=None, sep=",", label_index=self.label, domain_id=self.domain)
        if self.store_stat:
            white_list = ['Ui28', 'Ui29', 'Ui30', 'Ui31', 'Ii36', 'Ii37', 'Ii38', 'Ii39', 'Ii40', 'Ii41', 'Ii42', 'Ii43', 'Ii44', 'Ii45',
                          'Ii46', 'Ii47', 'Ii48', 'Ii49', 'Ii50', 'Ii51', 'Ii52', 'Ii53', 'Ii54', 'Ii55', 'Ii56', 'Ii57', 'Ii58', 'Ii59',
                          'Ii60', 'Ii61', 'Ii62', 'Ii63', 'Ii64', 'Ii65', 'Ii66', 'Ii67', 'Ii68', 'Ii69', 'Ii70', 'Ii71', 'Ii72', 'Ii73',
                          'Ii74', 'Ii75', 'Ii76', 'Ii77', 'Ii78', 'Ii79', 'Ii80', 'Ii81', 'Ii82', 'Ii83', 'Ii84', 'Ii85', 'Ii86', 'Ii87',
                          'Ii88']
            print('white_list len:', len(white_list))
            self.generate_and_filter(threshold=self.threshold, label_index=self.label, domain_id=self.domain, white_list=white_list)

        tr, te, val = self.random_split(ratio=self.split)
        self.transform_tfrecord(tr, self.path, "train", label_index=self.label, domain_id=self.domain)
        self.transform_tfrecord(val, self.path, "validation", label_index=self.label, domain_id=self.domain)
        self.transform_tfrecord(te, self.path, "test", label_index=self.label, domain_id=self.domain)

    def _process_x(self):
        print(self.data[self.data[self.label[0]] == 1].shape)
        print(self.data[self.data[self.label[1]] == 1].shape)

        def bucket(value):
            if not pd.isna(value):
                if value > 2:
                    value = int(np.floor(np.log(value) ** 2))
                else:
                    value = int(value)
            return value
        numeric_list = ['Ui28', 'Ui29', 'Ui30', 'Ui31', 'Ii36', 'Ii37', 'Ii38', 'Ii39', 'Ii40', 'Ii41', 'Ii42', 'Ii43', 'Ii44', 'Ii45',
                          'Ii46', 'Ii47', 'Ii48', 'Ii49', 'Ii50', 'Ii51', 'Ii52', 'Ii53', 'Ii54', 'Ii55', 'Ii56', 'Ii57', 'Ii58', 'Ii59',
                          'Ii60', 'Ii61', 'Ii62', 'Ii63', 'Ii64', 'Ii65', 'Ii66', 'Ii67', 'Ii68', 'Ii69', 'Ii70', 'Ii71', 'Ii72', 'Ii73',
                          'Ii74', 'Ii75', 'Ii76', 'Ii77', 'Ii78', 'Ii79', 'Ii80', 'Ii81', 'Ii82', 'Ii83', 'Ii84', 'Ii85', 'Ii86', 'Ii87',
                          'Ii88']
        for col_name in numeric_list:
            self.data[col_name] = self.data[col_name].apply(bucket)

    def _process_y(self):
        pass

if __name__ == "__main__":
    tranformer = KuairandTransform(args.dataset, args.record, args.stats,
                               args.threshold, args.label, args.ratio, store_stat=args.store_stat)
    tranformer.process()
