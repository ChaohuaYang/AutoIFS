import pandas as pd
import argparse
import os


def raw_data_process(path):
    # load ratings.dat
    ratings_columns = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv(path + 'ratings.dat', sep='::', header=None, names=ratings_columns, engine='python')

    # load users.dat
    users_columns = ['user_id', 'gender', 'age', 'occupation', 'zip']
    users = pd.read_csv(path + 'users.dat', sep='::', header=None, names=users_columns, engine='python')

    # load movies.dat
    movies_columns = ['movie_id', 'title', 'genres']
    movies = pd.read_csv(path + 'movies.dat', sep='::', header=None, names=movies_columns, engine='python')

    # merge data
    merged_data = pd.merge(ratings, movies, on='movie_id')  # 按电影 ID 合并结果和 movies 数据
    merged_data = pd.merge(merged_data, users, on='user_id')  # 按用户 ID 合并 ratings 和 users 数据

    # reindex columns
    target_columns = ['user_id', 'movie_id', 'rating', 'timestamp', 'title', 'genres', 'gender', 'age', 'occupation', 'zip']
    merged_data = merged_data[target_columns]

    # save CSV
    merged_data.to_csv(path + 'ml-1m.csv', index=False)

def map_group_indicator(age, list_group):
    l = len(list(list_group))
    for i in range(l):
        if age in list_group[i]:
            return i

def convert_target(val, target):
    v = int(val)
    if target == 'click':
        thre = 3
    elif target == 'like':
        thre = 4
    else:
        assert 0, 'wrong target'
    if v > thre:
        return int(1)
    else:
        return int(0)


def data_process(path):
    data = pd.read_csv(path + "ml-1m.csv")

    data["cate_id"] = data["genres"].apply(lambda x: x.split("|")[0])
    del data["genres"]

    x_used_cols = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip', 'cate_id', 'domain_indicator']

    group1 = {1, 18}
    group2 = {25}
    group3 = {35, 45, 50, 56}

    domain_num = 3

    data["domain_indicator"] = data["age"].apply(lambda x: map_group_indicator(x, [group1, group2, group3]))

    print('domain:', data["domain_indicator"].nunique())
    print('user num:', data[data["domain_indicator"] == 0]['user_id'].nunique(),
          data[data["domain_indicator"] == 1]['user_id'].nunique(),
          data[data["domain_indicator"] == 2]['user_id'].nunique())
    print('item num:', data[data["domain_indicator"] == 0]['movie_id'].nunique(),
          data[data["domain_indicator"] == 1]['movie_id'].nunique(),
          data[data["domain_indicator"] == 2]['movie_id'].nunique())
    print('inter num:', data[data["domain_indicator"] == 0].shape[0],
          data[data["domain_indicator"] == 1].shape[0],
          data[data["domain_indicator"] == 2].shape[0])

    useless_features = ['title', 'timestamp']

    for feature in useless_features:
        del data[feature]

    data['click'] = data['rating'].apply(lambda x: convert_target(x, 'click'))
    data['like'] = data['rating'].apply(lambda x: convert_target(x, 'like'))
    data['domain_id'] = data["domain_indicator"].copy()
    del data['rating']

    data.to_csv(path + 'ml-1m-processed.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='../data/ml-1m/raw_data/', help="ml-1m")
    args, unparsed = parser.parse_known_args()
    dataset_path = args.dataset_path
    raw_data_process(dataset_path)
    data_process(dataset_path)

