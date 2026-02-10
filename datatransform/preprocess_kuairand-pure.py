import pandas as pd
import argparse

def merge_log_data(data_path, file_names):
    log_data = []
    row_num = 0
    for file_name in file_names:
        data = pd.read_csv(data_path + file_name, sep=",")
        row_num += data.shape[0]
        log_data.append(data)
    merged_log_data = pd.concat(log_data, axis=0, ignore_index=True)
    assert merged_log_data.shape[0] == row_num
    print('log_data sample num:', merged_log_data.shape[0], 'log_data col num:', merged_log_data.shape[1])
    return merged_log_data

def filter_domain_data(raw_data, domain_id_list, domain_field='tab'):
    if max(domain_id_list) > 14 or min(domain_id_list) < 0:
        print('domain_id out of range error')
    else:
        domain_data = raw_data[raw_data[domain_field].isin(domain_id_list)]
        for domain_id in domain_id_list:
            print('domain {} sample num:'.format(domain_id), domain_data[domain_data[domain_field]==domain_id].shape[0])
            print('domain {} user num:'.format(domain_id), len(domain_data[domain_data[domain_field] == domain_id]['user_id'].unique()))
            print('domain {} item num:'.format(domain_id), len(domain_data[domain_data[domain_field] == domain_id]['video_id'].unique()))
        return domain_data

def reindex_domain_id(domain_data, domain_id_list, domain_field='tab'):
    domain_new_id = 0
    domain_id_list = sorted(domain_id_list)
    for domain_id in domain_id_list:
        domain_data[domain_field] = domain_data[domain_field].replace(domain_id, domain_new_id)
        print('domain {} sample num:'.format(domain_new_id), domain_data[domain_data[domain_field] == domain_new_id].shape[0])
        domain_new_id += 1
    return domain_data

def get_origin_label(origin_data):
    keep_column = ['user_id', 'video_id', 'tab',  'is_click', 'long_view']
    origin_data = origin_data[keep_column]
    target_column = ['is_click', 'long_view']
    domain_column = ['tab']
    for col in target_column:
        origin_data[col] = origin_data[col].astype(float)

    return origin_data, target_column, domain_column

def process_user_features(data_path, file_name):
    user_features = pd.read_csv(data_path + file_name)
    user_features = user_features.fillna(0)
    user_features['user_id_feature'] = user_features['user_id'].apply(lambda x: x)
    user_features = user_features.rename(columns={'follow_user_num': 'follow_user_num_uf'})

    key_features = ['user_id']
    numeric_features = 'follow_user_num_uf,fans_user_num,friend_user_num,register_days'.split(',')
    cate_features = 'user_id_feature,user_active_degree,is_lowactive_period,is_live_streamer,is_video_author,follow_user_num_range,' \
                    'fans_user_num_range,friend_user_num_range,register_days_range'.split(',')
    onehot_features = ['onehot_feat' + str(i) for i in range(18)]
    cate_features = cate_features + onehot_features
    # print(numeric_features)
    # print(cate_features)
    all_columns = key_features + numeric_features + cate_features
    user_features = user_features[all_columns]

    return user_features, numeric_features, cate_features, key_features


def process_item_features(data_path, basic_file_name, sta_file_name):
    item_basic_features = pd.read_csv(data_path + basic_file_name)
    item_statistics_features = pd.read_csv(data_path + sta_file_name)
    item_features = pd.merge(item_basic_features, item_statistics_features, on='video_id', how='outer')
    item_features = item_features.fillna(0)
    item_features['video_id_feature'] = item_features['video_id'].apply(lambda x: x)

    key_features = ['video_id']
    cate_features = 'video_id_feature,video_type,upload_type,music_type'.split(',')
    basic_numeric_features = 'video_duration,server_width,server_height'.split(',')
    statistics_numeric_features = [x for x in item_statistics_features.columns if x not in ['video_id', 'counts']]
    numeric_features = basic_numeric_features + statistics_numeric_features

    all_columns = key_features + numeric_features + cate_features
    item_features = item_features[all_columns]

    return item_features, numeric_features, cate_features, key_features

def origin_data_process(data_path, domain_data):
    origin_label, target_column, domain_column = get_origin_label(domain_data)
    print(origin_label.head(5), origin_label.shape)
    user_features, uf_numeric_features, uf_cate_features, user_id = process_user_features(data_path, "user_features_pure.csv")
    item_features, if_numeric_features, if_cate_features, video_id = process_item_features(data_path, "video_features_basic_pure.csv", "video_features_statistic_pure.csv")
    sample_data = pd.merge(origin_label, user_features, on='user_id', how='left')
    sample_data = pd.merge(sample_data, item_features, on='video_id', how='left')

    user_features_columns = uf_cate_features + uf_numeric_features
    item_features_columns = if_cate_features + if_numeric_features
    print('uf_cate_features num:', len(uf_cate_features), 'uf_numeric_features num:', len(uf_numeric_features))
    print('if_cate_features num:', len(if_cate_features), 'if_numeric_features num:', len(if_numeric_features))
    print('domain_features num:', len(domain_column), 'target_column num:', len(target_column))
    save_columns = user_features_columns + item_features_columns + domain_column + target_column
    print('save_column num:', len(save_columns))
    result_data = sample_data[save_columns]
    result_data = result_data.fillna(0)
    # result_data = result_data.sample(frac=1)

    return result_data, target_column, domain_column, user_features_columns, item_features_columns

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='../data/kuairand-pure/raw_data/', help="ml-1m")
    args, unparsed = parser.parse_known_args()
    data_path = args.dataset_path
    save_path = args.dataset_path
    file_names = ['log_standard_4_22_to_5_08_pure.csv', 'log_standard_4_08_to_4_21_pure.csv',
                  'log_random_4_22_to_5_08_pure.csv']
    domain_field = 'tab'
    domain_id_list = [0, 1, 4]

    merged_data = merge_log_data(data_path, file_names)
    domain_data = filter_domain_data(merged_data, domain_id_list, domain_field)
    domain_data = reindex_domain_id(domain_data, domain_id_list, domain_field)

    result_data, target_column, domain_column, user_features_columns, item_features_columns = origin_data_process(data_path, domain_data)
    result_data['domain_id'] = result_data[domain_column[0]].apply(lambda x: x)
    print(result_data.shape)
    result_data.to_csv(save_path + 'kuairand-pure-processed.csv', header=None, index=False)
