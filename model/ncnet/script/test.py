import torch

from ncNetNew.common import Counter, is_matching
from ncNetNew.common.translate import (
    postprocessing,
    get_all_table_columns
)
from ncNetNew.model.ncnet import ncNet

import random
import numpy as np
import pandas as pd


if __name__ == "__main__":

    from argparse import Namespace

    opt = Namespace()
    base_dir = '/repo/ncNetNew'
    opt.model = f'{base_dir}/save_models/trained_model.pt'
    opt.data_dir = f'{base_dir}/dataset/dataset_final/'
    opt.db_info = f'{base_dir}/dataset/database_information.csv'
    opt.test_data = f'{base_dir}/dataset/dataset_final/test.csv'
    opt.db_schema = f'{base_dir}/dataset/db_tables_columns.json'
    opt.db_tables_columns_types = f'{base_dir}/dataset/db_tables_columns_types.json'
    opt.batch_size = 128
    opt.max_input_length = 128
    opt.show_progress = False
    opt.train_model_path = f"{base_dir}/save_models/trained_model.pt"

    print("the input parameters: ", opt)

    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    m1 = ncNet(
        trained_model_path=opt.train_model_path
    )

    ncNet = m1.ncNet
    db_tables_columns = get_all_table_columns(opt.db_schema)
    db_tables_columns_types = get_all_table_columns(opt.db_tables_columns_types)

    test_df = pd.read_csv(opt.test_data)

    only_nl_cnt = 0
    only_nl_match = 0
    nl_template_cnt = 0
    nl_template_match = 0

    counter = Counter(total=len(test_df))
    counter.start()

    for index, row in test_df.iterrows():

        try:
            gold_query = row['labels'].lower()
            src = row['source'].lower()
            tok_types = row['token_types']
            db_id = row['db_id']
            table_name = gold_query.split(' ')[gold_query.split(' ').index('data') + 1]

            pred_query, attention, enc_attention = m1.translate(
                input_src=src,
                token_types=tok_types,
                visualization_aware_translation=True,
                show_progress=False,
                db_id=db_id,
                table_name=table_name,
                db_tables_columns=db_tables_columns,
                db_tables_columns_types=db_tables_columns_types,
            )

            old_pred_query = pred_query

            if '[t]' not in src:
                # with template
                pred_query = postprocessing(gold_query, pred_query, True, src)

                nl_template_cnt += 1
                if is_matching(gold_query, pred_query):
                    nl_template_match += 1
            else:
                # without template
                pred_query = postprocessing(gold_query, pred_query, False, src)

                only_nl_cnt += 1
                if is_matching(gold_query, pred_query):
                    only_nl_match += 1

        except Exception as e:
            print(f'error {e}')

        counter.update()
        # if index > 100:
        #     break

    print("--")
    print('ncNet w/o chart template:', only_nl_match / only_nl_cnt)
    print('ncNet with chart template:', nl_template_match / nl_template_cnt)
    print('ncNet overall:', (only_nl_match + nl_template_match) / (only_nl_cnt + nl_template_cnt))
