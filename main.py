import config
import os
from create_labels import *
from trainer import *


def dataset_creation(config, multi_label=False, src_folder='../Dataset/'):
    dataset_csv = config.dataset_path
    with open(dataset_csv, 'a') as csv_file:

        # config,_ = config.get_config()
        writer = csv.writer(csv_file, delimiter=',',
            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # creating multi-label multi-class dataset
        if multi_label:
            for i in os.listdir(src_folder):
                folder = os.path.join(src_folder, i)
                create_multi_labels(dataset_csv, folder, categ="beauty")
        else:
            for i in os.listdir(src_folder):
                if not i.endswith('.csv'):
                    folder = os.path.join(src_folder, i)
                    create_single_labels(dataset_csv, folder)
    


def main(config):
    # multi_label = False
    # if not os.path.exists(config.dataset_path):
    #     with open(config.dataset_path, 'a') as csv_file:
    #         writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    #         if multi_label:
    #             writer.writerow(['file_path', 'isbeauty', 'skill'])
    #         else:
    #             writer.writerow(['file_path', 'skill'])

    # dataset_creation(config, multi_label=multi_label)
    trainer = TrainModel()

    model = trainer.train()
config,_ = config.get_config()
main(config)