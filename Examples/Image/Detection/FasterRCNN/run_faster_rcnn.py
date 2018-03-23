# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import numpy as np
import argparse
import cntk
from FasterRCNN_train import prepare, train_faster_rcnn, store_eval_model_with_native_udf
from FasterRCNN_eval import compute_test_set_aps, FasterRCNN_Evaluator
from utils.config_helpers import merge_configs
from utils.plot_helpers import plot_test_set_results

def get_configuration():
    # load configs for detector, base network and data set
    from FasterRCNN_config import cfg as detector_cfg
    # for VGG16 base model use:         from utils.configs.VGG16_config import cfg as network_cfg
    # for AlexNet base model use:       from utils.configs.AlexNet_config import cfg as network_cfg
    from utils.configs.AlexNet_config import cfg as network_cfg
    # for Pascal VOC 2007 data set use: from utils.configs.Pascal_config import cfg as dataset_cfg
    # for the Grocery data set use:     from utils.configs.Grocery_config import cfg as dataset_cfg
    from utils.configs.MyDataSet_config import cfg as dataset_cfg

    return merge_configs([detector_cfg, network_cfg, dataset_cfg])

# trains and evaluates a Fast R-CNN model. 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='sub-command help', dest='command')

    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('-m', '--model', help='Model file', required=True)
    train_parser.add_argument('-p', '--epochs', help='Number of epochs', required=False, default=1, type=int)

    eval_parser = subparsers.add_parser('eval', help='Evaluate the model')
    eval_parser.add_argument('-m', '--model', help='Model file', required=True)

    args = parser.parse_args()

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.model)
    print('path is: {0}'.format(path))


    cfg = get_configuration()
    prepare(cfg, False)
    cntk.device.try_set_default_device(cntk.device.gpu(cfg.GPU_ID))
    
    if args.command == 'train':        
        print("TRAINING...")

        cfg['CNTK'].E2E_MAX_EPOCHS = args.epochs

        # train and test
        trained_model = train_faster_rcnn(cfg)
        eval_results = compute_test_set_aps(trained_model, cfg)

        # write AP results to output
        for class_name in eval_results: print('AP for {:>15} = {:.4f}'.format(class_name, eval_results[class_name]))
        print('Mean AP = {:.4f}'.format(np.nanmean(list(eval_results.values()))))


        print('Saving model to {0}'.format(path))
        trained_model.save(path)

        if cfg.STORE_EVAL_MODEL_WITH_NATIVE_UDF:
            store_eval_model_with_native_udf(trained_model, cfg)
            
    elif args.command == 'eval':
        print("EVALUATING...")
        trained_model = cntk.load_model(args.model)
        num_eval = min(cfg["DATA"].NUM_TEST_IMAGES, 10)
        results_folder = os.path.join(cfg.OUTPUT_PATH, cfg["DATA"].DATASET)
        evaluator = FasterRCNN_Evaluator(trained_model, cfg)
        plot_test_set_results(evaluator, num_eval, results_folder, cfg)        
    else:
        print("unknown command")