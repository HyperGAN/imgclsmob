import os
import time
import logging
import argparse

from chainer import global_config
from chainercv.utils import apply_to_iterator
from chainercv.utils import ProgressHook

from common.logger_utils import initialize_logging
from chainer_.utils import prepare_ch_context, prepare_model, Predictor
from chainer_.utils import get_composite_metric, report_accuracy
from chainer_.dataset_utils import get_dataset_metainfo
from chainer_.dataset_utils import get_val_data_source, get_test_data_source


def add_eval_parser_arguments(parser):
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="type of model to use. see model_provider for options")
    parser.add_argument(
        "--use-pretrained",
        action="store_true",
        help="enable using pretrained model from github repo")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="resume from previously saved parameters")
    parser.add_argument(
        "--data-subset",
        type=str,
        default="val",
        help="data subset. options are val and test")

    parser.add_argument(
        "--num-gpus",
        type=int,
        default=0,
        help="number of gpus to use")
    parser.add_argument(
        "-j",
        "--num-data-workers",
        dest="num_workers",
        default=4,
        type=int,
        help="number of preprocessing workers")

    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="training batch size per device (CPU/GPU)")

    parser.add_argument(
        "--save-dir",
        type=str,
        default="",
        help="directory of saved models and log-files")
    parser.add_argument(
        "--logging-file-name",
        type=str,
        default="train.log",
        help="filename of training log")

    parser.add_argument(
        "--log-packages",
        type=str,
        default="chainer, chainercv",
        help="list of python packages for logging")
    parser.add_argument(
        "--log-pip-packages",
        type=str,
        default="cupy-cuda100, chainer, chainercv",
        help="list of pip packages for logging")

    parser.add_argument(
        "--disable-cudnn-autotune",
        action="store_true",
        help="disable cudnn autotune for segmentation models")
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="show progress bar")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a model for image classification/segmentation (Chainer)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--dataset",
        type=str,
        default="ImageNet1K",
        help="dataset name. options are ImageNet1K, CUB200_2011, CIFAR10, CIFAR100, SVHN, VOC2012, ADE20K, Cityscapes, "
             "COCO")
    parser.add_argument(
        "--work-dir",
        type=str,
        default=os.path.join("..", "imgclsmob_data"),
        help="path to working directory only for dataset root path preset")

    args, _ = parser.parse_known_args()
    dataset_metainfo = get_dataset_metainfo(dataset_name=args.dataset)
    dataset_metainfo.add_dataset_parser_arguments(
        parser=parser,
        work_dir_path=args.work_dir)

    add_eval_parser_arguments(parser)

    args = parser.parse_args()
    return args


def test(net,
         test_data,
         metric,
         calc_weight_count=False,
         extended_log=False):
    tic = time.time()

    predictor = Predictor(
        model=net,
        transform=None)

    if calc_weight_count:
        weight_count = net.count_params()
        logging.info("Model: {} trainable parameters".format(weight_count))

    in_values, out_values, rest_values = apply_to_iterator(
        func=predictor,
        iterator=test_data["iterator"],
        hook=ProgressHook(test_data["ds_len"]))
    assert (len(rest_values) == 1)
    assert (len(out_values) == 1)
    assert (len(in_values) == 1)

    if True:
        labels = iter(rest_values[0])
        preds = iter(out_values[0])
        inputs = iter(in_values[0])
        for label, pred, inputi in zip(labels, preds, inputs):
            metric.update(label, pred)
            del label
            del pred
            del inputi
    else:
        import numpy as np
        metric.update(
            labels=np.array(list(rest_values[0])),
            preds=np.array(list(out_values[0])))

    accuracy_msg = report_accuracy(
        metric=metric,
        extended_log=extended_log)
    logging.info("Test: {}".format(accuracy_msg))
    logging.info("Time cost: {:.4f} sec".format(
        time.time() - tic))


def main():
    args = parse_args()

    if args.disable_cudnn_autotune:
        os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

    _, log_file_exist = initialize_logging(
        logging_dir_path=args.save_dir,
        logging_file_name=args.logging_file_name,
        script_args=args,
        log_packages=args.log_packages,
        log_pip_packages=args.log_pip_packages)

    ds_metainfo = get_dataset_metainfo(dataset_name=args.dataset)
    ds_metainfo.update(args=args)
    assert (ds_metainfo.ml_type != "imgseg") or (args.batch_size == 1)
    assert (ds_metainfo.ml_type != "imgseg") or args.disable_cudnn_autotune

    global_config.train = False
    use_gpus = prepare_ch_context(args.num_gpus)

    net = prepare_model(
        model_name=args.model,
        use_pretrained=args.use_pretrained,
        pretrained_model_file_path=args.resume.strip(),
        use_gpus=use_gpus,
        net_extra_kwargs=ds_metainfo.net_extra_kwargs,
        num_classes=args.num_classes,
        in_channels=args.in_channels)
    assert (hasattr(net, "classes"))
    assert (hasattr(net, "in_size"))

    if args.data_subset == "val":
        get_test_data_source_class = get_val_data_source
        test_metric = get_composite_metric(
            metric_names=ds_metainfo.val_metric_names,
            metric_extra_kwargs=ds_metainfo.val_metric_extra_kwargs)
    else:
        get_test_data_source_class = get_test_data_source
        test_metric = get_composite_metric(
            metric_names=ds_metainfo.test_metric_names,
            metric_extra_kwargs=ds_metainfo.test_metric_extra_kwargs)
    test_data = get_test_data_source_class(
        ds_metainfo=ds_metainfo,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    assert (args.use_pretrained or args.resume.strip())
    test(
        net=net,
        test_data=test_data,
        metric=test_metric,
        calc_weight_count=True,
        extended_log=True)


if __name__ == "__main__":
    main()
