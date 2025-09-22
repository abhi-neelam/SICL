import argparse

from solo.args.dataset import custom_dataset_args, dataset_args


def parse_args_knn() -> argparse.Namespace:
    """Parses arguments for offline K-NN.

    Returns:
        argparse.Namespace: a namespace containing all args needed for pretraining.
    """

    parser = argparse.ArgumentParser()

    # add knn args
    parser.add_argument("--pretrained_checkpoint_dir", type=str)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--linear_ckpt",default="", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--corrupt_level", type=int)
    parser.add_argument("--rank_range_1", type=int)
    parser.add_argument("--rank_range_2", type=int)
    parser.add_argument("--dimensionality", type=bool)
    parser.add_argument("--fgsm", type=bool)
    parser.add_argument("--eigen_list", type=bool)
    parser.add_argument("--rank_subset", type=bool)
    parser.add_argument("--eigenvector_alignment", type=bool)
    parser.add_argument("--numpy_dir", type=str)
    parser.add_argument("--target_type_val", type=str)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--k", type=int, nargs="+")
    parser.add_argument("--temperature", type=float, nargs="+")
    parser.add_argument("--distance_function", type=str, nargs="+")
    parser.add_argument("--feature_type", type=str, nargs="+")

    # add shared arguments
    dataset_args(parser)
    custom_dataset_args(parser)

    # parse args
    args = parser.parse_args()

    return args
