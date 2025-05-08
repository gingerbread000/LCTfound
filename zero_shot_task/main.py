import os
import sys
import argparse
import logging
import shutil
import traceback
import yaml

import torch
import numpy as np
import torch.utils.tensorboard as tb

from guided_diffusion.diffusion import Diffusion


torch.set_printoptions(sci_mode=False)


def parse_args_and_config():
    """Parses command-line arguments and loads configuration file."""
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for reproducibility")
    parser.add_argument("--exp", type=str, default="exp", help="Directory for saving logs and outputs")
    parser.add_argument("--input_folder", type=str, default="../0_CT_denoise", help="Input folder path")
    parser.add_argument("--deg", type=str, required=True, help="Degradation type")
    parser.add_argument("--path_y", type=str, required=True, help="Path of the test dataset")
    parser.add_argument("--sigma_y", type=float, default=0.0, help="Noise level in degraded observation")
    parser.add_argument("--eta", type=float, default=0.85, help="Eta parameter for sampling")
    parser.add_argument("--simplified", action="store_true", help="Use simplified DDNM (no SVD)")
    parser.add_argument("-i", "--image_folder", type=str, default="images", help="Image folder name")
    parser.add_argument("--deg_scale", type=float, default=0.0, help="Degradation scale")
    parser.add_argument("--verbose", type=str, default="info", help="Logging level: info | debug | warning | critical")
    parser.add_argument("--ni", action="store_true", help="No interaction (suitable for Slurm jobs)")
    parser.add_argument("--subset_start", type=int, default=-1, help="Subset start index")
    parser.add_argument("--subset_end", type=int, default=-1, help="Subset end index")
    parser.add_argument("-n", "--noise_type", type=str, default="gaussian", help="Noise type: gaussian | 3d_gaussian | poisson | speckle")
    parser.add_argument("--add_noise", action="store_true", help="Add synthetic noise")
    parser.add_argument("--noise_rate", type=float, default=0.0, help="Noise rate")
    parser.add_argument("--lambda_t", type=float, default=0.0, help="Lambda_t parameter")
    parser.add_argument("--output_folder", type=str, default="image_samples", help="Name of the output image folder")
    parser.add_argument("--tag_name", type=str, default="", help="Tag for experiment")
    parser.add_argument("--output_path", type=str, default="results_all", help="Path to save outputs")

    args = parser.parse_args()

    # Compute derived parameter
    args.sigma_y = args.lambda_t / 100

    # Load configuration file
    config_path = os.path.join("configs", args.config)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    # Set up logging
    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Logging level {args.verbose} not supported")

    logger = logging.getLogger()
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s - %(filename)s - %(asctime)s - %(message)s"))
    logger.addHandler(handler)

    # Prepare output directories
    os.makedirs(os.path.join(args.exp, "image_samples"), exist_ok=True)
    sub_image_folder = f"lambdat_{int(args.lambda_t)}"
    os.makedirs(args.output_path, exist_ok=True)

    args.image_folder = os.path.join(args.output_path, args.output_folder, sub_image_folder)
    if os.path.exists(args.image_folder):
        if args.ni:
            overwrite = True
        else:
            response = input(f"Image folder {args.image_folder} already exists. Overwrite? (Y/N) ")
            overwrite = response.strip().upper() == "Y"

        if overwrite:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        else:
            print("Output image folder exists. Program halted.")
            sys.exit(0)
    else:
        os.makedirs(args.image_folder)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    new_config.device = device

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    """Recursively converts a dict to an argparse.Namespace."""
    namespace = argparse.Namespace()
    for key, value in config.items():
        setattr(namespace, key, dict2namespace(value) if isinstance(value, dict) else value)
    return namespace


def main():
    os.environ["CUDA_LAUNCH_BLOCKING"] = "3"
    args, config = parse_args_and_config()

    try:
        args.subset_end = config.data.image_size
        logging.info(f"Subset end set to: {args.subset_end}")

        runner = Diffusion(args, config)
        avg_psnr, psnr_list = runner.sample(args.simplified)

        # Save results
        from scipy.io import savemat
        mat_file = os.path.join(args.output_path, args.output_folder, f"{args.lambda_t}.mat")
        savemat(mat_file, {'psnr_list': psnr_list})

        result_file = os.path.join(args.output_path, args.output_folder, f"{args.lambda_t}_{round(avg_psnr, 2)}.txt")
        with open(result_file, "w") as f:
            f.write("Hello, World!\n")

    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())
