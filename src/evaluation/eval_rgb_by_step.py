import os
import sys
import argparse

sys.path.append(os.getcwd())

from tensorboardX import SummaryWriter
from src.naruto.cfg_loader import load_cfg
from src.slam import init_SLAM_model
from src.utils.general_utils import fix_random_seed, InfoPrinter


def argument_parsing() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Arguments to run SplaTAM RGB evaluation by step."
    )
    parser.add_argument("--cfg", type=str, default="configs/default.py",
                        help="NARUTO config")
    parser.add_argument("--setting", type=str, default="configs/semantic/setting.py",
                        help="Semantic voxelize config (sem + planner overrides)")
    parser.add_argument("--result_dir", type=str, default=None,
                        help="result directory")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed; also used as the initial pose idx for Replica")
    parser.add_argument("--enable_vis", type=int, default=None,
                        help="enable visualization. 1: True, 0: False")
    parser.add_argument("--stage", type=str, default="final",
                        help="ONLY for SplaTAM result evaluation")
    parser.add_argument("--step", type=int, default=1100,
                        help="ONLY for SplaTAM result evaluation")
    parser.add_argument("--eval_data_dir", type=str, default=None,
                        help="Exact eval trajectory folder containing traj.txt and results_habitat/")
    parser.add_argument("--eval_data_basedir", type=str, default=None,
                        help="Eval dataset basedir; used with --eval_sequence")
    parser.add_argument("--eval_sequence", type=str, default=None,
                        help="Eval dataset sequence folder name under --eval_data_basedir")
    parser.add_argument("--eval_suffix", type=str, default=None,
                        help="Output suffix for eval directory; defaults to --stage")
    return parser.parse_args()


if __name__ == "__main__":
    info_printer = InfoPrinter("ActiveSem")

    info_printer("Parsing arguments...", 0, "Initialization")
    args = argument_parsing()

    info_printer("Loading configuration...", 0, "Initialization")
    main_cfg = load_cfg(args)
    main_cfg.dump(os.path.join(main_cfg.dirs.result_dir, "main_cfg.json"))
    info_printer.update_total_step(main_cfg.general.num_iter)
    info_printer.update_scene(main_cfg.general.dataset + " - " + main_cfg.general.scene)

    info_printer("Fix random seed...", 0, "Initialization")
    fix_random_seed(main_cfg.general.seed)

    log_savedir = os.path.join(main_cfg.dirs.result_dir, "logger")
    os.makedirs(log_savedir, exist_ok=True)
    logger = SummaryWriter(f"{log_savedir}")

    slam = init_SLAM_model(main_cfg, info_printer, logger)
    slam.load_params_by_step(step=args.step, stage=args.stage)

    eval_suffix = args.eval_suffix or args.stage
    slam.eval_result(eval_dir_suffix=eval_suffix, ignore_first_frame=True, save_frames=True)
