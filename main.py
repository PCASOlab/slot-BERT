"""Entry point for the public Slot-BERT release.

This launcher intentionally keeps the runnable path focused on Slot-BERT +
contrastive slot training, while reusing the validated training flow from
`0_full_code_back/main.py`.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

import numpy as np
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate Slot-BERT")
    parser.add_argument(
        "config",
        nargs="?",
        default="model_config.yml",
        help="Path to model config (inside 0_full_code_back)",
    )
    parser.add_argument("--config_overrides_file", help="Configuration override file")
    parser.add_argument("config_overrides", nargs="*", help="Additional config overrides")

    parser.add_argument(
        "--code-root",
        # default="0_full_code_back",
        default="src",
        help="Internal code root used by this release entrypoint",
    )
    parser.add_argument(
        "--mode",
        default="train_miccai",
        help="Value for WORKING_DIR_IMPORT_MODE (e.g., train_miccai/train_cholec)",
    )
    parser.add_argument("--contrastive-temp", type=float, default=1.0)
    parser.add_argument("--sim-threshold", type=float, default=1.0)  # lower to enable Xslot
    parser.add_argument("--slot-ini", default="rnn")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.environ["WORKING_DIR_IMPORT_MODE"] = args.mode

    repo_root = pathlib.Path(__file__).resolve().parent
    code_root = pathlib.Path(args.code_root).resolve()
    if not code_root.exists():
        raise FileNotFoundError(f"code root not found: {code_root}")

    # Keep import behavior predictable no matter where `python main.py` is launched.
    # If a public `src/` layout is present, prioritize it so local package imports work.
    src_root = repo_root / "src"
    import_roots = [repo_root, src_root, code_root]
    for root in reversed(import_roots):
        if root.exists() and str(root) not in sys.path:
            sys.path.insert(0, str(root))

    from dataset.dataset import myDataloader
    from dataset import io
    from display import Display
    from model import model_infer_slot_att
    from working_dir_root import (
        Batch_size,
        Continue_flag,
        Data_percentage,
        Display_down_sample,
        Display_flag,
        Enable_student,
        Evaluation,
        Evaluation_slots,
        GPU_mode,
        Gpu_selection,
        Load_feature,
        Load_flow,
        Max_epoch,
        Output_root,
        Save_feature_OLG,
        Visdom_flag,
        img_size,
        loadmodel_index,
        selected_data,
    )

    if torch.cuda.is_available():
        print("CUDA available devices:", torch.cuda.device_count())

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    run_args = argparse.Namespace(
        config=args.config,
        config_overrides_file=args.config_overrides_file,
        config_overrides=args.config_overrides,
    )

    model_infer = model_infer_slot_att._Model_infer(
        run_args,
        GPU_mode,
        num_gpus,
        Using_contrast=True,
        Using_SP_regu=False,
        Using_SP=False,
        Using_slot_bert=True,
        slot_ini=args.slot_ini,
        Sim_threshold=args.sim_threshold,
        cTemp=args.contrastive_temp,
        gpu_selection=Gpu_selection,
        pooling="max",
        TPC=True,
    )
    device = model_infer.device

    dataset_tag = "+".join(selected_data) if isinstance(selected_data, list) else selected_data
    output_root = (
        Output_root
        + f"slotbert_contrastive_temp{args.contrastive_temp}_"
        + f"{dataset_tag}{Data_percentage}/"
    )
    io.self_check_path_create(output_root)

    data_loader = myDataloader(
        img_size=img_size,
        Display_loading_video=False,
        Read_from_pkl=True,
        Save_pkl=False,
        Load_flow=Load_flow,
        Load_feature=Load_feature,
        Train_list="else",
        Device=device,
    )

    if Continue_flag:
        ckpt_path = code_root / "Model_checkpoint" / "Abdominal" / f"model{loadmodel_index}"
        model_infer.model.load_state_dict(torch.load(ckpt_path, map_location=device))

    displayer = Display(run_args)
    epoch, read_id, visdom_id, saver_id = 0, 0, 0, 0
    features = None

    while True:
        start_time = time.time()
        input_videos, labels = data_loader.read_a_batch(this_epoch=epoch)

        input_videos_gpu = torch.from_numpy(np.float32(input_videos)).to(device)
        labels_gpu = torch.from_numpy(np.float32(labels)).to(device)
        input_flows = data_loader.input_flows * 1.0 / 255.0
        input_flows_gpu = torch.from_numpy(np.float32(input_flows)).to(device)

        if Load_feature:
            features = data_loader.features.to(device)

        model_infer.forward(input_videos_gpu, input_flows_gpu, features, Enable_student, epoch=epoch)

        if not Evaluation and not Evaluation_slots:
            model_infer.optimization(labels_gpu, Enable_student)

        if Display_flag and read_id % Display_down_sample == 0:
            data_loader.labels = data_loader.labels * 0 + 1
            displayer.train_display(model_infer, data_loader, read_id, output_root)

        if read_id % 1000 == 0:
            torch.save(model_infer.model.state_dict(), f"{output_root}model{saver_id}.pth")
            saver_id = (saver_id + 1) % 2
            print("iteration time:", time.time() - start_time)

        read_id += 1
        visdom_id += 1

        if data_loader.all_read_flag == 1:
            epoch += 1
            data_loader.all_read_flag = 0
            read_id = 0
            print("finished epoch", epoch)
            if Evaluation or Save_feature_OLG or epoch > Max_epoch:
                break


if __name__ == "__main__":
    main()
