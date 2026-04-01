import argparse
import json
from datetime import datetime
from pathlib import Path

import isaacgym
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from legged_gym.envs.g1_vision.highlevel_policy import HighLevelMLP, parse_hidden_dims


def parse_args():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="legged_gym/logs/g1_vision_bc")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--hidden_dims", type=str, default="128,128")
    parser.add_argument("--wz_weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_files", type=int, default=-1)
    parser.add_argument("--only_success_episodes", action="store_true", default=False)
    parser.add_argument("--min_final_path_ratio", type=float, default=0.0)
    parser.add_argument("--split_by_episode", action="store_true", default=True)
    parser.add_argument("--split_by_sample", action="store_false", dest="split_by_episode")
    return parser.parse_args()


def _resolve_dir(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def discover_npz_files(dataset_dir: Path, max_files: int):
    files = sorted((dataset_dir / "rollouts").glob("episode_*.npz"))
    if len(files) == 0:
        files = sorted(dataset_dir.glob("*.npz"))
    if max_files is not None and max_files > 0:
        files = files[:max_files]
    if len(files) == 0:
        raise FileNotFoundError(f"No dataset files found under: {dataset_dir}")
    return files


def filter_episode_files(files, only_success_episodes: bool, min_final_path_ratio: float):
    if (not only_success_episodes) and float(min_final_path_ratio) <= 0.0:
        return files

    kept = []
    for p in files:
        arr = np.load(p, allow_pickle=True)
        keep = True
        if only_success_episodes:
            meta = {}
            if "episode_meta_json" in arr:
                try:
                    meta = json.loads(str(arr["episode_meta_json"][0]))
                except Exception:
                    meta = {}
            keep = bool(meta.get("terminated_by_success", False))

        if keep and float(min_final_path_ratio) > 0.0:
            if "path_progress_ratio" in arr and arr["path_progress_ratio"].shape[0] > 0:
                final_ratio = float(arr["path_progress_ratio"][-1])
                keep = final_ratio >= float(min_final_path_ratio)
            else:
                keep = False

        if keep:
            kept.append(p)
    return kept


def load_dataset(files):
    feats = []
    acts = []
    feature_names = None
    action_names = None

    for p in files:
        arr = np.load(p, allow_pickle=True)
        x = arr["features"].astype(np.float32)
        y = arr["actions"].astype(np.float32)
        feats.append(x)
        acts.append(y)
        if feature_names is None and "feature_names" in arr:
            feature_names = [str(v) for v in arr["feature_names"].tolist()]
        if action_names is None and "action_names" in arr:
            action_names = [str(v) for v in arr["action_names"].tolist()]

    features = np.concatenate(feats, axis=0)
    actions = np.concatenate(acts, axis=0)
    return features, actions, feature_names, action_names


def split_indices(num_samples: int, val_ratio: float, seed: int):
    val_ratio = float(np.clip(val_ratio, 0.0, 0.9))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_samples)
    n_val = int(num_samples * val_ratio)
    if n_val <= 0:
        n_val = min(1024, max(1, num_samples // 20))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    if len(train_idx) == 0:
        train_idx = val_idx
    return train_idx, val_idx


def split_files(files, val_ratio: float, seed: int):
    val_ratio = float(np.clip(val_ratio, 0.0, 0.9))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(files))
    n_val = int(len(files) * val_ratio)
    if n_val <= 0:
        n_val = min(max(1, len(files) // 10), 64)
    val_ids = perm[:n_val]
    train_ids = perm[n_val:]
    if len(train_ids) == 0:
        train_ids = val_ids
    train_files = [files[i] for i in train_ids]
    val_files = [files[i] for i in val_ids]
    return train_files, val_files


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_dir = _resolve_dir(args.dataset_dir)
    output_root = _resolve_dir(args.output_dir)
    run_name = args.run_name or datetime.now().strftime("bc_%Y%m%d_%H%M%S")
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    files_all = discover_npz_files(dataset_dir=dataset_dir, max_files=args.max_files)
    files = filter_episode_files(
        files=files_all,
        only_success_episodes=bool(args.only_success_episodes),
        min_final_path_ratio=float(args.min_final_path_ratio),
    )
    if len(files) == 0:
        raise RuntimeError(
            "No files left after episode filtering. "
            "Relax --only_success_episodes / --min_final_path_ratio."
        )

    if bool(args.split_by_episode):
        train_files, val_files = split_files(files=files, val_ratio=args.val_ratio, seed=args.seed)
        train_x_np, train_y_np, feature_names, action_names = load_dataset(files=train_files)
        val_x_np, val_y_np, _f2, _a2 = load_dataset(files=val_files)
    else:
        features_np, actions_np, feature_names, action_names = load_dataset(files=files)
        finite_mask = np.isfinite(features_np).all(axis=1) & np.isfinite(actions_np).all(axis=1)
        features_np = features_np[finite_mask]
        actions_np = actions_np[finite_mask]

        if features_np.shape[0] < 16:
            raise RuntimeError(f"Not enough samples after filtering: {features_np.shape[0]}")

        train_idx, val_idx = split_indices(
            num_samples=features_np.shape[0],
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        train_x_np = features_np[train_idx]
        train_y_np = actions_np[train_idx]
        val_x_np = features_np[val_idx]
        val_y_np = actions_np[val_idx]

    finite_train = np.isfinite(train_x_np).all(axis=1) & np.isfinite(train_y_np).all(axis=1)
    finite_val = np.isfinite(val_x_np).all(axis=1) & np.isfinite(val_y_np).all(axis=1)
    train_x_np = train_x_np[finite_train]
    train_y_np = train_y_np[finite_train]
    val_x_np = val_x_np[finite_val]
    val_y_np = val_y_np[finite_val]

    if train_x_np.shape[0] < 16 or val_x_np.shape[0] < 8:
        raise RuntimeError(
            f"Not enough samples after split/filter. train={train_x_np.shape[0]} val={val_x_np.shape[0]}"
        )

    feat_mean_np = train_x_np.mean(axis=0, dtype=np.float64).astype(np.float32)
    feat_std_np = train_x_np.std(axis=0, dtype=np.float64).astype(np.float32)
    feat_std_np = np.where(feat_std_np < 1e-6, 1.0, feat_std_np)

    train_x_np = (train_x_np - feat_mean_np) / feat_std_np
    val_x_np = (val_x_np - feat_mean_np) / feat_std_np

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_x = torch.from_numpy(train_x_np).to(torch.float32)
    train_y = torch.from_numpy(train_y_np).to(torch.float32)
    val_x = torch.from_numpy(val_x_np).to(torch.float32)
    val_y = torch.from_numpy(val_y_np).to(torch.float32)

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=max(1, int(args.batch_size)),
        shuffle=True,
        drop_last=False,
        num_workers=max(0, int(args.num_workers)),
    )
    val_loader = DataLoader(
        TensorDataset(val_x, val_y),
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        drop_last=False,
        num_workers=max(0, int(args.num_workers)),
    )

    hidden_dims = parse_hidden_dims(args.hidden_dims)
    model = HighLevelMLP(
        input_dim=int(train_x.shape[1]),
        hidden_dims=hidden_dims,
        output_dim=2,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    best_val_loss = float("inf")
    best_path = run_dir / "model_best.pt"
    last_path = run_dir / "model_last.pt"
    history = []

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_loss_sum = 0.0
        train_mae_sum = 0.0
        train_count = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            pred = model(xb)
            loss_vx = F.mse_loss(pred[:, 0], yb[:, 0])
            loss_wz = F.mse_loss(pred[:, 1], yb[:, 1])
            loss = loss_vx + float(args.wz_weight) * loss_wz

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bs = int(xb.shape[0])
            train_count += bs
            train_loss_sum += float(loss.item()) * bs
            train_mae_sum += float(torch.mean(torch.abs(pred - yb)).item()) * bs

        model.eval()
        val_loss_sum = 0.0
        val_mae_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                pred = model(xb)
                loss_vx = F.mse_loss(pred[:, 0], yb[:, 0])
                loss_wz = F.mse_loss(pred[:, 1], yb[:, 1])
                loss = loss_vx + float(args.wz_weight) * loss_wz

                bs = int(xb.shape[0])
                val_count += bs
                val_loss_sum += float(loss.item()) * bs
                val_mae_sum += float(torch.mean(torch.abs(pred - yb)).item()) * bs

        train_loss = train_loss_sum / max(1, train_count)
        train_mae = train_mae_sum / max(1, train_count)
        val_loss = val_loss_sum / max(1, val_count)
        val_mae = val_mae_sum / max(1, val_count)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_mae": train_mae,
                "val_loss": val_loss,
                "val_mae": val_mae,
            }
        )
        print(
            f"[bc] epoch={epoch:03d} train_loss={train_loss:.6f} train_mae={train_mae:.6f} "
            f"val_loss={val_loss:.6f} val_mae={val_mae:.6f}"
        )

        ckpt = {
            "model_state_dict": model.state_dict(),
            "input_dim": int(train_x.shape[1]),
            "output_dim": 2,
            "hidden_dims": list(hidden_dims),
            "feature_mean": feat_mean_np,
            "feature_std": feat_std_np,
            "feature_names": feature_names or [],
            "action_names": action_names or ["vx", "wz"],
            "best_val_loss": float(min(best_val_loss, val_loss)),
            "args": vars(args),
        }
        torch.save(ckpt, last_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt["best_val_loss"] = float(best_val_loss)
            torch.save(ckpt, best_path)

    # Small inference test on first few val samples.
    with torch.no_grad():
        n_test = min(32, val_x.shape[0])
        x_test = val_x[:n_test].to(device)
        y_test = val_y[:n_test].to(device)
        y_pred = model(x_test)
        test_mae = torch.mean(torch.abs(y_pred - y_test)).item()
        test_mse = torch.mean((y_pred - y_test) ** 2).item()
    print(f"[bc] inference_test n={n_test} mse={test_mse:.6f} mae={test_mae:.6f}")

    summary = {
        "run_dir": str(run_dir),
        "dataset_dir": str(dataset_dir),
        "num_files_before_filter": len(files_all),
        "num_files_after_filter": len(files),
        "split_by_episode": bool(args.split_by_episode),
        "train_samples": int(train_x.shape[0]),
        "val_samples": int(val_x.shape[0]),
        "input_dim": int(train_x.shape[1]),
        "hidden_dims": list(hidden_dims),
        "best_val_loss": float(best_val_loss),
        "inference_test_mse": float(test_mse),
        "inference_test_mae": float(test_mae),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"[bc] saved best model: {best_path}")
    print(f"[bc] saved summary:    {run_dir / 'summary.json'}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
