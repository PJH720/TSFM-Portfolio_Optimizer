from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable

from huggingface_hub import snapshot_download

MODEL_MAP: dict[str, str] = {
    "chronos2": "amazon/chronos-2",
    "timesfm1": "google/timesfm-1.0-200m-pytorch",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preload HF models to local cache")
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_MAP.keys()),
        action="append",
        default=[],
        help="Model key to preload. Can be specified multiple times.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Preload all supported models.",
    )
    parser.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="Environment variable name for HF token.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional cache directory override (defaults to HF_HOME/hub).",
    )
    return parser.parse_args()


def resolve_targets(selected: list[str], all_flag: bool) -> list[str]:
    if all_flag:
        return list(MODEL_MAP.values())
    if selected:
        return [MODEL_MAP[key] for key in selected]
    raise ValueError("Choose --all or at least one --model.")


def preload_models(
    model_ids: Iterable[str],
    token: str | None,
    cache_dir: str | None,
) -> tuple[list[str], list[tuple[str, str]]]:
    success: list[str] = []
    failed: list[tuple[str, str]] = []
    for model_id in model_ids:
        try:
            path = snapshot_download(
                repo_id=model_id,
                token=token,
                cache_dir=cache_dir,
                local_files_only=False,
            )
            success.append(f"{model_id} -> {path}")
        except Exception as exc:
            failed.append((model_id, str(exc)))
    return success, failed


def main() -> int:
    args = parse_args()
    try:
        targets = resolve_targets(args.model, args.all)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 2

    token = os.getenv(args.token_env, "").strip() or None
    if token is None:
        print(
            f"[WARN] No token found in {args.token_env}. "
            "Public access only; private/gated models will fail."
        )

    print(f"[INFO] Targets: {', '.join(targets)}")
    print(f"[INFO] Cache dir: {args.cache_dir or 'default'}")
    success, failed = preload_models(targets, token, args.cache_dir)

    if success:
        print("[OK] Preloaded models:")
        for item in success:
            print(f"  - {item}")
    if failed:
        print("[FAIL] Failed models:")
        for model_id, message in failed:
            print(f"  - {model_id}: {message}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
