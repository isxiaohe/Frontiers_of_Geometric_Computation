"""Prepare one ShapeNetCore category for the local point-cloud loader.

The Hugging Face ShapeNetCore repository is gated. Run `huggingface-cli login`
and accept the dataset terms before using `--download`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from fgc3d.shapenetcore import category_to_synset, convert_category_to_pc15k


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="chair")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--num-points", type=int, default=15000)
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("data/ShapeNetCore.v2.PC15k"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--repo-id", default="ShapeNet/ShapeNetCore")
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser


def _download_category(*, repo_id: str, category: str, cache_dir: Path | None, max_models: int | None) -> Path:
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError as error:
        raise RuntimeError("install huggingface_hub or run with --raw-root") from error

    synset = category_to_synset(category)
    if max_models is not None:
        files = list_repo_files(repo_id=repo_id, repo_type="dataset")
        obj_files = sorted(
            file
            for file in files
            if file.startswith(f"{synset}/") and file.endswith("/models/model_normalized.obj")
        )[:max_models]
        selected_root = (cache_dir or Path(".cache")) / "shapenetcore_selected" / repo_id.replace("/", "__")
        if obj_files:
            for repo_file in obj_files:
                local_file = Path(
                    hf_hub_download(
                        repo_id=repo_id,
                        repo_type="dataset",
                        filename=repo_file,
                        cache_dir=str(cache_dir) if cache_dir is not None else None,
                    )
                )
                destination = selected_root / repo_file
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(local_file, destination)
            return selected_root

        archive_name = f"{synset}.zip"
        if archive_name in files:
            archive_path = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=archive_name,
                    cache_dir=str(cache_dir) if cache_dir is not None else None,
                )
            )
            selected_root = (cache_dir or Path(".cache")) / "shapenetcore_selected" / repo_id.replace("/", "__")
            _extract_limited_objs(archive_path=archive_path, selected_root=selected_root, synset=synset, max_models=max_models)
            return selected_root

        raise FileNotFoundError(f"no OBJ files or archive found for {category} ({synset})")

    archive_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=f"{synset}.zip",
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
    )
    selected_root = (cache_dir or Path(".cache")) / "shapenetcore_selected" / repo_id.replace("/", "__")
    _extract_limited_objs(archive_path=archive_path, selected_root=selected_root, synset=synset, max_models=None)
    return selected_root


def _extract_limited_objs(
    *,
    archive_path: Path,
    selected_root: Path,
    synset: str,
    max_models: int | None,
) -> None:
    extracted = 0
    with zipfile.ZipFile(archive_path) as archive:
        obj_names = sorted(name for name in archive.namelist() if name.endswith(".obj") and f"{synset}/" in name)
        for member in obj_names:
            parts = Path(member).parts
            synset_index = parts.index(synset)
            relative = Path(*parts[synset_index:])
            destination = selected_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            extracted += 1
            if max_models is not None and extracted >= max_models:
                break
    if extracted == 0:
        raise FileNotFoundError(f"no OBJ files for {synset} in archive: {archive_path}")


def main() -> None:
    args = build_parser().parse_args()
    if args.raw_root is None:
        if not args.download:
            raise SystemExit("provide --raw-root or pass --download after Hugging Face access is configured")
        raw_root = _download_category(
            repo_id=args.repo_id,
            category=args.category,
            cache_dir=args.cache_dir,
            max_models=args.max_models,
        )
    else:
        raw_root = args.raw_root

    written = convert_category_to_pc15k(
        raw_root=raw_root,
        output_root=args.output_root,
        category=args.category,
        split=args.split,
        num_points=args.num_points,
        max_models=args.max_models,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "category": args.category,
                "synset": category_to_synset(args.category),
                "raw_root": str(raw_root),
                "output_root": str(args.output_root),
                "split": args.split,
                "num_points": args.num_points,
                "num_models": len(written),
                "first_files": [str(path) for path in written[:5]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
