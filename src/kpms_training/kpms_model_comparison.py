"""Calculates the expected marginal likelihood scores between multiple KPMS models.

This script computes the EML scores across multiple different models. See 
https://keypoint-moseq.readthedocs.io/en/latest/advanced.html#selecting-a-model for code to plot the
computed scores.

Usage:
    python multi_model_selection.py \
        --project_name <project_name> \
        --model_basename <model_basename> \
        --kpms_dir <path_to_kpms_projects> \
        --num_models <num_models> \
        --result_path <result_path>
"""

import argparse
import json
import pandas as pd

from pathlib import Path

import keypoint_moseq as kpms

def main(
    project_name: str,
    model_basename: str,
    kpms_dir: str,
    num_models: int,
    result_path: str,
):
    kpms_dir = Path(kpms_dir)
    project_dir = kpms_dir / project_name
    result_path = Path(result_path)

    model_names = [f"{model_basename}{i}" for i in range(1, num_models+1)]
    eml_scores, eml_std_errs = kpms.expected_marginal_likelihoods(project_dir, model_names)

    data = {
        "models": model_names,
        "eml_scores": eml_scores.tolist(),
        "eml_std_errs": eml_std_errs.tolist()
    }
    print(pd.DataFrame(data))

    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w") as f:
        json.dump(data, f, indent=2)
    print(f"comparison of {num_models} saved to {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input parameters for Keypoint-MoSeq model selection")

    parser.add_argument("--project_name", type=str, required=True,
                        help="Name of the keypoint-MoSeq project")
    parser.add_argument("--model_basename", type=str, required=True,
                        help="Basename of keypoint-MoSeq model (e.g. model- generates model-1, model-2, ...)")
    parser.add_argument("--kpms_dir", type=str, required=True,
                        help="Path of the keypoint-MoSeq project directory")
    parser.add_argument("--num_models", type=int, required=True,
                        help="Number of keypoint-MoSeq models")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the result JSON to")

    args = parser.parse_args()

    print("\n--- RUN CONFIG ---")
    for k, v in vars(args).items():
        print(f"{k:20}: {v}")
    print("------------------\n")

    main(args.project_name, args.model_basename, args.kpms_dir, args.num_models, args.result_path)
