# -*- coding: utf-8 -*-
"""
Full pipeline: feature extraction -> feature reduction -> feature fusion ->
ExtraTrees classification -> save results.

Functions:
    1. Support single-image input (one B-mode image and one M-mode image)
    2. Support folder input (one folder of B-mode images and one folder of M-mode images)
    3. Automatically perform feature extraction, reduction, fusion and classification
    4. Save results into detect/runX directory
"""

import os
import sys
import re
import shutil
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import ViTForImageClassification
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier
from tqdm import tqdm
from typing import Optional

try:
    # Optional dependency: used to auto-download checkpoints from Hugging Face
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None

# ===================================================================
# Constants
# ===================================================================

# Path configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_MIAFEX_B = os.path.join(BASE_DIR, "checkpoint", "MIAFEx", "B_model", "miafex_checkpoint.pth")
CHECKPOINT_MIAFEX_M = os.path.join(BASE_DIR, "checkpoint", "MIAFEx", "M_model", "miafex_checkpoint.pth")
CHECKPOINT_PSO_B = os.path.join(BASE_DIR, "checkpoint", "PSO", "B_model", "selected_features_idx.txt")
CHECKPOINT_PSO_M = os.path.join(BASE_DIR, "checkpoint", "PSO", "M_model", "selected_features_idx.txt")
CHECKPOINT_RF_DIR = os.path.join(BASE_DIR, "checkpoint", "ExtraTrees")
DETECT_OUTPUT_DIR = os.path.join(BASE_DIR, "detect")

# Hugging Face checkpoint repo (folder "checkpoint/" is stored in the repo)
HF_REPO_ID = "Wuxiao19/Diaphragm-ultrasound-based-BOS-risk-prediction"
HF_CHECKPOINT_SUBDIR = "checkpoint"

# Model parameters
BATCH_SIZE = 16
FALLBACK_NUM_CLASSES = 2
IMAGE_SIZE = 224

# ===================================================================
# MIAFEx model definition
# ===================================================================

class MIAFEx(nn.Module):
    """MIAFEx model: ViT-based feature extractor"""
    def __init__(self, num_classes: int):
        super().__init__()
        self.vit = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224-in21k',
            output_hidden_states=True
        )
        self.fc = nn.Linear(768, num_classes)
        self.refinement_weights = nn.Parameter(torch.randn(768))

    def forward(self, x):
        out = self.vit(x)
        if out.hidden_states is None:
            raise ValueError("Hidden states not returned.")
        cls_features = out.hidden_states[-1][:, 0, :]
        refined = cls_features * self.refinement_weights.view(1, -1)
        return refined, cls_features


def safe_load_ckpt(path, device):
    """Safely load checkpoint file"""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_backbone_and_refinement(model: MIAFEx, ckpt: dict, device):
    """
    Load backbone and refinement weights in a safe and consistent way.
    """

    # 1. Prefer full model_state_dict if available
    if 'model_state_dict' in ckpt:
        missing, unexpected = model.load_state_dict(
            ckpt['model_state_dict'],
            strict=False
        )
    elif 'vit_state_dict' in ckpt:
        missing, unexpected = model.vit.load_state_dict(
            ckpt['vit_state_dict'],
            strict=False
        )
    else:
        raise KeyError("Checkpoint missing model weights.")

    if missing:
        # 写到 stderr，避免污染 MCP stdio 的 JSON-RPC 通道
        print(f"[extract] Warning: missing keys: {missing}", file=sys.stderr, flush=True)
    if unexpected:
        print(f"[extract] Warning: unexpected keys: {unexpected}", file=sys.stderr, flush=True)

    # 2. Load refinement_weights (same as original logic)
    if 'refinement_weights' in ckpt:
        with torch.no_grad():
            rw = ckpt['refinement_weights'].to(device)
            if rw.numel() != model.refinement_weights.numel():
                raise ValueError("refinement_weights size mismatch.")
            model.refinement_weights.copy_(rw)

    # 3. Return num_classes (keep original behavior)
    if 'num_classes' in ckpt:
        return int(ckpt['num_classes'])

    return FALLBACK_NUM_CLASSES



# ===================================================================
# Custom dataset
# ===================================================================

class ImageDataset(Dataset):
    """Dataset for single image or image list (folder)"""
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths if isinstance(image_paths, list) else [image_paths]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, os.path.basename(img_path)
        except Exception as e:
            # 写到 stderr，避免污染 MCP stdio 的 JSON-RPC 通道
            print(f"Error loading image {img_path}: {e}", file=sys.stderr, flush=True)
            image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color='black')
            if self.transform:
                image = self.transform(image)
            return image, os.path.basename(img_path)


# ===================================================================
# Core pipeline
# ===================================================================
class DetectionPipeline:
    """End-to-end detection pipeline"""

    def __init__(self, gui_callback=None):
        # gui_callback is used to send logs to external GUI / web UI
        self.gui_callback = gui_callback
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models: dict[str, Any] = {}
        self.selected_features: dict[str, list[int]] = {}

    def log(self, message: str) -> None:
        """Log message to stderr and optional GUI callback.

        注意：在 MCP stdio 模式下，stdout 必须只输出 JSON-RPC，
        所以所有人类可读日志都写到 stderr。
        """
        print(f"[Pipeline] {message}", file=sys.stderr, flush=True)
        if self.gui_callback:
            self.gui_callback(f"[Pipeline] {message}\n")

    def load_models(self) -> None:
        """Load MIAFEx models and ExtraTrees models."""
        self.log("Loading models...")

        # Ensure checkpoints exist locally (download from Hugging Face if needed)
        ensure_checkpoints_available(gui_log=self.log)

        # Load MIAFEx models
        for model_type in ["B", "M"]:
            checkpoint_path = CHECKPOINT_MIAFEX_B if model_type == "B" else CHECKPOINT_MIAFEX_M
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"MIAFEx checkpoint not found: {checkpoint_path}")

            ckpt = safe_load_ckpt(checkpoint_path, self.device)
            model = MIAFEx(FALLBACK_NUM_CLASSES).to(self.device)
            load_backbone_and_refinement(model, ckpt, self.device)
            model.eval()
            self.models[f"miafex_{model_type}"] = model
            self.log(f"MIAFEx {model_type} model loaded")

        # Load ExtraTrees models (5 folds)
        self.models["et"] = []
        for fold in range(5):
            et_path = os.path.join(CHECKPOINT_RF_DIR, f"et_model_fold_{fold}.pkl")
            if not os.path.exists(et_path):
                raise FileNotFoundError(f"ExtraTrees model not found: {et_path}")
            with open(et_path, "rb") as f:
                et_model = pickle.load(f)
            self.models["et"].append(et_model)
        self.log(f"Loaded {len(self.models['et'])} ExtraTrees models")

        # Load ExtraTrees feature order used during training (critical)
        feature_order_path = os.path.join(CHECKPOINT_RF_DIR, "et_feature_order.pkl")
        if not os.path.exists(feature_order_path):
            raise FileNotFoundError(
                f"ExtraTrees feature order file not found: {feature_order_path}"
            )

        with open(feature_order_path, "rb") as f:
            self.et_feature_order = pickle.load(f)

        self.log(f"Loaded ExtraTrees feature order, dim={len(self.et_feature_order)}")

        # Load feature selection indices (PSO)
        for model_type in ["B", "M"]:
            pso_path = CHECKPOINT_PSO_B if model_type == "B" else CHECKPOINT_PSO_M
            if not os.path.exists(pso_path):
                raise FileNotFoundError(f"PSO index file not found: {pso_path}")
            with open(pso_path, "r", encoding="utf-8") as f:
                selected_idx = [int(line.strip()) for line in f if line.strip().isdigit()]
            self.selected_features[model_type] = selected_idx
            self.log(f"Loaded {len(selected_idx)} selected features for {model_type} model")

    def extract_features(self, image_paths, model_type: str) -> tuple[np.ndarray, list[str]]:
        """Extract features from images."""
        self.log(f"Start extracting {model_type}-mode image features...")

        model = self.models[f"miafex_{model_type}"]
        transform = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
            ]
        )

        dataset = ImageDataset(image_paths, transform=transform)
        dataloader = DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        )

        all_features: list[np.ndarray] = []
        all_filenames: list[str] = []

        with torch.no_grad():
            for images, filenames in tqdm(dataloader, desc=f"Extracting {model_type}"):
                images = images.to(self.device)
                refined, _ = model(images)
                all_features.append(refined.cpu().numpy())
                all_filenames.extend(filenames)

        features = (
            np.vstack(all_features) if all_features else np.zeros((0, 768), dtype=np.float32)
        )
        self.log(f"Feature extraction done: {len(all_filenames)} images")
        return features, all_filenames

    def reduce_features(
        self, features: np.ndarray, filenames: list[str], model_type: str
    ) -> pd.DataFrame:
        """Feature reduction using selected indices."""
        self.log(f"Start feature reduction for {model_type}-mode features...")

        selected_idx = self.selected_features[model_type]
        selected_features = features[:, selected_idx]

        feature_cols = [f"f_{i}" for i in selected_idx]
        df = pd.DataFrame(selected_features, columns=feature_cols)
        df.insert(0, "filename", filenames)

        self.log(
            f"Feature reduction done: {features.shape[1]} -> {selected_features.shape[1]} dims"
        )
        return df

    def reduce_features_from_df(self, df_raw: pd.DataFrame, model_type: str) -> pd.DataFrame:
        """Reduce features from raw feature CSV (requires filename + f_0... columns)."""
        if "filename" not in df_raw.columns:
            raise ValueError(f"{model_type} CSV missing 'filename' column")
        selected_idx = self.selected_features[model_type]
        feature_cols = [f"f_{i}" for i in selected_idx]
        missing = [c for c in feature_cols if c not in df_raw.columns]
        if missing:
            raise ValueError(
                f"{model_type} CSV missing feature columns {missing[:5]} (total {len(missing)})"
            )
        selected = df_raw[feature_cols].values
        df = pd.DataFrame(selected, columns=feature_cols)
        df.insert(0, "filename", df_raw["filename"].values)
        self.log(
            f"CSV feature reduction done: original {df_raw.shape[1]} columns -> {len(feature_cols)} selected"
        )
        return df

    def merge_features(self, df_b: pd.DataFrame, df_m: pd.DataFrame) -> pd.DataFrame:
        """Merge B- and M-mode features by (date, patient_id) parsed from filenames."""
        self.log("Start merging B- and M-mode features...")

        def extract_date_pid(name: str) -> tuple[str | None, str | None]:
            m = re.match(r"(\d{2}-\d{2}-\d{2})-(C\d{3}|B\d{3}|P\d{3})", str(name))
            if m:
                return m.group(1), m.group(2)
            return None, None

        df_b[["date", "pid"]] = df_b["filename"].apply(
            lambda x: pd.Series(extract_date_pid(x))
        )
        df_m[["date", "pid"]] = df_m["filename"].apply(
            lambda x: pd.Series(extract_date_pid(x))
        )

        b_keys = set(zip(df_b["date"], df_b["pid"]))
        m_keys = set(zip(df_m["date"], df_m["pid"]))

        only_b = b_keys - m_keys
        only_m = m_keys - b_keys

        missing_records: list[dict[str, str]] = []
        for d, p in sorted(only_b):
            missing_records.append(
                {"date": d, "pid": p, "missing_modality": "missing M modality"}
            )
        for d, p in sorted(only_m):
            missing_records.append(
                {"date": d, "pid": p, "missing_modality": "missing B modality"}
            )

        self.missing_modalities = missing_records

        b_features = [c for c in df_b.columns if c not in ["filename", "date", "pid"]]
        m_features = [c for c in df_m.columns if c not in ["filename", "date", "pid"]]

        merged_rows: list[list[float]] = []
        merged_filenames: list[dict[str, str]] = []

        for _, row_m in df_m.iterrows():
            m_date = row_m["date"]
            m_pid = row_m["pid"]
            m_filename = row_m["filename"]

            candidates = df_b[(df_b["date"] == m_date) & (df_b["pid"] == m_pid)]
            for _, row_b in candidates.iterrows():
                b_filename = row_b["filename"]
                merged_filename = f"{m_date}-{m_pid}"
                merged_row = list(row_m[m_features]) + list(row_b[b_features])
                merged_rows.append(merged_row)
                merged_filenames.append(
                    {
                        "merged_filename": merged_filename,
                        "b_filename": b_filename,
                        "m_filename": m_filename,
                    }
                )

        if not merged_rows:
            raise ValueError("No matched B/M pairs found for merging")

        merged_df = pd.DataFrame(
            merged_rows,
            columns=[f"M_{c}" for c in m_features] + [f"B_{c}" for c in b_features],
        )
        filename_df = pd.DataFrame(merged_filenames)
        merged_df = pd.concat([filename_df, merged_df], axis=1)

        self.log(f"Merging done: {len(merged_df)} matched samples")
        return merged_df

    def predict(self, merged_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Run ExtraTrees-based prediction."""
        self.log("Start classification prediction...")

        missing_cols = [c for c in self.et_feature_order if c not in merged_df.columns]
        if missing_cols:
            raise RuntimeError(
                f"Inference features missing {len(missing_cols)} columns, e.g. {missing_cols[:5]}"
            )

        X = merged_df[self.et_feature_order].values
        all_probas: list[np.ndarray] = []

        for i, et_model in enumerate(self.models["et"]):
            classes = et_model.classes_
            sick_label = 1
            if sick_label not in classes:
                raise RuntimeError(
                    f"ExtraTrees fold {i} does not contain positive label {sick_label}, classes_={classes}"
                )
            sick_index = list(classes).index(sick_label)
            proba = et_model.predict_proba(X)[:, sick_index]
            all_probas.append(proba)

        avg_proba = np.mean(all_probas, axis=0)
        predictions = (avg_proba >= 0.5).astype(int)

        self.log(f"Prediction done: {len(predictions)} samples")
        return avg_proba, predictions

    def save_results(
        self,
        merged_df: pd.DataFrame,
        probabilities: np.ndarray,
        predictions: np.ndarray,
        output_dir: str,
    ) -> tuple[str, pd.DataFrame]:
        """Save prediction results and intermediate files."""
        self.log(f"Saving results to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)

        results_df = pd.DataFrame(
            {
                "merged_filename": merged_df["merged_filename"].values,
                "b_filename": merged_df["b_filename"].values,
                "m_filename": merged_df["m_filename"].values,
                "risk_probability": probabilities,
                "prediction": predictions,
                "prediction_label": [
                    "diseased" if p == 1 else "healthy" for p in predictions
                ],
            }
        )

        if len(results_df) > 1:
            group_key = "merged_filename"
            if results_df[group_key].duplicated().any():
                self.log(
                    "Found multiple rows with the same date and patient_id, start aggregating..."
                )
                aggregated_rows: list[dict[str, Any]] = []
                for name, group in results_df.groupby(group_key):
                    avg_proba = group["risk_probability"].mean()
                    mode_pred = group["prediction"].mode().iloc[0]
                    mode_label = "diseased" if mode_pred == 1 else "healthy"
                    aggregated_rows.append(
                        {
                            "merged_filename": name,
                            "b_filename": ";".join(
                                group["b_filename"].astype(str).unique()
                            ),
                            "m_filename": ";".join(
                                group["m_filename"].astype(str).unique()
                            ),
                            "risk_probability": avg_proba,
                            "prediction": mode_pred,
                            "prediction_label": mode_label,
                        }
                    )
                results_df = pd.DataFrame(aggregated_rows)
                self.log(
                    f"Aggregation done: {len(results_df)} unique date-patient_id samples"
                )

        if hasattr(self, "missing_modalities") and self.missing_modalities:
            missing_df = pd.DataFrame(self.missing_modalities)
            missing_csv_path = os.path.join(
                output_dir, "missing_modality_samples.csv"
            )
            missing_df.to_csv(missing_csv_path, index=False, encoding="utf-8-sig")
            self.log(
                f"Found {len(missing_df)} samples missing B or M modality; "
                f"saved to: {missing_csv_path}"
            )
        else:
            self.log("All samples have both B and M modalities, no missing-modality samples")

        result_csv_path = os.path.join(output_dir, "detect_result.csv")
        results_df.to_csv(result_csv_path, index=False, encoding="utf-8-sig")
        merged_df.to_csv(os.path.join(output_dir, "merged_features.csv"), index=False)

        self.log(f"Results saved: {result_csv_path}")
        return result_csv_path, results_df

    def run(self, b_input: str, m_input: str, is_folder: bool, use_csv: bool):
        """Run full pipeline."""
        try:
            run_num = self._get_next_run_number()
            output_dir = os.path.join(DETECT_OUTPUT_DIR, f"run{run_num}")

            self.log("=" * 60)
            self.log("Start detection pipeline")
            self.log(f"Output directory: {output_dir}")
            self.log("=" * 60)

            if use_csv:
                self.log("Using feature-CSV mode")
                df_b_raw = pd.read_csv(b_input)
                df_m_raw = pd.read_csv(m_input)
                df_b = self.reduce_features_from_df(df_b_raw, "B")
                df_m = self.reduce_features_from_df(df_m_raw, "M")
                self.log(
                    f"B CSV samples: {len(df_b_raw)}, M CSV samples: {len(df_m_raw)}"
                )
            else:
                if is_folder:
                    b_paths = sorted(
                        [
                            os.path.join(b_input, f)
                            for f in os.listdir(b_input)
                            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
                        ]
                    )
                    m_paths = sorted(
                        [
                            os.path.join(m_input, f)
                            for f in os.listdir(m_input)
                            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
                        ]
                    )
                else:
                    b_paths = [b_input]
                    m_paths = [m_input]

                self.log(f"B-mode image count: {len(b_paths)}")
                self.log(f"M-mode image count: {len(m_paths)}")

                b_features, b_filenames = self.extract_features(b_paths, "B")
                m_features, m_filenames = self.extract_features(m_paths, "M")

                df_b = self.reduce_features(b_features, b_filenames, "B")
                df_m = self.reduce_features(m_features, m_filenames, "M")

            os.makedirs(output_dir, exist_ok=True)
            df_b.to_csv(os.path.join(output_dir, "b_features_reduced.csv"), index=False)
            df_m.to_csv(os.path.join(output_dir, "m_features_reduced.csv"), index=False)

            merged_df = self.merge_features(df_b, df_m)
            probabilities, predictions = self.predict(merged_df)
            _, results_df = self.save_results(
                merged_df, probabilities, predictions, output_dir
            )

            self.log("=" * 60)
            self.log("Detection finished!")
            self.log(f"Results saved in: {output_dir}")
            self.log("=" * 60)

            return output_dir, results_df

        except Exception as e:
            error_msg = f"Error occurred during processing: {str(e)}"
            self.log(error_msg)
            import traceback

            traceback.print_exc()
            raise Exception(error_msg)

    def _get_next_run_number(self) -> int:
        """Get next available run number."""
        if not os.path.exists(DETECT_OUTPUT_DIR):
            return 1

        existing_runs: list[int] = []
        for item in os.listdir(DETECT_OUTPUT_DIR):
            if item.startswith("run") and os.path.isdir(os.path.join(DETECT_OUTPUT_DIR, item)):
                try:
                    run_num = int(item[3:])
                    existing_runs.append(run_num)
                except ValueError:
                    continue

        if not existing_runs:
            return 1
        return max(existing_runs) + 1


def _checkpoint_paths() -> list[str]:
    """Return a list of checkpoint paths that must exist locally."""
    paths = [
        CHECKPOINT_MIAFEX_B,
        CHECKPOINT_MIAFEX_M,
        CHECKPOINT_PSO_B,
        CHECKPOINT_PSO_M,
        os.path.join(CHECKPOINT_RF_DIR, "et_feature_order.pkl"),
    ]
    for fold in range(5):
        paths.append(os.path.join(CHECKPOINT_RF_DIR, f"et_model_fold_{fold}.pkl"))
    return paths


def ensure_checkpoints_available(
    hf_repo_id: str = HF_REPO_ID,
    hf_subdir: str = HF_CHECKPOINT_SUBDIR,
    local_checkpoint_dir: str = os.path.join(BASE_DIR, "checkpoint"),
    gui_log: Optional[callable] = None,
) -> None:
    """
    Ensure the local checkpoint directory exists.

    - If required checkpoint files are missing and huggingface_hub is installed,
      download repo subfolder `checkpoint/` from Hugging Face into local `checkpoint/`.
    - If still missing after download, raise FileNotFoundError.

    Notes:
    - For private repos, set environment variable HF_TOKEN before running.
    """
    required = _checkpoint_paths()
    missing_before = [p for p in required if not os.path.exists(p)]
    if not missing_before:
        return

    if gui_log:
        gui_log(
            "Checkpoint files are missing locally. Attempting to download from Hugging Face..."
        )

    if snapshot_download is None:
        raise FileNotFoundError(
            "Checkpoint files are missing and huggingface_hub is not installed. "
            "Please install huggingface_hub or place 'checkpoint/' folder in the project root."
        )

    os.makedirs(local_checkpoint_dir, exist_ok=True)

    snapshot_download(
        repo_id=hf_repo_id,
        allow_patterns=[f"{hf_subdir}/**"],
        local_dir=BASE_DIR,
        local_dir_use_symlinks=False,
        token=os.environ.get("HF_TOKEN"),
    )

    missing_after = [p for p in required if not os.path.exists(p)]
    if missing_after:
        raise FileNotFoundError(
            "Checkpoint download finished but some required files are still missing, e.g. "
            f"{missing_after[:3]}"
        )

    if gui_log:
        gui_log("Checkpoint download completed.")
