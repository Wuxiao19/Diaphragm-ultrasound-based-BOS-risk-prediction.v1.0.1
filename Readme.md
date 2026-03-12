# Diaphragm Ultrasound Analysis System

A deep learning-based system for BOS (Bronchiolitis Obliterans Syndrome) risk prediction using diaphragm B-mode and M-mode ultrasound images.

## Features

- **Automated Risk Assessment**: Predicts BOS risk probability from ultrasound image pairs
- **Multi-Modal Fusion**: Combines B-mode and M-mode ultrasound features
- **AI Agent Integration**: LLM-powered analysis with natural language reports
- **Web Interface**: User-friendly Streamlit application
- **Batch Processing**: Supports single patient or multiple patients analysis

## Architecture

The system uses a 4-stage pipeline:

1. **Feature Extraction**: MIAFEx (ViT-based) extracts 768-dimensional features from each modality
2. **Feature Reduction**: PSO-optimized feature selection reduces dimensionality
3. **Feature Fusion**: Merges B-mode and M-mode features by patient ID and exam date
4. **Classification**: 5-fold ExtraTrees ensemble predicts risk probability (0-1)

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, for faster inference)

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Run Web Application

```bash
streamlit run streamlit_app.py
```

Open your browser and navigate to `http://localhost:8501`

### 2. Configure LLM Agent (Optional)

Create `.streamlit/secrets.toml`:

```toml
llm_api_key = "your-api-key-here"
```

Or set environment variables:

```bash
export LLM_BASE_URL="https://api.siliconflow.cn/v1"
export LLM_MODEL="Qwen/Qwen3-8B"
```

## Usage

### Image Naming Convention

**IMPORTANT**: All image filenames must follow the pattern `YY-MM-DD-<ID>_xxx.png`

Examples:
- `24-05-01-P001_b_mode.png` (B-mode image for patient P001 on May 1, 2024)
- `24-05-01-P001_m_mode.png` (M-mode image for the same patient)

The system matches B-mode and M-mode images by date and patient ID.

### Single Patient Analysis

1. Select "Single patient" mode
2. Upload one B-mode image and one M-mode image
3. Click "Run LLM Agent" for AI-powered analysis

### Batch Analysis

1. Select "Batch patients" mode
2. Upload multiple B-mode images
3. Upload multiple M-mode images
4. Click "Run LLM Agent" for comprehensive analysis

## Model Checkpoints

Checkpoints are automatically downloaded from Hugging Face on first run:

**Repository**: `Wuxiao19/Diaphragm-ultrasound-based-BOS-risk-prediction`

Expected structure:
```
checkpoint/
├── MIAFEx/
│   ├── B_model/miafex_checkpoint.pth
│   └── M_model/miafex_checkpoint.pth
├── PSO/
│   ├── B_model/selected_features_idx.txt
│   └── M_model/selected_features_idx.txt
└── ExtraTrees/
    ├── et_model_fold_0.pkl to et_model_fold_4.pkl
    └── et_feature_order.pkl
```

## Output

Results are saved to `detect/runX/`:

- `detect_result.csv`: Final predictions with risk probabilities
- `merged_features.csv`: Fused B/M features
- `missing_modality_samples.csv`: Samples with incomplete B/M pairs

## MCP Tools

The system exposes FastMCP tools for LLM agent integration:

### `detect_single_pair`
Analyzes a single B-mode and M-mode image pair.

**Parameters**:
- `b_image_path`: Path to B-mode image
- `m_image_path`: Path to M-mode image

### `detect_batch_folders`
Batch analysis for multiple patients.

**Parameters**:
- `b_folder_path`: Folder containing B-mode images
- `m_folder_path`: Folder containing M-mode images

Run MCP server standalone:
```bash
python mcp_tools.py
```

## Technical Details

### Models
- **MIAFEx**: Vision Transformer (ViT-base-patch16-224) with refinement weights
- **Feature Selection**: PSO (Particle Swarm Optimization)
- **Classifier**: ExtraTrees ensemble (5-fold cross-validation)

### Risk Interpretation
- **0.0 - 0.3**: Low risk
- **0.3 - 0.6**: Medium risk
- **0.6 - 1.0**: High risk (requires clinical attention)

## Troubleshooting

### No matched B/M pairs found
Ensure filenames follow `YY-MM-DD-<ID>` pattern with matching date and patient ID.

### Checkpoint files missing
Install `huggingface_hub` for auto-download:
```bash
pip install huggingface_hub
```

### LLM Agent not working
Check API key configuration in `.streamlit/secrets.toml` or environment variables.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{diaphragm_ultrasound_bos,
  title={Diaphragm Ultrasound Analysis System for BOS Risk Prediction},
  author={AlMSLab},
  year={2024},
  url={https://github.com/yourusername/yourrepo}
}
```

## License

This project is for research purposes only and cannot replace professional medical diagnosis.

## Acknowledgments

Developed by **AlMSLab**

---

**Disclaimer**: This is a machine learning model for research purposes. Results should not be used as the sole basis for clinical decisions. Always consult qualified healthcare professionals.
