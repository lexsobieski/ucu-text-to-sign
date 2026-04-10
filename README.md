# Ukrainian Text to Sign Language Production

Master's diploma project: generating Ukrainian Sign Language (USL) pose sequences from Ukrainian text.

## Repository Structure

```
data/
  usl-suspilne/                 # dataset
    annotations.csv             # name|text|annotator
    features/                   # cropped signer videos (510x510)
    poses/2d/                   # MediaPipe 50-joint, (T, 150) npy
    poses/3d/                   # 3D lifted poses, (T, 150) npy
  firebase/                     # timestamped Firebase RTDB exports
  cache/                        # intermediates (raw_videos, uncropped, splits.csv)

scripts/                        # shared dataset pipeline
  download_firebase.py          # fetch Firebase RTDB export
  build_annotations_from_firebase.py  # export → annotations.csv + splits.csv
  download_videos.py            # download YouTube videos via yt-dlp
  split_videos.py               # split into sentence-level clips
  crop_signer.py                # crop to signer region (bottom-right)
  extract_poses.py              # MediaPipe 50-joint pose extraction
  lift_to_3d.py                 # 2D → 3D pose lifting
  visualize_poses.py            # skeleton overlay on video

experiments/                    # model experiments (each self-contained)
  progressive_transformers/
    prepare_data.py             # dataset → .text/.skels/.files
    prepare.ipynb               # pose extraction + data preparation
    train.ipynb                 # training notebook
    model/                      # patched Progressive Transformers fork

notebooks/
  build_dataset.ipynb           # end-to-end dataset build pipeline
  analysis/                     # dataset analysis notebooks
  data-prep/                    # labeling utilities (split_video, transcribe)
```

## Dataset: USL-Suspilne

- Annotated clips from Suspilne (Ukrainian public broadcasting) YouTube videos
- Annotations created via a custom Gradio labeling app backed by Firebase RTDB
- Pipeline: Firebase export → annotations → download → split → crop → extract poses → lift 3D

## Experiments

### Progressive Transformers

Text-to-pose generation based on [Saunders et al., 2020](https://arxiv.org/abs/2004.14874).
Fork with torchtext removed and rewritten for modern PyTorch.

### Adding new experiments

Create `experiments/{name}/` with its own `prepare_data.py` and notebooks.
Shared dataset in `data/usl-suspilne/` feeds all experiments.
Scripts in `scripts/` expose importable functions with explicit path parameters — no hardcoded defaults.
