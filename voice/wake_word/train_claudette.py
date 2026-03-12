#!/usr/bin/env python3
"""
Claudette Home — Wake Word Model Trainer
Trains an openWakeWord ONNX model for the 'Claudette' wake word.

Uses locally generated training data from generate_training_data.py.
No Jupyter, no Colab, no internet required after dependencies are installed.

Usage:
    python3 train_claudette.py                   # full training run
    python3 train_claudette.py --quick           # quick 500-step test run
    python3 train_claudette.py --steps 2000      # custom step count
    python3 train_claudette.py --test            # dry run (validate data, skip training)

Output:
    voice/wake_word/models/claudette.onnx        # final ONNX model
    voice/wake_word/models/training_log.json     # loss/recall history

Requirements:
    pip install openwakeword==0.6.0 torch torchaudio torchinfo pronouncing acoustics audiomentations torch-audiomentations
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent.resolve()

TRAINING_DATA_DIR = SCRIPT_DIR / "training_data"
MODELS_DIR = SCRIPT_DIR / "models"
OUTPUT_MODEL_PATH = MODELS_DIR / "claudette.onnx"
TRAINING_LOG_PATH = MODELS_DIR / "training_log.json"

# Training hyperparameters (tuned for a ~400-sample synthetic dataset)
DEFAULT_STEPS = 3000
QUICK_STEPS = 500
WARMUP_STEPS = 50
HOLD_STEPS = 100
LEARNING_RATE = 0.0001
BATCH_SIZE = 128

# openWakeWord window: ~1.3 seconds at 16kHz = 16 feature frames × 96 dims
# Each embedding = 0.08s, so 16 embeddings = 1.28s look-back window
WINDOW_FRAMES = 16

# Minimum samples required to attempt training
MIN_POSITIVE_SAMPLES = 50
MIN_NEGATIVE_SAMPLES = 50


def check_dependencies() -> bool:
    """Verify required packages are available."""
    missing = []
    for pkg in ["openwakeword", "torch", "torchaudio", "numpy"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("   Run: pip install openwakeword==0.6.0 torch torchaudio torchinfo pronouncing acoustics audiomentations torch-audiomentations")
        return False
    return True


def load_audio_paths(data_dir: Path) -> Tuple[List[str], List[str]]:
    """Load positive and negative sample paths from training_data/."""
    pos_dir = data_dir / "positive"
    neg_dir = data_dir / "negative"

    positive = sorted(glob.glob(str(pos_dir / "*.wav")))
    negative = sorted(glob.glob(str(neg_dir / "*.wav")))

    # Also include any real voice samples in positive/real_*.wav
    real_samples = sorted(glob.glob(str(pos_dir / "real_*.wav")))
    if real_samples:
        print(f"   🎙️  Found {len(real_samples)} real voice samples — these will boost accuracy!")
        positive += real_samples

    return positive, negative


def validate_audio(paths: List[str], label: str, required_sr: int = 16000) -> List[str]:
    """Validate WAV files are correct format. Returns valid paths."""
    import wave
    valid = []
    errors = 0
    for path in paths:
        try:
            with wave.open(path, 'r') as wf:
                sr = wf.getframerate()
                ch = wf.getnchannels()
                if sr != required_sr:
                    print(f"   ⚠️  {Path(path).name}: wrong sample rate {sr}Hz (need {required_sr}Hz)")
                    errors += 1
                    continue
                if ch != 1:
                    print(f"   ⚠️  {Path(path).name}: stereo file (need mono)")
                    errors += 1
                    continue
                valid.append(path)
        except Exception as e:
            print(f"   ⚠️  {Path(path).name}: {e}")
            errors += 1

    if errors:
        print(f"   {label}: {len(valid)} valid, {errors} invalid (skipped)")
    else:
        print(f"   ✅ {label}: {len(valid)} valid samples")
    return valid


def compute_features(audio_paths: List[str], label: str) -> np.ndarray:
    """
    Compute audio embeddings using openWakeWord's AudioFeatures pipeline.
    Returns array of shape (N_windows, WINDOW_FRAMES, 96).
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from openwakeword.train import AudioFeatures, compute_features_from_generator

    print(f"   Computing embeddings for {len(audio_paths)} {label} clips...")

    # AudioFeatures computes melspectrograms + speech embeddings
    # Each 80ms chunk → 96-dim embedding
    # WINDOW_FRAMES consecutive embeddings → 1 training example

    all_features = []

    af = AudioFeatures(inference_framework="onnx")

    for i, path in enumerate(audio_paths):
        if (i + 1) % 50 == 0:
            print(f"     {i+1}/{len(audio_paths)}...")
        try:
            import soundfile as sf
            audio, sr = sf.read(path, dtype='int16')
            if sr != 16000:
                continue
            if audio.ndim > 1:
                audio = audio[:, 0]

            # Get embeddings for this clip — returns array of (T, 96)
            embeddings = []
            chunk_size = 1280  # 80ms at 16kHz
            n_chunks = len(audio) // chunk_size
            if n_chunks < WINDOW_FRAMES:
                continue

            for j in range(n_chunks):
                chunk = audio[j*chunk_size:(j+1)*chunk_size].astype(np.float32)
                # AudioFeatures expects int16 PCM as numpy int16
                chunk_i16 = chunk.astype(np.int16)
                emb = af.get_embeddings(chunk_i16)
                embeddings.append(emb)

            # Slide a WINDOW_FRAMES window over the embeddings
            embeddings = np.array(embeddings)  # (T, 96)
            for start in range(len(embeddings) - WINDOW_FRAMES + 1):
                window = embeddings[start:start + WINDOW_FRAMES]  # (16, 96)
                all_features.append(window)

        except Exception as e:
            pass

    if not all_features:
        return np.array([])

    return np.array(all_features)  # (N, 16, 96)


def compute_features_simple(audio_paths: List[str], label: str) -> np.ndarray:
    """
    Compute audio embeddings using openWakeWord's AudioFeatures.embed_clips().
    Input: list of 16kHz mono WAV paths.
    Returns array of shape (N, WINDOW_FRAMES, 96) — one embedding window per clip.
    """
    import soundfile as sf
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from openwakeword.train import AudioFeatures

    print(f"   Computing features for {len(audio_paths)} {label} clips...")

    af = AudioFeatures(inference_framework="onnx")

    # Target clip length: WINDOW_FRAMES embeddings × 160 samples/embedding = samples
    # openWakeWord produces 1 embedding per 160 samples (10ms) after melspectrogram
    # But embed_clips works on any length → returns (N, T, 96) where T = frames in clip
    # We want exactly WINDOW_FRAMES (16) frames → use 2s clips (32000 samples)
    target_samples = 32000  # 2 seconds at 16kHz → ~16 embedding frames

    valid_audios = []
    valid_indices = []

    for i, path in enumerate(audio_paths):
        try:
            audio, sr = sf.read(path, dtype='int16')
            if sr != 16000:
                continue
            if audio.ndim > 1:
                audio = audio[:, 0]
            # Pad or trim to target length
            if len(audio) < target_samples:
                audio = np.pad(audio, (0, target_samples - len(audio)))
            else:
                audio = audio[:target_samples]
            valid_audios.append(audio)
            valid_indices.append(i)
        except Exception:
            continue

    if not valid_audios:
        return np.array([])

    # Batch embed all clips at once
    X = np.stack(valid_audios)  # (N, 32000)
    BATCH = 64
    all_embeddings = []

    for start in range(0, len(X), BATCH):
        batch = X[start:start + BATCH]
        if (start // BATCH + 1) % 5 == 0 or start == 0:
            print(f"     {min(start + BATCH, len(X))}/{len(X)} clips...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            embs = af.embed_clips(batch)  # (B, T, 96)
        # Take the last WINDOW_FRAMES frames (end of clip = peak probability)
        T = embs.shape[1]
        if T >= WINDOW_FRAMES:
            window = embs[:, T - WINDOW_FRAMES:T, :]  # (B, 16, 96)
        else:
            # Pad frame dimension if needed
            pad = np.zeros((embs.shape[0], WINDOW_FRAMES - T, 96), dtype=np.float32)
            window = np.concatenate([pad, embs], axis=1)
        all_embeddings.append(window.astype(np.float32))

    result = np.concatenate(all_embeddings, axis=0)  # (N, 16, 96)
    print(f"     → {result.shape[0]} feature windows ({label})")
    return result


def export_to_onnx(model, output_path: Path, input_shape: tuple):
    """Export trained PyTorch model to ONNX format."""
    import torch
    import warnings

    model.eval()
    dummy_input = torch.randn(1, *input_shape)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model.model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

    print(f"   ✅ Model exported: {output_path}")


def train(
    positive_paths: List[str],
    negative_paths: List[str],
    steps: int,
    output_path: Path,
    log_path: Path,
):
    """Full training pipeline."""
    import torch
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from openwakeword.train import Model

    print(f"\n🔧 Computing feature embeddings...")
    print(f"   This may take several minutes (CPU-only mode)...")

    X_pos = compute_features_simple(positive_paths, "positive")
    X_neg = compute_features_simple(negative_paths, "negative")

    if X_pos.size == 0:
        print("❌ Could not compute any positive features. Check audio files.")
        return False
    if X_neg.size == 0:
        print("❌ Could not compute any negative features. Check audio files.")
        return False

    # Create labels
    y_pos = np.ones(len(X_pos), dtype=np.int64)
    y_neg = np.zeros(len(X_neg), dtype=np.int64)

    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([y_pos, y_neg], axis=0)

    # Shuffle
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]

    print(f"\n📊 Dataset:")
    print(f"   Positive windows: {len(X_pos)}")
    print(f"   Negative windows: {len(X_neg)}")
    print(f"   Total:            {len(X)}")
    print(f"   Feature shape:    {X.shape[1:]}")

    # Split train/val (90/10)
    split = int(0.9 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print(f"   Train: {len(X_train)}, Val: {len(X_val)}")

    # Build PyTorch data loaders
    from torch.utils.data import DataLoader, TensorDataset

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)

    # Cyclic training loader — loops indefinitely
    from itertools import cycle
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # Build model: DNN on 16×96 features → binary
    model = Model(
        n_classes=1,
        input_shape=(WINDOW_FRAMES, 96),
        model_type="dnn",
        layer_dim=128,
        n_blocks=2,
    )
    model.to(model.device)
    model.optimizer = torch.optim.Adam(model.model.parameters(), lr=LEARNING_RATE)

    print(f"\n🏋️  Training for {steps} steps...")
    print(f"   Device: {model.device}")

    # Validation steps — check every 500 steps (or 250 for quick runs)
    val_interval = 250 if steps <= QUICK_STEPS else 500
    val_steps = list(range(val_interval, steps + 1, val_interval))

    history = {"loss": [], "recall": [], "val_recall": [], "val_fp": []}
    best_val_recall = 0.0

    # Manual training loop (simpler than train_model API for our use case)
    from torch import nn

    loss_fn = nn.BCELoss()
    step = 0
    start_time = time.time()

    for epoch in range(1000):  # will break early
        for batch_X, batch_y in loader:
            if step >= steps:
                break

            # LR warmup + cosine decay
            if step < WARMUP_STEPS:
                lr = LEARNING_RATE * (step + 1) / WARMUP_STEPS
            elif step < WARMUP_STEPS + HOLD_STEPS:
                lr = LEARNING_RATE
            else:
                progress = (step - WARMUP_STEPS - HOLD_STEPS) / max(1, steps - WARMUP_STEPS - HOLD_STEPS)
                lr = LEARNING_RATE * 0.5 * (1 + np.cos(np.pi * progress))

            for g in model.optimizer.param_groups:
                g["lr"] = lr

            batch_X = batch_X.to(model.device)
            batch_y = batch_y.to(model.device)

            model.optimizer.zero_grad()
            preds = model.model(batch_X).squeeze()
            loss = loss_fn(preds, batch_y)
            loss.backward()
            model.optimizer.step()

            history["loss"].append(float(loss.detach().cpu()))

            # Compute recall on training batch
            predicted_pos = (preds > 0.5).float()
            actual_pos = batch_y.float()
            tp = (predicted_pos * actual_pos).sum()
            fn = ((1 - predicted_pos) * actual_pos).sum()
            recall = float(tp / (tp + fn + 1e-8))
            history["recall"].append(recall)

            step += 1

            # Validation
            if step in val_steps or step == steps:
                model.model.eval()
                with torch.no_grad():
                    val_X = X_val_t.to(model.device)
                    val_y = y_val_t.to(model.device)
                    val_preds = model.model(val_X).squeeze()
                    val_predicted = (val_preds > 0.5).float()
                    val_tp = (val_predicted * val_y).sum()
                    val_fn = ((1 - val_predicted) * val_y).sum()
                    val_fp = (val_predicted * (1 - val_y)).sum()
                    val_recall = float(val_tp / (val_tp + val_fn + 1e-8))
                    val_fp_rate = float(val_fp / (val_y == 0).sum())

                history["val_recall"].append(val_recall)
                history["val_fp"].append(val_fp_rate)

                if val_recall > best_val_recall:
                    best_val_recall = val_recall
                    # Save best model state
                    best_state = {k: v.clone() for k, v in model.model.state_dict().items()}

                elapsed = time.time() - start_time
                avg_loss = np.mean(history["loss"][-50:]) if history["loss"] else 0
                print(f"   Step {step:4d}/{steps} | loss={avg_loss:.4f} | recall={recall:.3f} | val_recall={val_recall:.3f} | val_fp={val_fp_rate:.3f} | lr={lr:.6f} | {elapsed:.0f}s")
                model.model.train()

        if step >= steps:
            break

    # Restore best model weights
    if "best_state" in dir():
        model.model.load_state_dict(best_state)
        print(f"\n   ↩️  Restored best model weights (val_recall={best_val_recall:.3f})")

    print(f"\n📦 Exporting model...")
    MODELS_DIR.mkdir(exist_ok=True)
    export_to_onnx(model, output_path, (WINDOW_FRAMES, 96))

    # Save training log
    log = {
        "model": str(output_path),
        "steps": steps,
        "positive_samples": len(X_pos),
        "negative_samples": len(X_neg),
        "window_frames": WINDOW_FRAMES,
        "best_val_recall": best_val_recall,
        "history_final_loss": float(np.mean(history["loss"][-50:])) if history["loss"] else 0,
        "history_final_recall": float(np.mean(history["recall"][-50:])) if history["recall"] else 0,
        "history": {k: v[-100:] for k, v in history.items()},  # last 100 steps
    }
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"   📋 Training log: {log_path}")

    return True


def run_quick_test(positive_paths: List[str], negative_paths: List[str]):
    """Validate data and check the training pipeline without full training."""
    print("🔍 Quick validation mode (--test)")

    print(f"   Positive samples: {len(positive_paths)}")
    print(f"   Negative samples: {len(negative_paths)}")

    # Test feature extraction on first 5 samples
    print("\n   Testing feature extraction on 5 samples...")
    X_pos_sample = compute_features_simple(positive_paths[:5], "positive (sample)")
    X_neg_sample = compute_features_simple(negative_paths[:5], "negative (sample)")

    if X_pos_sample.size > 0:
        print(f"   ✅ Feature shape: {X_pos_sample.shape[1:]} (expected: ({WINDOW_FRAMES}, 96))")
    else:
        print("   ❌ Feature extraction failed — check audio format")
        return False

    print("\n✅ Validation passed. Run without --test for full training.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Train openWakeWord model for 'Claudette' wake word",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help=f"Training steps (default: {DEFAULT_STEPS})")
    parser.add_argument("--quick", action="store_true",
                        help=f"Quick training ({QUICK_STEPS} steps only)")
    parser.add_argument("--test", action="store_true",
                        help="Dry run: validate data and feature extraction, skip training")
    parser.add_argument("--data-dir", type=Path, default=TRAINING_DATA_DIR,
                        help="Path to training data directory")
    parser.add_argument("--output", type=Path, default=OUTPUT_MODEL_PATH,
                        help="Output ONNX model path")
    args = parser.parse_args()

    print("🏠 Claudette Home — Wake Word Model Trainer")
    print("=" * 50)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Check for soundfile
    try:
        import soundfile
    except ImportError:
        print("📦 Installing soundfile...")
        os.system("pip3 install soundfile --break-system-packages --quiet")
        import soundfile

    # Load audio paths
    print(f"\n📁 Loading training data from: {args.data_dir}")
    positive_paths, negative_paths = load_audio_paths(args.data_dir)

    if len(positive_paths) < MIN_POSITIVE_SAMPLES:
        print(f"❌ Need at least {MIN_POSITIVE_SAMPLES} positive samples, found {len(positive_paths)}")
        print("   Run: python3 generate_training_data.py --count 200")
        sys.exit(1)

    if len(negative_paths) < MIN_NEGATIVE_SAMPLES:
        print(f"❌ Need at least {MIN_NEGATIVE_SAMPLES} negative samples, found {len(negative_paths)}")
        print("   Run: python3 generate_training_data.py --count 200")
        sys.exit(1)

    # Validate audio format
    print("\n🔊 Validating audio files...")
    positive_paths = validate_audio(positive_paths, "positive")
    negative_paths = validate_audio(negative_paths, "negative")

    if args.test:
        success = run_quick_test(positive_paths, negative_paths)
        sys.exit(0 if success else 1)

    steps = QUICK_STEPS if args.quick else args.steps
    if args.quick:
        print(f"\n⚡ Quick mode: {steps} steps (accuracy will be limited)")

    print(f"\n📊 Training config:")
    print(f"   Steps:           {steps}")
    print(f"   Warmup:          {WARMUP_STEPS}")
    print(f"   Hold:            {HOLD_STEPS}")
    print(f"   Learning rate:   {LEARNING_RATE}")
    print(f"   Batch size:      {BATCH_SIZE}")
    print(f"   Window:          {WINDOW_FRAMES} frames × 96 dims = 1.28s")
    print(f"   Output:          {args.output}")

    start = time.time()
    success = train(
        positive_paths=positive_paths,
        negative_paths=negative_paths,
        steps=steps,
        output_path=args.output,
        log_path=TRAINING_LOG_PATH,
    )

    elapsed = time.time() - start
    if success:
        print(f"\n✅ Training complete in {elapsed:.0f}s")
        print(f"   Model saved: {args.output}")
        print(f"\n🔌 Integration:")
        print(f"   from openwakeword.model import Model")
        print(f"   oww = Model(wakeword_model_paths=['{args.output}'])")
        print(f"   prediction = oww.predict(audio_frame)  # 1280 samples @ 16kHz")
        print(f"\n📋 Next steps:")
        print(f"   1. Test model: python3 oww_listener.py  (needs USB mic)")
        print(f"   2. Tune threshold in oww_listener.py (default 0.5, try 0.7)")
        print(f"   3. Add Mattie's real voice recordings to training_data/positive/real_*.wav")
        print(f"   4. Re-train with real data for much better accuracy")
    else:
        print(f"\n❌ Training failed. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
