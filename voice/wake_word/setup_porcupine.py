#!/usr/bin/env python3
"""
Claudette Home — Porcupine Setup & Validation Script
Run this ONCE after getting your Picovoice credentials to validate everything is
in order before going live. Also useful as a smoke test before each deployment.

Steps covered:
  1. Check pvporcupine SDK is installed + version
  2. Check PORCUPINE_ACCESS_KEY is set (env or Infisical)
  3. Validate the access key (can create Porcupine handle with a built-in keyword)
  4. Check .ppn model file exists at expected path
  5. Try creating Porcupine with custom model (the real 'claudette' model)
  6. Print a setup checklist if anything is missing

Usage:
  python3 setup_porcupine.py [--model models/claudette_linux.ppn]

Environment:
  PORCUPINE_ACCESS_KEY — required (get at console.picovoice.ai)
  PICOVOICE_ACCESS_KEY — alias (same thing, alternate env name)
"""

import argparse
import os
import sys
from pathlib import Path

# ── ANSI colours ──────────────────────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"

OK = f"{GREEN}✅{RESET}"
FAIL = f"{RED}❌{RESET}"
WARN = f"{YELLOW}⚠️ {RESET}"
INFO = f"{CYAN}ℹ️ {RESET}"


def step(label: str):
    print(f"\n{BOLD}{'─' * 60}{RESET}")
    print(f"{BOLD}  {label}{RESET}")
    print(f"{BOLD}{'─' * 60}{RESET}")


def check_sdk() -> bool:
    step("1. pvporcupine SDK")
    try:
        import pvporcupine
        # pvporcupine 4.x doesn't expose __version__ directly — check the package
        try:
            from importlib.metadata import version
            ver = version("pvporcupine")
        except Exception:
            ver = "unknown"
        print(f"  {OK} pvporcupine installed (version: {ver})")
        print(f"  {INFO} Built-in keywords available: {len(pvporcupine.KEYWORDS)}")
        # Warn if using 3.x — 4.x API changed slightly
        if ver != "unknown" and ver.startswith("3"):
            print(f"  {WARN} You're on pvporcupine 3.x — 4.x is recommended (pip install -U pvporcupine)")
        return True
    except ImportError:
        print(f"  {FAIL} pvporcupine NOT installed")
        print(f"       Fix: pip install pvporcupine")
        return False


def check_access_key() -> str | None:
    step("2. Picovoice Access Key")
    # Try both common env var names
    key = (
        os.environ.get("PORCUPINE_ACCESS_KEY")
        or os.environ.get("PICOVOICE_ACCESS_KEY")
    )
    if not key:
        print(f"  {FAIL} PORCUPINE_ACCESS_KEY not set in environment")
        print(f"       1. Go to https://console.picovoice.ai")
        print(f"       2. Sign up / log in (free)")
        print(f"       3. Copy your Access Key from the dashboard")
        print(f"       4. Add to /etc/environment: PORCUPINE_ACCESS_KEY=<key>")
        print(f"       5. Add to Infisical: infisical secrets set PORCUPINE_ACCESS_KEY <key>")
        return None
    # Mask for display — show first 6 + last 4 chars
    masked = key[:6] + "…" + key[-4:] if len(key) > 12 else "***"
    print(f"  {OK} PORCUPINE_ACCESS_KEY set ({masked})")
    return key


def validate_key(access_key: str) -> bool:
    """Try creating a Porcupine handle with a built-in keyword — validates the key."""
    step("3. Access Key Validation")
    try:
        import pvporcupine
        # Use a simple built-in keyword to test the key without a custom model
        porcupine = pvporcupine.create(
            access_key=access_key,
            keywords=["porcupine"],
        )
        sr = porcupine.sample_rate
        fl = porcupine.frame_length
        porcupine.delete()
        print(f"  {OK} Access key is valid!")
        print(f"  {INFO} Porcupine engine: sample_rate={sr}Hz, frame_length={fl} samples")
        return True
    except pvporcupine.PorcupineActivationError:
        print(f"  {FAIL} Access key activation failed — key may be expired or invalid")
        print(f"       Go to console.picovoice.ai and check/regenerate your key")
        return False
    except pvporcupine.PorcupineActivationLimitError:
        print(f"  {FAIL} Access key usage limit exceeded (free tier)")
        print(f"       Wait for quota reset or upgrade at picovoice.ai")
        return False
    except Exception as e:
        print(f"  {FAIL} Unexpected error validating key: {e}")
        return False


def check_model(model_path: str, access_key: str) -> bool:
    step("4. Custom Wake Word Model (.ppn)")
    path = Path(model_path)
    if not path.exists():
        print(f"  {FAIL} Model not found: {path}")
        print()
        print(f"  {BOLD}To get the 'Claudette' wake word model:{RESET}")
        print(f"  1. Go to https://console.picovoice.ai")
        print(f"  2. Navigate to: Wake Word → Custom Keyword")
        print(f"  3. Create new keyword: claudette")
        print(f"  4. Select platform: Linux (x86_64)")
        print(f"  5. Download the .ppn file")
        print(f"  6. Place it at: {path.resolve()}")
        print()
        print(f"  {INFO} The training takes ~30 seconds on Picovoice servers")
        print(f"  {INFO} Free tier allows 3 custom keywords")
        return False

    print(f"  {OK} Model file exists: {path}")
    file_size = path.stat().st_size
    print(f"  {INFO} File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")

    # Try loading the model
    try:
        import pvporcupine
        porcupine = pvporcupine.create(
            access_key=access_key,
            keyword_paths=[str(path)],
            sensitivities=[0.5],
        )
        porcupine.delete()
        print(f"  {OK} Model loaded successfully — 'Claudette' wake word is ready!")
        return True
    except Exception as e:
        print(f"  {FAIL} Failed to load model: {e}")
        return False


def check_pipeline_integration() -> bool:
    step("5. Pipeline Integration Check")
    base = Path(__file__).parent.parent  # voice/

    checks = [
        ("wake_word_bridge.py", base / "wake_word" / "wake_word_bridge.py"),
        ("pipeline.py", base / "pipeline.py"),
        ("tts_responder.py", base / "tts_responder.py"),
        ("transcribe_api.py", base / "stt_pipeline" / "transcribe_api.py"),
        ("ha_bridge.py", base / "ha_bridge" / "ha_bridge.py"),
        ("intent_parser.py", base / "intent_parser" / "intent_parser.py"),
    ]

    all_ok = True
    for name, path in checks:
        if path.exists():
            print(f"  {OK} {name}")
        else:
            print(f"  {FAIL} {name} — missing at {path}")
            all_ok = False

    if all_ok:
        print(f"\n  {OK} Full pipeline chain is in place")
    return all_ok


def print_summary(results: dict):
    step("Summary")
    all_pass = all(results.values())
    for step_name, passed in results.items():
        icon = OK if passed else FAIL
        print(f"  {icon} {step_name}")

    print()
    if all_pass:
        print(f"  {GREEN}{BOLD}🎉 All checks passed! Claudette wake word is ready to deploy.{RESET}")
        print()
        print(f"  To start the listener:")
        print(f"    python3 voice/wake_word/wake_word_bridge.py --backend porcupine")
        print()
        print(f"  To install as a systemd service:")
        print(f"    sudo cp voice/wake_word/claudette-wake-word.service /etc/systemd/system/")
        print(f"    sudo systemctl enable claudette-wake-word")
        print(f"    sudo systemctl start claudette-wake-word")
    else:
        print(f"  {YELLOW}{BOLD}Some checks failed. Fix the issues above then re-run.{RESET}")
        if not results.get("Access Key"):
            print()
            print(f"  {YELLOW}Priority action: Get your Picovoice access key first{RESET}")
            print(f"  → https://console.picovoice.ai (free account, takes 2 minutes)")


def main():
    parser = argparse.ArgumentParser(
        description="Claudette Home — Porcupine setup validator"
    )
    parser.add_argument(
        "--model",
        default=os.path.join(os.path.dirname(__file__), "models", "claudette_linux.ppn"),
        help="Path to .ppn model (default: models/claudette_linux.ppn)",
    )
    parser.add_argument(
        "--skip-key-validation",
        action="store_true",
        help="Skip live API call to validate key (use if offline)",
    )
    args = parser.parse_args()

    print(f"\n{BOLD}{CYAN}Claudette Home — Porcupine Wake Word Setup Check{RESET}")
    print(f"{'─' * 60}")

    results = {}

    # Step 1: SDK
    sdk_ok = check_sdk()
    results["pvporcupine SDK"] = sdk_ok
    if not sdk_ok:
        print_summary(results)
        sys.exit(1)

    # Step 2: Access key
    access_key = check_access_key()
    results["Access Key"] = access_key is not None
    if not access_key:
        results["Key Validation"] = False
        results["Model (.ppn)"] = False
        results["Pipeline files"] = check_pipeline_integration()
        print_summary(results)
        sys.exit(1)

    # Step 3: Validate key (live API call)
    if not args.skip_key_validation:
        key_valid = validate_key(access_key)
        results["Key Validation"] = key_valid
    else:
        print(f"\n  {WARN} Key validation skipped (--skip-key-validation)")
        results["Key Validation"] = None  # type: ignore

    # Step 4: Model file
    key_for_model = access_key if results.get("Key Validation") else None
    if key_for_model:
        model_ok = check_model(args.model, key_for_model)
        results["Model (.ppn)"] = model_ok
    else:
        # Check if file exists at least, even if we can't load it
        model_exists = Path(args.model).exists()
        results["Model (.ppn)"] = model_exists
        if not model_exists:
            step("4. Custom Wake Word Model (.ppn)")
            print(f"  {FAIL} Model not found: {args.model}")
            print(f"       (Key validation failed — can't test model loading)")

    # Step 5: Pipeline
    results["Pipeline files"] = check_pipeline_integration()

    print_summary(results)

    # Exit code: 0 if all pass (or skipped), 1 if any failed
    has_failures = any(v is False for v in results.values())
    sys.exit(1 if has_failures else 0)


if __name__ == "__main__":
    main()
