"""
inspect_model.py
-----------------
RUN THIS FIRST.

You said you're not sure how your .pt file was saved. This script loads it
in the safest possible way and tells you EXACTLY what's inside, so we know
which loading path to use in model_loader.py.

There are three common cases for a saved .pt file:
  1. A full model object      -> torch.save(model, "file.pt")
  2. A state_dict (weights)   -> torch.save(model.state_dict(), "file.pt")
  3. A checkpoint dict        -> torch.save({"model": ..., "optimizer": ..., ...}, "file.pt")

Usage:
    python inspect_model.py best.pt
    python inspect_model.py /path/to/your_model.pt
"""

import sys
import torch


def inspect(path: str):
    print(f"\nLoading: {path}\n{'-' * 60}")

    # weights_only=False is required to load full pickled model objects.
    # Only do this for files YOU trained/saved yourself — never for
    # untrusted .pt files, since this can execute arbitrary code.
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"torch.load failed: {e}")
        print("The file may be corrupted or require a custom class to unpickle.")
        return

    print(f"Top-level type: {type(obj)}\n")

    # Case 1: Full model object
    if isinstance(obj, torch.nn.Module):
        print("=> This is a FULL MODEL OBJECT (nn.Module).")
        print("   You can load it directly with:")
        print("     model = torch.load(path, map_location='cpu', weights_only=False)")
        print("     model.eval()\n")
        print("Model architecture:")
        print(obj)
        print("\nNote: to unpickle this on another machine, the original class")
        print("definition (the .py file where the model class was defined) must")
        print("be importable in that environment.")
        return

    # Case 2: Plain state_dict (an OrderedDict of tensor weights)
    if isinstance(obj, dict) and all(
        hasattr(v, "shape") for v in obj.values() if not isinstance(v, dict)
    ) and not any(k in obj for k in ("state_dict", "model", "model_state_dict")):
        print("=> This looks like a STATE_DICT (raw weights only).")
        print("   You need the original model class to load it, e.g.:")
        print("     model = YourModelClass(...)")
        print("     model.load_state_dict(torch.load(path, map_location='cpu'))")
        print("     model.eval()\n")
        print(f"Number of parameter tensors: {len(obj)}")
        print("First 15 keys and shapes:")
        for i, (k, v) in enumerate(obj.items()):
            if i >= 15:
                print("   ...")
                break
            shape = tuple(v.shape) if hasattr(v, "shape") else "N/A"
            print(f"   {k:50s} {shape}")

        keys = list(obj.keys())
        print("\nHeuristics:")
        print(f"   First layer key: {keys[0]} -> shape {tuple(obj[keys[0]].shape)}")
        print(f"   Last layer key:  {keys[-1]} -> shape {tuple(obj[keys[-1]].shape)}")
        print("   (The last Linear/Conv layer's output-channel size usually")
        print("    tells you the number of output classes.)")
        return

    # Case 3: Checkpoint dict wrapping a state_dict + metadata
    if isinstance(obj, dict):
        print("=> This is a CHECKPOINT DICT (contains metadata + weights).")
        print(f"   Top-level keys: {list(obj.keys())}\n")
        for k, v in obj.items():
            if hasattr(v, "shape"):
                print(f"   {k}: tensor, shape {tuple(v.shape)}")
            elif isinstance(v, dict):
                print(f"   {k}: dict with {len(v)} keys (likely a state_dict)")
                sub_keys = list(v.keys())[:5]
                print(f"        sample keys: {sub_keys}")
            else:
                print(f"   {k}: {type(v)} = "
                      f"{v if not isinstance(v, (list, tuple)) else f'{type(v)} of len {len(v)}'}")
        print("\n   Look for a key like 'model', 'state_dict', or 'model_state_dict'")
        print("   above — that's the actual weights dict. Load it with:")
        print("     model = YourModelClass(...)")
        print("     ckpt = torch.load(path, map_location='cpu')")
        print("     model.load_state_dict(ckpt['state_dict'])  # adjust key name")
        return

    print("=> Unrecognised structure. Printing repr for manual inspection:")
    print(repr(obj)[:2000])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_model.py /path/to/best.pt")
        sys.exit(1)
    inspect(sys.argv[1])
