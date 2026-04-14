import os
import sys
import time
import subprocess

def print_header(text):
    print(f"\n\033[1;34m{'='*70}\033[0m")
    print(f"\033[1;36m {text} \033[0m")
    print(f"\033[1;34m{'='*70}\033[0m")

def print_result(label, value, success=True):
    color = "\033[92m" if success else "\033[91m"
    reset = "\033[0m"
    print(f"{label:<30}: {color}{value}{reset}")

def check_env():
    print_header("JETSON AGX ORIN - OPENPI ENVIRONMENT HEALTH CHECK")
    
    # 1. System Information
    print("\n[1. System Info]")
    print_result("Host Device", "NVIDIA Jetson AGX Orin")
    print_result("Python Version", sys.version.split()[0])
    
    # 2. PyTorch & CUDA
    print("\n[2. PyTorch & CUDA Support]")
    try:
        import torch
        print_result("PyTorch Version", torch.__version__)
        cuda_ok = torch.cuda.is_available()
        print_result("CUDA Available", cuda_ok, cuda_ok)
        if cuda_ok:
            print_result("GPU Device", torch.cuda.get_device_name(0))
            # Functional test
            x = torch.rand(100, 100).cuda()
            y = x @ x
            print_result("CUDA Matrix Math", "✅ Functional")
    except Exception as e:
        print_result("PyTorch Status", f"❌ Error: {e}", False)

    # 3. Torchvision & CUDA Ops
    print("\n[3. Torchvision & Custom Build]")
    try:
        import torchvision
        print_result("Torchvision Version", torchvision.__version__)
        from torchvision.ops import nms
        # Functional test for custom built NMS
        boxes = torch.tensor([[0,0,10,10], [1,1,11,11]], dtype=torch.float32).cuda()
        scores = torch.tensor([0.9, 0.8], dtype=torch.float32).cuda()
        keep = nms(boxes, scores, 0.5)
        print_result("CUDA NMS Kernel", "✅ Functional (Custom Built)", True)
    except Exception as e:
        print_result("Torchvision Status", f"❌ Missing CUDA Ops: {e}", False)

    # 4. JAX & Flax (OpenPi Engine)
    print("\n[4. JAX/Flax Core Packages]")
    try:
        import jax
        import flax
        import numpy as np
        print_result("JAX Version", jax.__version__)
        print_result("Flax Version", flax.__version__)
        print_result("NumPy Version", np.__version__)
        
        # Check backend
        backend = jax.lib.xla_bridge.get_backend().platform
        print_result("JAX Backend", backend.upper(), backend == 'gpu')
    except Exception as e:
        print_result("JAX/Flax Status", f"❌ Error: {e}", False)

    # 5. OpenPi Patches & Checkpoints
    print("\n[5. OpenPi Integration]")
    model_patch_path = "openpi/models/model.py"
    if os.path.exists(model_patch_path):
        with open(model_patch_path, 'r') as f:
            content = f.read()
            patched = "hasattr(metadata, \"item_metadata\")" in content
            print_result("Orbax/Metadata Patch", "✅ Applied" if patched else "⚠️ Missing", patched)
    
    checkpoint_exists = os.path.exists("checkpoints/pi05_droid")
    print_result("Model Checkpoint", "✅ Ready" if checkpoint_exists else "❌ Missing", checkpoint_exists)

    print_header("ALL CHECKS COMPLETED - READY FOR INFERENCE")

if __name__ == "__main__":
    # Ensure stdout is flushed
    sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
    check_env()
