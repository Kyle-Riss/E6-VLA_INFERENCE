import torch
import torchvision
import sys

print(f"Python version: {sys.version}")
print(f"Torch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

try:
    boxes = torch.tensor([[0,0,10,10], [1,1,11,11]], dtype=torch.float32).cuda()
    scores = torch.tensor([0.9, 0.8], dtype=torch.float32).cuda()
    keep = torchvision.ops.nms(boxes, scores, 0.5)
    print(f"Keep indices: {keep}")
    print("✅ SUCCESS: CUDA NMS executed")
except Exception as e:
    print(f"❌ FAILURE: {e}")
    # Print more details about the error if possible
    import traceback
    traceback.print_exc()

# Check for _C
try:
    print(f"Torchvision _C: {torchvision._C if hasattr(torchvision, '_C') else 'Missing'}")
except Exception as e:
    print(f"Error checking _C: {e}")
