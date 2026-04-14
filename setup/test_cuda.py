#!/usr/bin/env python3
"""CUDA 및 GPU 성능 테스트"""

import torch
import time

print("=" * 60)
print("CUDA 환경 확인")
print("=" * 60)

# 기본 정보
print(f"\n[PyTorch]")
print(f"  버전: {torch.__version__}")
print(f"  CUDA 사용 가능: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"\n[GPU 정보]")
    print(f"  디바이스 개수: {torch.cuda.device_count()}")
    print(f"  현재 디바이스: {torch.cuda.current_device()}")
    print(f"  디바이스 이름: {torch.cuda.get_device_name(0)}")
    
    # 메모리 정보
    props = torch.cuda.get_device_properties(0)
    total_memory = props.total_memory / (1024**3)  # GB로 변환
    print(f"  총 메모리: {total_memory:.2f} GB")
    print(f"  CUDA Capability: {props.major}.{props.minor}")
    
    # 현재 메모리 사용량
    allocated = torch.cuda.memory_allocated(0) / (1024**2)  # MB
    reserved = torch.cuda.memory_reserved(0) / (1024**2)  # MB
    print(f"\n[메모리 사용량]")
    print(f"  할당됨: {allocated:.2f} MB")
    print(f"  예약됨: {reserved:.2f} MB")
    
    # 간단한 연산 테스트
    print(f"\n[GPU 연산 테스트]")
    device = torch.device("cuda")
    
    # 작은 행렬 연산
    print("  테스트 1: 작은 행렬 곱셈 (1000x1000)")
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)
    
    start = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) * 1000
    print(f"    소요 시간: {elapsed:.2f} ms")
    
    # 큰 행렬 연산
    print("  테스트 2: 큰 행렬 곱셈 (5000x5000)")
    a = torch.randn(5000, 5000, device=device)
    b = torch.randn(5000, 5000, device=device)
    
    start = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) * 1000
    print(f"    소요 시간: {elapsed:.2f} ms")
    
    # 메모리 정리
    del a, b, c
    torch.cuda.empty_cache()
    
    allocated = torch.cuda.memory_allocated(0) / (1024**2)
    print(f"\n[정리 후 메모리]")
    print(f"  할당됨: {allocated:.2f} MB")
    
    print("\n" + "=" * 60)
    print("✓ CUDA 테스트 완료!")
    print("=" * 60)
else:
    print("\n⚠ CUDA를 사용할 수 없습니다.")
