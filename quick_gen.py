#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
빠른 모델 생성 스크립트 (Quick Generator)
"""
import torch
import torch.nn as nn
import numpy as np

# ============================================================
# 상수 정의
# ============================================================
JOKER_VALUE = -1
NUM_CARDS = 26
MAX_HAND_SIZE = 13
NUM_NUMBERS = 13
INPUT_SIZE = 65
OUTPUT_SIZE = 169

class DaVinciAIModel(nn.Module):
    def __init__(self):
        super(DaVinciAIModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(INPUT_SIZE, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, OUTPUT_SIZE)
        )
    
    def forward(self, x):
        return self.network(x)

def main():
    print("빠른 모델 생성 시작 (1 Epoch)...")
    model = DaVinciAIModel()
    model.eval()
    
    # 더미 입력
    dummy_input = torch.randn(1, INPUT_SIZE)
    
    # ONNX 내보내기 (Opset 12)
    output_path = "davinci_ai.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['game_state'],
        output_names=['action_scores'],
        dynamic_axes={
            'game_state': {0: 'batch_size'},
            'action_scores': {0: 'batch_size'}
        },
        verbose=False
    )
    print(f"ONNX 모델 생성 완료: {output_path}")

    # 모델 유효성 검증
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("   ONNX 모델 유효성 검증 완료! (Opset 13)")
    except ImportError:
        pass
    except Exception as e:
        print(f"   검증 실패: {e}")

if __name__ == "__main__":
    main()
