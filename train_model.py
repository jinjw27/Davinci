#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
다빈치 코드 AI 모델 학습 및 ONNX 변환 스크립트

이 스크립트는:
1. 게임 상태를 입력으로 받는 신경망 모델을 정의
2. 휴리스틱 기반 학습 데이터 생성
3. 모델 학습
4. ONNX 형식으로 내보내기
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# ============================================================
# 게임 상수
# ============================================================
JOKER_VALUE = -1  # 조커 숫자 값
NUM_CARDS = 26    # 총 카드 수 (0-11 흑/백 + 조커 2장)
MAX_HAND_SIZE = 13  # 최대 패 크기
NUM_NUMBERS = 13  # -1(조커) ~ 11 = 13가지 숫자

# ============================================================
# 입력 특성 설계
# - 공개된 카드 정보: 26개 (각 카드별 공개 여부)
# - AI 패 정보: 13개 (각 위치의 카드 숫자, 비공개면 -2)
# - 플레이어 패 정보: 13개 (위치별 비공개 카드 존재 여부)
# - 가능한 숫자 마스크: 13개 (각 숫자가 아직 가능한지)
# 총: 26 + 13 + 13 + 13 = 65
# ============================================================
INPUT_SIZE = 65
OUTPUT_SIZE = MAX_HAND_SIZE * NUM_NUMBERS  # 위치 x 숫자 조합


class DaVinciAIModel(nn.Module):
    """
    다빈치 코드 AI 모델
    
    게임 상태를 입력받아 각 (위치, 숫자) 조합의 추리 확률을 출력
    """
    
    def __init__(self):
        super(DaVinciAIModel, self).__init__()
        
        self.network = nn.Sequential(
            # 입력층 -> 은닉층 1
            nn.Linear(INPUT_SIZE, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # 은닉층 2
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # 은닉층 3
            nn.Linear(256, 128),
            nn.ReLU(),
            
            # 출력층: 각 (위치, 숫자) 조합의 점수
            nn.Linear(128, OUTPUT_SIZE)
        )
    
    def forward(self, x):
        """순전파"""
        return self.network(x)


def encode_game_state(revealed_cards, ai_hand, player_hand_hidden, possible_numbers):
    """
    게임 상태를 신경망 입력 벡터로 인코딩
    
    Args:
        revealed_cards: 공개된 카드 리스트 [(숫자, 색상), ...]
        ai_hand: AI 패 [(숫자, 공개여부), ...]
        player_hand_hidden: 플레이어 비공개 카드 위치 리스트
        possible_numbers: 아직 가능한 숫자 리스트
    
    Returns:
        65차원 입력 벡터
    """
    features = np.zeros(INPUT_SIZE, dtype=np.float32)
    
    # 1. 공개된 카드 정보 (26개)
    # 각 카드: 숫자 * 2 + (흰색이면 1)
    for num, color in revealed_cards:
        if num == JOKER_VALUE:
            idx = 24 if color == 'black' else 25
        else:
            idx = num * 2 + (1 if color == 'white' else 0)
        if 0 <= idx < 26:
            features[idx] = 1.0
    
    # 2. AI 패 정보 (13개) - 정규화된 숫자 값
    offset = 26
    for i, (num, revealed) in enumerate(ai_hand[:MAX_HAND_SIZE]):
        if revealed:
            features[offset + i] = (num + 1) / 13.0  # 정규화
        else:
            features[offset + i] = -0.1  # 비공개
    
    # 3. 플레이어 비공개 카드 위치 (13개)
    offset = 39
    for i in range(MAX_HAND_SIZE):
        if i < len(player_hand_hidden):
            features[offset + i] = 1.0  # 비공개 카드 존재
    
    # 4. 가능한 숫자 마스크 (13개)
    offset = 52
    for num in possible_numbers:
        idx = num + 1  # -1 -> 0, 0 -> 1, ..., 11 -> 12
        if 0 <= idx < 13:
            features[offset + idx] = 1.0
    
    return features


def generate_training_data(num_samples=10000):
    """
    휴리스틱 기반 학습 데이터 생성
    
    스마트한 AI가 선택할 만한 추리를 타겟으로 생성
    """
    X = []
    y = []
    
    for _ in range(num_samples):
        # 랜덤 게임 상태 생성
        num_revealed = random.randint(0, 15)
        revealed_cards = []
        all_cards = []
        
        # 가능한 모든 카드 생성
        for num in range(12):
            all_cards.append((num, 'black'))
            all_cards.append((num, 'white'))
        all_cards.append((JOKER_VALUE, 'black'))
        all_cards.append((JOKER_VALUE, 'white'))
        
        random.shuffle(all_cards)
        revealed_cards = all_cards[:num_revealed]
        remaining_cards = all_cards[num_revealed:]
        
        # AI 패 (4-8장)
        ai_hand_size = random.randint(4, 8)
        ai_hand = []
        for i in range(ai_hand_size):
            if i < len(remaining_cards):
                num, color = remaining_cards[i]
                revealed = random.random() < 0.3
                ai_hand.append((num, revealed))
        
        # 플레이어 비공개 카드 위치 (숨겨진 카드들)
        player_hand_size = random.randint(3, 8)
        player_hand_hidden = list(range(player_hand_size))
        
        # 가능한 숫자 계산
        revealed_nums = [c[0] for c in revealed_cards]
        possible_numbers = []
        
        if revealed_nums.count(JOKER_VALUE) < 2:
            possible_numbers.append(JOKER_VALUE)
        for i in range(12):
            if revealed_nums.count(i) < 2:
                possible_numbers.append(i)
        
        if not possible_numbers or not player_hand_hidden:
            continue
        
        # 입력 인코딩
        features = encode_game_state(
            revealed_cards, ai_hand, player_hand_hidden, possible_numbers
        )
        
        # 타겟: 휴리스틱 기반 추리 선택
        # - 가능한 숫자 중 덜 공개된 숫자 선호
        # - 중간 위치 카드 선호 (정보가 더 많음)
        target = np.zeros(OUTPUT_SIZE, dtype=np.float32)
        
        for pos in player_hand_hidden:
            if pos >= MAX_HAND_SIZE:
                continue
            for num in possible_numbers:
                num_idx = num + 1  # -1 -> 0, 0 -> 1, ...
                if num_idx < 0 or num_idx >= NUM_NUMBERS:
                    continue
                
                output_idx = pos * NUM_NUMBERS + num_idx
                if output_idx >= OUTPUT_SIZE:
                    continue
                
                # 휴리스틱 점수 계산
                score = 0.5
                
                # 덜 공개된 숫자 선호
                count = revealed_nums.count(num)
                score += (2 - count) * 0.2
                
                # 중간 위치 선호
                mid = len(player_hand_hidden) / 2
                score += max(0, 0.3 - abs(pos - mid) * 0.1)
                
                # 노이즈 추가
                score += random.gauss(0, 0.1)
                
                target[output_idx] = max(0, min(1, score))
        
        # 정규화
        if target.sum() > 0:
            target = target / target.sum()
        
        X.append(features)
        y.append(target)
    
    return np.array(X), np.array(y)


def train_model(model, X, y, epochs=100, batch_size=64):
    """
    모델 학습
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X), 
        torch.FloatTensor(y)
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    
    print("학습 시작...")
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(loader)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
    
    print("학습 완료!")


def export_to_onnx(model, output_path="davinci_ai.onnx"):
    """
    모델을 ONNX 형식으로 내보내기
    """
    model.eval()
    
    # 더미 입력
    dummy_input = torch.randn(1, INPUT_SIZE)
    
    # ONNX 내보내기 (onnxscript 없이 기본 방식 사용)
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
    
    print(f"ONNX 모델 저장됨: {output_path}")

    # 모델 유효성 검증
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("   ONNX 모델 유효성 검증 완료! (Opset 13)")
    except Exception as e:
        print(f"   검증 실패: {e}")


def main():
    print("=" * 50)
    print("다빈치 코드 AI 모델 학습")
    print("=" * 50)
    
    # 학습 데이터 생성
    print("\n1. 학습 데이터 생성 중...")
    X, y = generate_training_data(num_samples=20000)
    print(f"   생성된 샘플 수: {len(X)}")
    
    # 모델 생성
    print("\n2. 모델 생성...")
    model = DaVinciAIModel()
    print(f"   입력 크기: {INPUT_SIZE}")
    print(f"   출력 크기: {OUTPUT_SIZE}")
    
    # 학습
    print("\n3. 모델 학습...")
    train_model(model, X, y, epochs=50, batch_size=64)
    
    # ONNX 내보내기
    print("\n4. ONNX 형식으로 변환...")
    export_to_onnx(model, "davinci_ai.onnx")
    
    # 검증
    print("\n5. ONNX 모델 검증...")
    try:
        import onnx
        onnx_model = onnx.load("davinci_ai.onnx")
        onnx.checker.check_model(onnx_model)
        print("   ONNX 모델 유효성 검증 완료!")
    except ImportError:
        print("   (onnx 패키지 미설치 - 검증 건너뜀)")
    except Exception as e:
        print(f"   검증 실패: {e}")
    
    print("\n" + "=" * 50)
    print("완료! davinci_ai.onnx 파일이 생성되었습니다.")
    print("=" * 50)


if __name__ == "__main__":
    main()
