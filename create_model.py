import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import onnx
import os
import random

# ==========================================
# 1. 모델 아키텍처 (52 -> 128 -> 128 -> 64 -> 24)
# ==========================================
class DaVinciRLModel(nn.Module):
    def __init__(self):
        super(DaVinciRLModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(52, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 24)
        )

    def forward(self, x):
        return self.network(x)

# ==========================================
# 2. 강화학습 환경 (Environment)
# ==========================================
class DaVinciEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        # 0-11 Black (0-11), 0-11 White (12-23)
        self.deck = list(range(24))
        random.shuffle(self.deck)
        
        # 4개씩 분배
        self.ai_hand = sorted(self.deck[:4])
        self.opp_hand = sorted(self.deck[4:8])
        self.deck = self.deck[8:]
        
        self.ai_revealed = [False] * 24
        self.opp_revealed = [False] * 24
        self.turn_count = 0
        return self.get_state()

    def get_state(self):
        state = np.zeros(52, dtype=np.float32)
        # 내 타일 [0-23]
        for idx in self.ai_hand:
            state[idx] = 1.0
        # 상대 공개 타일 [24-47]
        for i in range(24):
            if self.opp_revealed[i]:
                state[24 + i] = 1.0
        # 정보 [48-51]
        state[48] = self.turn_count / 20.0
        state[49] = len(self.ai_hand) / 12.0
        state[50] = len(self.opp_hand) / 12.0
        state[51] = len(self.deck) / 24.0
        return state

    def step(self, action_idx):
        # action_idx: 0-23 (추측할 카드 번호)
        reward = 0
        done = False
        
        # 유효한 추측인지 확인 (상대 패에 있고 아직 안 알려진 카드)
        if action_idx in self.opp_hand and not self.opp_revealed[action_idx]:
            self.opp_revealed[action_idx] = True
            reward = 1.0 # 성공 보상
            if all(self.opp_revealed[h] for h in self.opp_hand):
                reward += 5.0 # 장기적 보상 (승리)
                done = True
        else:
            reward = -1.0 # 실패 패널티
            # 실패 시 내 카드 중 하나 공개 (편의상 첫 번째 비공개 카드)
            for h in self.ai_hand:
                if not self.ai_revealed[h]:
                    self.ai_revealed[h] = True
                    break
            if all(self.ai_revealed[h] for h in self.ai_hand):
                reward -= 2.0 # 패배 패널티
                done = True
        
        self.turn_count += 1
        if self.turn_count >= 30: done = True
        
        return self.get_state(), reward, done

# ==========================================
# 3. 강화학습 (Policy Gradient 기반 단순화)
# ==========================================
def train_rl(episodes=2000):
    env = DaVinciEnv()
    model = DaVinciRLModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"강화학습 시작: {episodes} 에피소드...")
    
    for ep in range(episodes):
        state = env.reset()
        log_probs = []
        rewards = []
        
        done = False
        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            probs = torch.softmax(model(state_t), dim=1)
            
            # Action sampling
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            log_prob = m.log_prob(action)
            
            next_state, reward, done = env.step(action.item())
            
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state

        # Update policy
        R = 0
        policy_loss = []
        returns = []
        for r in rewards[::-1]:
            R = r + 0.9 * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-6)
        
        for log_prob, Gt in zip(log_probs, returns):
            policy_loss.append(-log_prob * Gt)
        
        optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()
        
        if (ep + 1) % 500 == 0:
            print(f"   Episode {ep+1}: Last Total Reward = {sum(rewards):.2f}")

    print("강화학습 완료!")
    return model

# ==========================================
# 4. 저장 및 검증
# ==========================================
def main():
    model = train_rl(3000)
    model.eval()
    
    # 파라미터 확인
    total_params = sum(p.numel() for p in model.parameters())
    weight_sum = sum(p.sum().item() for p in model.parameters())
    print(f"--- 모델 상태 확인 ---")
    print(f"총 파라미터 수: {total_params}")
    print(f"가중치 합계: {weight_sum:.4f}")
    
    output_path = "davinci_ai.onnx"
    dummy_input = torch.randn(1, 52)
    
    # [변경] Opset 12로 하향 조정 (호환성 극대화)
    print(f"4. ONNX 내보내기 중 (Opset 12)...")
    torch.onnx.export(
        model, dummy_input, output_path,
        export_params=True, 
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    import onnx
    import os
    if not os.path.exists(output_path):
        print("❌ 모델 파일 생성 실패!")
        return

    file_size = os.path.getsize(output_path) / 1024
    print(f"✅ 생성된 파일 크기: {file_size:.2f} KB")

    try:
        onnx_model = onnx.load(output_path)
        # 만약 IR 버전이 너무 높으면 강제로 낮춤 (ORT-Web 호환성)
        if onnx_model.ir_version > 8:
            print(f"⚠️ IR 버전이 너무 높음 ({onnx_model.ir_version}) -> 8로 조정")
            onnx_model.ir_version = 8
            onnx.save(onnx_model, output_path)

        onnx.checker.check_model(onnx_model)
        print(f"✅ 모델 검증 완료: {output_path}")
        print(f"✅ 최종 IR Version: {onnx_model.ir_version}")
        print(f"✅ 최종 Opset Version: {onnx_model.opset_import[0].version}")
    except Exception as e:
        print(f"❌ 검증 실패: {e}")

    print("\n🎉 모든 작업이 완료되었습니다!")

if __name__ == "__main__":
    main()
