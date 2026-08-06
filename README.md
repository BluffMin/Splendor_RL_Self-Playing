# Splendor Self-Play Environment

스플렌더 기본판을 강화학습 self-play에 사용할 수 있도록 만든 **독립 규칙 엔진 + PettingZoo AEC 환경**입니다.
그림이나 상표 자산은 포함하지 않고 카드/귀족의 수치와 게임 규칙만 구현했습니다.

## 포함된 기능

- 2~4인 플레이
- 기본판 개발 카드 90장, 귀족 10개
- 플레이어 수에 따른 컬러 토큰 수: 2인 4개, 3인 5개, 4인 7개
- 12장 공개 시장, 티어별 비공개 덱
- 서로 다른 색 토큰 최대 3개 가져오기
- 같은 색 2개 가져오기: 행동 전 은행에 4개 이상 있을 때만 가능
- 공개 카드 구매/예약, 티어 덱 맨 위 카드 비공개 예약
- 골드 조커 결제 및 골드가 없어도 예약 가능
- 예약 카드 최대 3장
- 10개를 초과한 토큰의 강제 반납
- 여러 귀족을 동시에 만족할 때 에이전트가 직접 하나 선택
- 15점 도달 후 현재 라운드를 끝까지 진행
- 동점 시 구매한 개발 카드가 더 적은 플레이어 우선
- 부분 관측: 공개 시장 예약 카드는 모두에게 공개하고, 덱 예약 카드는 상대에게 티어만 공개
- 중앙집중식 critic용 `state()` 제공
- 고정 `Discrete(324)` 행동 공간과 `action_mask`
- sparse terminal reward 또는 zero-sum score shaping
- ANSI/human 텍스트 렌더링

## 설치

```bash
cd splendor-selfplay-env
pip install -e .
```

개발 및 테스트 의존성까지 설치하려면:

```bash
pip install -e ".[dev]"
pytest
```

## 가장 빠른 실행

```bash
python examples/random_self_play.py
python examples/reservation_visibility_demo.py
```

직접 코드에서 사용할 때:

```python
import numpy as np
from splendor_env import env

game = env(num_players=2, reward_mode="sparse", render_mode="ansi")
game.reset(seed=42)

for agent in game.agent_iter():
    observation, reward, terminated, truncated, info = game.last()

    if terminated or truncated:
        action = None
    else:
        mask = observation["action_mask"]
        legal_actions = np.flatnonzero(mask)
        action = int(np.random.choice(legal_actions))

    game.step(action)

print(game.render())
game.close()
```

PettingZoo의 AEC 루프에서는 보상이 해당 에이전트의 **다음 `last()` 호출**에서 전달됩니다. 실제 replay buffer를 만들 때는 에이전트별 pending transition을 따로 유지하는 것이 안전합니다.

## Self-play에 유리한 설계

모든 플레이어 관측은 **자기 자신을 첫 번째 플레이어 슬롯**에 놓는 상대적 관점으로 인코딩됩니다. 따라서 2인전에서는 하나의 네트워크를 양쪽 좌석이 공유하는 parameter-sharing self-play를 바로 적용할 수 있습니다.

스플렌더는 한 턴 안에서도 토큰 반납 또는 귀족 선택이라는 추가 의사결정이 생깁니다. 이 구현은 그 선택을 자동 처리하지 않고 같은 에이전트가 연속해서 행동하게 합니다.

- `normal`: 일반 행동
- `payment`: 구매 시 색 토큰과 골드의 지불 조합 선택
- `discard`: 토큰 하나씩 반납
- `noble`: 만족한 귀족 중 하나 선택

이 때문에 동시 행동용 Parallel API가 아니라 턴 순서를 정확히 표현하는 AEC API를 사용합니다.

## 행동 공간: 324개

| 범위 | 행동 |
|---:|---|
| 0–24 | 서로 다른 색 1~3개 조합 가져오기 |
| 25–29 | 같은 색 토큰 2개 가져오기 |
| 30–41 | 공개 카드 구매 |
| 42–53 | 공개 카드 예약 |
| 54–56 | 티어 덱에서 비공개 예약 |
| 57–59 | 내 예약 카드 구매 |
| 60–65 | 토큰 하나 반납 |
| 66–70 | 귀족 선택 |
| 71 | 공식 행동이 하나도 없을 때만 가능한 deadlock pass |
| 72–323 | 카드 구매 시 골드 대체 지불 조합 선택 |

정책은 324개 로짓을 출력하고, `action_mask == 0`인 로짓을 매우 작은 값으로 바꾼 뒤 categorical sampling 또는 argmax를 수행하면 됩니다.

```python
masked_logits = logits.masked_fill(action_mask == 0, -1e9)
```

## 관측 공간

```python
Dict({
    "observation": Box(0, 1, shape=(454,), dtype=float32),
    "action_mask": MultiBinary(324),
})
```

454차원 벡터는 194차원 전역 보드 블록과 최대 4명의 65차원 플레이어 블록으로 구성됩니다.

- 현재 phase, 종료 여부, final round 여부, 진행률
- 은행 토큰과 티어별 남은 덱 크기
- 공개 카드 12장
- 귀족 최대 5개
- 최대 4명의 토큰, 영구 보너스, 점수, 구매 카드 수, 귀족 수
- 각 플레이어 예약 카드 최대 3장: 존재 여부, 예약 출처, 티어, 카드 payload
- 공개 시장에서 예약한 카드는 모든 플레이어에게 전체 정보 공개
- 덱에서 예약한 카드는 소유자에게 전체 정보, 상대에게 존재·비공개 출처·티어만 공개

`game.unwrapped.state()`는 모든 비공개 예약의 payload까지 포함하는 454차원 omniscient state를 반환합니다.

텍스트 렌더링도 같은 공개 규칙을 따릅니다. PettingZoo 환경에서 전체 카드를 확인하는 디버그 렌더링이 필요하면 `render_omniscient=True`를 지정합니다.

```python
game = env(
    num_players=3,
    render_mode="human",
    render_omniscient=True,
)
```

## 보상

### `reward_mode="sparse"` 권장 시작점

- 승자: `+1`
- 패자: zero-sum이 되도록 음수 보상
- 완전 동점: `0`

2인전에서는 일반적인 `+1 / -1`입니다.

### `reward_mode="score"`

prestige를 얻을 때 행동한 플레이어에게 `shaping_scale × 획득 점수`를 주고, 상대에게 합계가 같도록 음수로 분배합니다. 최종 승패 보상도 추가됩니다.

초기 실험은 다음 순서가 좋습니다.

1. random-vs-random으로 규칙과 평균 게임 길이 확인
2. shared DQN + action mask + sparse reward
3. 학습이 느리면 `score` shaping 비교
4. 최근 체크포인트를 opponent pool에 저장하는 league self-play
5. 최종 평가는 고정된 heuristic/random/과거 checkpoint 묶음으로 수행

## 핵심 파일

- `splendor_env/core.py`: 외부 RL 라이브러리와 독립적인 규칙 엔진
- `splendor_env/pettingzoo_env.py`: PettingZoo AEC 어댑터
- `splendor_env/actions.py`: 324개 행동 정의 및 설명
- `splendor_env/data.py`: 90장 카드와 10개 귀족 수치
- `examples/random_self_play.py`: 합법 행동 마스크를 이용한 실행 예제
- `examples/reservation_visibility_demo.py`: 관점별 예약 카드 공개 범위 예제
- `tests/test_core.py`: 보존 법칙 및 랜덤 롤아웃 테스트

## 주의

이 프로젝트는 Space Cowboys/Asmodee의 공식 소프트웨어가 아닌 연구·교육 목적의 독립 구현입니다. 게임 이름과 규칙은 해당 권리자에게 귀속됩니다.
