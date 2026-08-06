# Splendor Self-Play Environment

## Two-player PPO league v0.5.0

The league keeps a continuously trained Candidate, a gated frozen Champion, permanent
Hall-of-Fame Champions, and capped recent Candidate snapshots. It uses PFSP to favor
useful historical opponents while retaining current-policy self-play.

```powershell
python .\experiments\train_league_ppo.py --config .\configs\league_ppo_2p_smoke.yaml --run-dir .\runs\league_ppo_2p_v050_smoke --device cpu --progress always
python .\experiments\train_league_ppo.py --config .\configs\league_ppo_2p_1m.yaml --run-dir .\runs\league_ppo_2p_1m_seed42 --initial-checkpoint .\runs\shared_ppo_2p_1m_seed42\checkpoints\best_average_rank.pt --device cuda --progress always
python .\experiments\evaluate_league.py --run-dir .\runs\league_ppo_2p_1m_seed42 --games-per-matchup 1000 --device cuda --progress always --hall-of-fame --matchup-matrix
```

This is PPO-based league self-play, not an AlphaGo Zero or AlphaZero implementation:
there is no MCTS and no tree-search visit-count target. See
[league self-play](docs/league_selfplay.md), [promotion](docs/champion_promotion.md),
and [PFSP](docs/pfsp.md).

## Two-player PPO v0.4.3

```powershell
python experiments/train_shared_ppo.py --config configs/shared_ppo_2p_smoke.yaml --run-dir runs/shared_ppo_2p_smoke --device cpu
python .\experiments\train_shared_ppo.py --config .\configs\shared_ppo_2p_1m.yaml --run-dir .\runs\shared_ppo_2p_1m_seed42 --device cuda --progress always --stop-at-transitions 114688
python .\experiments\train_shared_ppo.py --config .\configs\shared_ppo_2p_1m.yaml --run-dir .\runs\shared_ppo_2p_1m_seed42 --resume .\runs\shared_ppo_2p_1m_seed42\checkpoints\step_000114688.pt --device cuda --progress always --stop-at-transitions 311296
python .\experiments\evaluate_shared_ppo.py --checkpoint .\runs\shared_ppo_2p_1m_seed42\checkpoints\best_average_rank.pt --games-per-matchup 1000 --output-dir .\runs\shared_ppo_2p_1m_seed42\final_evaluation --device cuda --actor-only --progress always
```

See [two-player experiments](docs/two_player_experiments.md). The 1M schedule's first 100k threshold is reached at 114,688 transitions (7 updates).
Progress uses stderr, is automatically disabled when redirected, and can be disabled explicitly with `--progress never`.

## v0.4.1 PPO stabilization

This patch adds correct post-truncation bootstrap values, periodic evaluation, numbered and best checkpoints, linear LR decay, epoch-mean target-KL stopping, and v0.4.0 checkpoint compatibility. The critic sees private reservation payloads but not full hidden deck order.

```powershell
python experiments/train_shared_ppo.py --config configs/shared_ppo_4p_1m.yaml --run-dir runs/shared_ppo_4p_1m_seed42
python experiments/train_shared_ppo.py --config configs/shared_ppo_4p_1m.yaml --run-dir runs/shared_ppo_4p_1m_seed42 --resume runs/shared_ppo_4p_1m_seed42/checkpoints/latest.pt
python experiments/evaluate_shared_ppo.py --checkpoint runs/shared_ppo_4p_1m_seed42/checkpoints/best_average_rank.pt --games-per-matchup 500 --output-dir runs/shared_ppo_4p_1m_seed42/final_evaluation
```

## Shared PPO v0.4.0

Install the optional learner and run the pipeline smoke test:

```powershell
pip install -e ".[rl]"
python experiments/train_shared_ppo.py --config configs/shared_ppo_smoke.yaml --run-dir runs/shared_ppo_smoke
```

Four-player self-play and fixed-bot evaluation:

```powershell
python experiments/train_shared_ppo.py --config configs/shared_ppo_4p.yaml --run-dir runs/shared_ppo_seed42
python experiments/evaluate_shared_ppo.py --checkpoint runs/shared_ppo_seed42/checkpoints/latest.pt --games-per-matchup 500 --output-dir runs/shared_ppo_seed42/final_evaluation
python experiments/watch_checkpoint_game.py --checkpoint runs/shared_ppo_seed42/checkpoints/latest.pt --players 4 --opponents greedy shortest blocking --seed 123 --perspective omniscient --manual --step-mode turn
```

See [Shared PPO](docs/shared_ppo.md), [player trajectories](docs/player_trajectory.md), and the [CTDE critic](docs/ctde_critic.md). Smoke updates validate the pipeline only; they do not demonstrate strong play.

## v0.3.2 turn-aware replay

```powershell
python examples/export_visual_replay.py --players 4 --agents greedy greedy random random --seed 42 --output-dir runs/v032_demo
python -m splendor_env.replay runs/v032_demo/game.json --verify
python -m splendor_env.replay runs/v032_demo/game.json --turn-only
python -m splendor_env.replay runs/v032_demo/game.json
```

Turn mode groups purchase, payment, discard, and noble decisions into one player turn. Decision mode exposes each choice. See [time semantics](docs/time_semantics.md), [v0.3.2 schema](docs/log_schema_v032.md), and [legacy migration](docs/legacy_log_migration.md).

```powershell
python -m splendor_env.migrations.migrate_logs_v032 runs/old_logs --output-dir runs/migrated_v032 --dry-run
python -m splendor_env.migrations.migrate_logs_v032 runs/old_logs --output-dir runs/migrated_v032 --recursive --verify
```

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
- 고정 `Discrete(373)` 행동 공간과 phase별 `action_mask`
- 자유 골드 결제, 조합 단위 토큰 반환, 복수 귀족 선택
- JSON/CSV 경기 기록, 최종 패 요약, 관점별 replay와 SHA-256 재현 검증
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

## v0.3 경기 기록과 리플레이

```bash
python examples/generate_demo_games.py \
    --players 4 \
    --games 5 \
    --seed 100 \
    --output-dir runs/demo_games \
    --record-level full
```

최종 패는 `runs/demo_games/game_0000_final_summary.txt`, 여러 경기 표는 `games_summary.csv`에서 확인합니다.

```bash
# 상태 hash와 최종 보유 상태 재현 검증
python -m splendor_env.replay runs/demo_games/game_0000.json --verify

# 전체 공개 턴 단위 자동 재생
python -m splendor_env.replay runs/demo_games/game_0000.json --omniscient --turn-only --delay 0.4

# P2 관점 수동 재생
python -m splendor_env.replay runs/demo_games/game_0000.json --perspective 2 --step
```

`summary`, `actions`, `full` 기록 레벨을 지원합니다. `full`은 decision event와 실제 턴 종료 snapshot을 함께 저장합니다. recorder는 passive listener이므로 observation, reward, action mask를 바꾸지 않습니다.

## 경기 시각화

기록 JSON을 외부 서버나 CDN이 필요 없는 single-file HTML replay로 변환합니다.

```bash
python -m splendor_env.visualization.html_export \
    runs/demo_games/game_0000.json \
    --output runs/demo_games/game_0000_viewer.html
```

브라우저 상단에서 Current player, Player 0~3, Omniscient 관점과 table/egocentric 배치를 선택할 수 있습니다. 이전·다음 decision/turn, 자동 재생, slider, 속도, 구매 카드 상세, 디버그 card ID 표시를 지원합니다.

정보 유출 검사용 export는 숨은 카드 데이터 자체를 제거합니다.

```bash
python -m splendor_env.visualization.html_export \
    runs/demo_games/game_0000.json \
    --output runs/demo_games/p1_viewer.html \
    --data-mode perspective-sanitized-data \
    --perspective 1
```

최종 패 비교:

```bash
python -m splendor_env.visualization.compare_games \
    runs/demo_games/game_0000.json \
    runs/demo_games/game_0001.json \
    runs/demo_games/game_0002.json \
    --output runs/demo_games/compare_final_boards.html
```

한 경기를 생성하고 즉시 시각화하려면:

```bash
python examples/export_visual_replay.py \
    --players 4 \
    --agents greedy greedy random random \
    --seed 42 \
    --output-dir runs/visual_demo
```

브라우저에서 `runs/visual_demo/game_viewer.html`을 열면 됩니다.

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
- `discard`: 정확히 초과한 수량의 토큰 반환 조합 선택
- `noble`: 만족한 귀족 중 하나 선택

이 때문에 동시 행동용 Parallel API가 아니라 턴 순서를 정확히 표현하는 AEC API를 사용합니다.

## 행동 공간: 373개

| 범위 | 행동 |
|---:|---|
| 0–24 | 서로 다른 색 1~3개 조합 가져오기 |
| 25–29 | 같은 색 토큰 2개 가져오기 |
| 30–41 | 공개 카드 구매 |
| 42–53 | 공개 카드 예약 |
| 54–56 | 티어 덱에서 비공개 예약 |
| 57–59 | 내 예약 카드 구매 |
| 60–311 | 결정적으로 정렬된 결제 plan index (최대 252) |
| 312–367 | 결정적으로 정렬된 반환 plan index (최대 56) |
| 368–372 | 귀족 선택 |

공식 pass 행동은 없습니다. 정상 상태에서 합법 행동이 0개면 `NoLegalActionError`가 발생합니다. 정책은 373개 로짓을 출력합니다.

```python
masked_logits = logits.masked_fill(action_mask == 0, -1e9)
```

## 관측 공간

```python
Dict(
    {
        "observation": Box(0, 1, shape=(475,), dtype=float32),
        "action_mask": MultiBinary(373),
    }
)
```

475차원 벡터는 215차원 전역 보드 블록과 최대 4명의 65차원 플레이어 블록으로 구성됩니다. `OBS_LAYOUT`에 주요 slice가 정의되어 있습니다.

- 5-way phase, 종료 여부, final round 여부, round 진행률
- pending purchase의 공개 가능한 카드 정보, 결제 plan 수, 반환 초과량
- 은행 토큰과 티어별 남은 덱 크기
- 공개 카드 12장
- 귀족 최대 5개
- 최대 4명의 토큰, 영구 보너스, 점수, 구매 카드 수, 귀족 수
- 각 플레이어 예약 카드 최대 3장: 존재 여부, 예약 출처, 티어, 카드 payload
- 공개 시장에서 예약한 카드는 모든 플레이어에게 전체 정보 공개
- 덱에서 예약한 카드는 소유자에게 전체 정보, 상대에게 존재·비공개 출처·티어만 공개

`game.unwrapped.state()`는 모든 비공개 예약의 payload까지 포함하는 475차원 omniscient state를 반환합니다.

## 공식 종료와 안전 truncation

`SplendorGame(..., seed=...)`에는 공식 15점/라운드 종료만 존재합니다. 공식 pass와 max-turn 종료는 없습니다. PettingZoo `env(max_turns=None)`가 공식 모드이며, 숫자를 지정한 경우에만 어댑터가 `max_turns_truncation`으로 episode를 안전 종료합니다.

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
- `splendor_env/actions.py`: 373개 phase별 행동 정의 및 설명
- `splendor_env/data.py`: 90장 카드와 10개 귀족 수치
- `splendor_env/recording.py`: JSON/CSV 기록과 최종 결과 저장
- `splendor_env/replay.py`: 관점별 replay와 action 재실행 검증 CLI
- `splendor_env/agents/`: random/greedy 검증 에이전트
- `examples/generate_demo_games.py`: 여러 경기 기록 생성
- `examples/random_self_play.py`: 합법 행동 마스크를 이용한 실행 예제
- `examples/reservation_visibility_demo.py`: 관점별 예약 카드 공개 범위 예제
- `tests/test_core.py`: 보존 법칙 및 랜덤 롤아웃 테스트

## 주의

이 프로젝트는 Space Cowboys/Asmodee의 공식 소프트웨어가 아닌 연구·교육 목적의 독립 구현입니다. 게임 이름과 규칙은 해당 권리자에게 귀속됩니다.
