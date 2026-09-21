# MMMU-val Baseline Evaluation Report — Qwen3-VL-4B-Instruct

- **팀명**: _(기입)_
- **팀원**: _(기입)_
- **작성일**: _(기입)_
- **재현 커맨드**: `bash scripts/run_mmmu_eval.sh`

---

## 1. 환경 / 재현성

| 항목 | 값 |
|---|---|
| 모델 checkpoint | `Qwen/Qwen3-VL-4B-Instruct` (ebb281ec70b05090aa6165b016eac8ec08e71b17), bf16 (config.json `torch_dtype` 기본값, 양자화/캐스팅 없음) |
| 추론 백엔드 | vLLM 0.29.0 (torch 2.13.0+cu129, transformers _(lock 파일 값)_). 선택 이유: (1) Qwen 공식 recipe의 `presence_penalty`, `seed`를 `SamplingParams`가 그대로 지원 — HF `generate()`에는 `presence_penalty`가 없음. (2) 과목 단위 배치 스케줄링으로 900문제를 수 분 내 처리, 프롬프트 ablation 재실행이 저렴. (3) 4090(sm_89)은 bf16 네이티브라 vLLM dtype 제약 없음. |
| 사용 GPU | RunPod RTX 4090 24 GB (71 GB RAM, 20 vCPU) |
| 실측 peak VRAM | _(run_meta.json `peak_vram_mib`)_ — vLLM이 `gpu_memory_utilization=0.9`만큼 KV 캐시를 선점하므로 device 사용량은 ≈ 21.6 GB로 고정됨 |
| 총 소요 시간 | _(run_meta.json `total_elapsed_sec`, 900문제)_ |
| 의존성 | [requirements.txt](../requirements.txt) (pin) / [results/requirements.lock.txt](../results/requirements.lock.txt) (실측 freeze) |
| 실행 커맨드 | ```bash\nMODEL_PATH=Qwen/Qwen3-VL-4B-Instruct MODEL_REVISION=ebb281ec70b05090aa6165b016eac8ec08e71b17 \\\nDATA_ROOT=MMMU/MMMU DATA_REVISION=98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68 \\\nOUT_DIR=results/mmmu_baseline bash scripts/run_mmmu_eval.sh\n``` `MODEL_PATH`/`DATA_ROOT`는 로컬 디렉토리도 허용(로컬이면 revision 무시). 모든 선택은 [configs/mmmu_baseline.yaml](../configs/mmmu_baseline.yaml) 한 파일에 있고, 실행 시 `config_resolved.yaml`로 저장됨. |

## 2. 프롬프트

**실제 모델에 들어간 프롬프트 전문** (Qwen chat template의 user turn 내용, 변수는 `{}`):

객관식:
```
{question}

(A) {option_1}
(B) {option_2}
...

Answer with the option's letter from the given choices directly.
```
주관식(open):
```
{question}

Answer the question using a single word or phrase.
```

- `<image N>` 토큰 위치에 해당 이미지를 interleave (첫 등장 위치, 텍스트에서 참조되지 않은 이미지는 맨 앞에 추가). system prompt 없음.
- **출처**: MMMU 공식 평가 코드 `mmmu/configs/llava1.5.yaml` (`multi_choice_example_format`, `short_ans_example_format`) 및 `mmmu/utils/data_utils.py::construct_prompt`의 옵션 렌더링 `(A) ...` — https://github.com/MMMU-Benchmark/MMMU
- **선택 이유**: 벤치마크 제작자가 배포한 표준 템플릿이라 출처가 명확하고, 같은 저장소의 파서(4절)와 짝이 맞음. fine-tune 후 재평가에서도 그대로 재사용.

## 3. 생성(Decoding) 설정

### 3.1 Sampling recipe

| 파라미터 | 값 |
|---|---|
| `do_sample` | true |
| `temperature` | 0.7 |
| `top_p` | 0.8 |
| `top_k` | 20 |
| `repetition_penalty` | 1.0 |
| `presence_penalty` | 1.5 |
| `seed` | 3407 (요청별 seed = 3407 + 문제 전역 인덱스, resume 순서와 무관) |

- **출처**: (1) `Qwen/Qwen3-VL-4B-Instruct` @ebb281e `generation_config.json`: `do_sample=true, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.0`. (2) Qwen3-VL GitHub README "Generation Hyperparameters" (Instruct): 위 값 + `presence_penalty=1.5`, `seed=3407`, `greedy=false` — https://github.com/QwenLM/Qwen3-VL

### 3.2 생성 예산 / 이미지 해상도

| 파라미터 | 값 |
|---|---|
| `max_new_tokens` | 1024 |
| 이미지 해상도 처리 | 이미지당 `min_pixels=65536`(=64 토큰, 프로세서 기본 `shortest_edge`) ~ `max_pixels=2,097,152`(=2048 토큰). Qwen 프로세서와 동일한 `smart_resize`(32의 배수 반올림)로 사전 리사이즈. 문제당 비주얼 토큰 합이 12,288을 넘으면 모든 이미지를 비례 축소. `max_model_len=16384`. |

**선택 근거**: 프로세서 기본 `longest_edge=16,777,216`(이미지당 최대 ~16k 토큰)은 7장짜리 문제에서 컨텍스트를 초과하므로 상한 필요. MMMU 이미지는 대부분 1M 픽셀 미만이라 2048토큰 상한은 실제로 거의 걸리지 않음(축소된 문제 수는 `predictions/*.jsonl`의 `image_sizes`로 확인 가능). `max_new_tokens=1024`는 direct-answer에서는 수 토큰만 쓰지만 모델이 설명을 붙여도 잘리지 않게 여유를 둔 값이며 vLLM은 미사용 예산에 비용이 없음. 24 GB에서 가중치 ~9 GB + KV 캐시 ~12 GB로 30문제 동시 처리 가능.

## 4. 채점(파싱) 방식

- 사용한 파서/로직: MMMU 공식 `mmmu/utils/eval_utils.py`의 `parse_multi_choice_response` / `parse_open_response` / `eval_multi_choice` / `eval_open`을 그대로 포팅 ([src/mmmu_eval/parser.py](../src/mmmu_eval/parser.py)) — https://github.com/MMMU-Benchmark/MMMU
- 동작 방식 요약 (객관식): 응답 양끝 구두점 제거 → ① `(A)` 형태 탐색 → ② 없으면 ` A ` 형태 탐색 → ③ 없고 응답이 5단어 초과면 선택지 본문 텍스트 매칭 → 후보가 여러 개면 응답에서 **가장 뒤에** 나온 것 → 후보 0개면 무작위 선택(fallback). 주관식: 키 문장("answer ", "is " 등 뒤)과 숫자 추출 후 정규화하여 정답(복수 허용)과 포함 매칭.
- 공식 코드와의 차이: fallback 무작위 선택에 문제별 고정 seed를 써서 결정적으로 만들고 `parse_fallback` 플래그로 건수를 집계(5절). CoT ablation에서만 마지막 `Answer: X` 줄을 먼저 추출(MMMU-Pro 규약)한 뒤 같은 파서를 적용.

## 5. 결과

_(results/mmmu_baseline/scores.md 내용을 붙여넣기)_

| No. | Subject | Data Num | Acc |
|---|---|---|---|
| 1 | Accounting | 30 | |
| ... | | | |
| | **Overall (macro avg)** | **900** | |

계산식: `Overall = mean(30개 과목 accuracy)`; 과목당 30문제로 균등하므로 micro(맞힌 수/900)와 동일 — scores.json에 둘 다 기록.

## 6. 공식 수치와의 비교

| | Overall (MMMU val) |
|---|---|
| 공식 (Qwen3-VL Technical Report) | 67.4 |
| 우리 재현 결과 | |
| 차이 (Δ) | |

## 7. 격차 분석

_(1000자 이내. 후보 원인: (a) direct-answer 프롬프트 vs 공식 평가의 추론 허용 여부 — `results/ablation_cot` 결과와 비교, (b) sampling(temperature 0.7)로 인한 분산 — seed 고정이라 단일 표본, (c) 파싱 fallback 건수, (d) 이미지 해상도 상한에 걸린 문제 수, (e) 주관식 채점 규칙. `predictions/*.jsonl`에서 오답 샘플을 직접 열어 근거 제시.)_

## 8. 기타 특이사항 / 한계 (Optional)

_(기입)_
