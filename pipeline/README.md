# pipeline/: 최종 제출 파일을 만든 스크립트

최종 제출 `blend_v40b8a4`(public 9.9433, private 10.6187)를 만든 스크립트 115개(649,943 bytes)를 작업 캐시에서 그대로 옮겨 온 사본이다.
폴더 구조는 캐시의 `scripts/` 아래 상대 경로를 그대로 따른다. 파일별 원본 md5와 이 사본의 md5는 [MANIFEST.tsv](MANIFEST.tsv)에 있다.

경로 표기:

| 기호 | 원래 위치 | 이 저장소에서 |
|---|---|---|
| S | `/mnt/project-files/work/hardness-cache/scripts` | `pipeline/` |
| C | `/mnt/project-files/work/hardness-cache` (중간 산출물, 제출 파일) | 없음 |
| R | `/home/claude/microstructure-hardness-prediction` | 저장소 루트 (`src/`, `tools/`) |

## 성격: 실행 패키지가 아니라 기록

- 코드 94개 중 89개가 S, C, R 같은 절대 경로를 직접 쓴다. 원래 환경(같은 경로의 캐시와 데이터)에서만 그대로 돈다.
- 여러 스크립트가 다른 파일의 sha256을 확인한 뒤에야 실행되므로(아래 표), 고치지 않고 바이트 단위로 복사했다.
- 대회 데이터, 특징 표, 판독기 출력, 제출 파일 같은 데이터는 하나도 들어 있지 않다. [.gitignore](.gitignore)가 `*.csv`, `*.parquet`, `*.npz`, `*.npy`, `*.json`, `*.pt`, `*.log`를 막는다.

## 최종 파일이 만들어진 순서

| 단계 | Public | 스크립트 (실행 순서) | 바뀐 것 |
|---|---|---|---|
| v21 | 11.0620 | 저장소 `src/`: features, extract_embeddings → train_gbm, train_head, train_cnn → `python -m src.blend_cells` (v21base, 셀 CNN 비중 0.8) → `python -m src.het_offset --base blend_v21base --out blend_v21` | 기본 블렌드 + 깨끗한 이미지 het4 보정 |
| v22b | (v22 10.9742) | `het-1008/stage2b/stage2b.py` → `build_v22.py` | 노이즈 이미지에 het 판독기 보정 |
| v24 | 10.5130 | `d1009/G/g_test.py` (사전 등록 검증) → `g_build.py` | 국소화 판독기를 G1 전체에 |
| v25 | 10.2394 | `d1009/J/cleanhead/{seg_c,feats_c,feats_lines}.py` → `d1009/J/j_build.py` | 깨끗한 이미지에 formula7 |
| v26r | 10.2136 | `d1010/C/residual-probe/{build_segalt,build_avg5}.py` → `d1010/R/r_build.py` | 깨끗한 이미지에 formula8 |
| v26L | 10.1302 | `d1010/L/lv_feats.py` → `l_test.py` → `l_build.py` | 경계선 보이는 중간 노이즈 이미지에 formula7 |
| v27u | 10.1041 | `d1010/U/u_build.py` | v26r과 v26L의 합집합 |
| v28n | 10.0712 | `d1010/D/A/a_test.py` (사전 등록 검증) → `a_build.py` | G1 덜 심한 절반의 기울기 확대 |
| v29g | 10.0564 | `d1011/bold/impl_C4/build_u1.py` → `d1011/bold/plan_bold/build_v29g.py` | het4를 분할 6종 평균으로 재측정 (C1, C2, C3) |
| v30b | 10.0207 | `d1011/N4/g1/run_nested.py` → `build_g1_test.py` → `d1011/N4/g1clip/clip_g1.py` → `build_v30ab.py` | G1 행에 A3 판독기 보정, 기울기 0.75, test 보정값을 train 범위로 자름 |
| 판독기 | | 노트북 GPU: 저장소 `src/het_cnn.py`, `tools/fsim/het_fsim.py`. 명령은 [notes/handoff/laptop-gpu.md](../notes/handoff/laptop-gpu.md) 섹션 21-3, 23-1/4/5 (B8f), 24-10, 27-7/8/9 (AP4f). 결과는 `src/het_cnn.py --ingest`로 `data/het_cnn/<run>/`에 들어간다 | B8f = B5 Noisy Student + B1 레시피 3시드, AP4f = B4 AdvProp + A3 레시피 3시드 |
| **v40b8a4** | **9.9433** | `d1018/g1mix11/build_g1mix11.py`, line B (`BASE=B8f`), 옵션 P4A = [B8f, AP4f] | v30b에서 G1 행 433개의 판독기만 교체 |

최종 파일 다시 만들기 (원래 환경에서):

```bash
bash S/d1018/rebuild_pool1019.sh S/d1018/run_env_1019b4.sh v40b8a4 9.9433362872 <새 태그>
# -> S/d1018/g1mix11/<태그>/lineB/P4A/test.csv  (md5 eac611155489d3ab70018297f5b694ce)

# 같은 결과, 빌더 하나만:
cd R && source S/d1018/run_env_1019b4.sh v40b8a4 9.9433362872 && export BASE=B8f && unset ALLOW_PARTIAL \
  && mkdir -p OUTDIR && OUT=OUTDIR python3 -B S/d1018/g1mix11/build_g1mix11.py     # -> OUTDIR/P4A/test.csv
```

- 원래 빌드는 `run_env_1019k30.sh`(태그 q8a4b_k30)에서 만들어졌고, `submissions/`로는 손으로 복사했다. `run_env_1019b4.sh`로 다시 빌드해도 md5가 같다.
- 빌더는 v30b의 OOF와 test, 이전 최고 파일들을 비트 단위로 재현하는지 먼저 확인한다.
- 환경 변수 중 결과 내용을 바꾸는 것은 `BASE`뿐이다. 나머지는 비교 대상(현재 최고 파일, 제출 목록, public 점수 모델)만 바꾼다.
- `rebuild_pool1019.sh`는 최종 파일과 관계없는 다른 후보 묶음(g1mix10, g1mix9, g1mix8)도 함께 빌드한다(22~27행). g1mix10, g1mix9 빌더는 여기 넣지 않았고, 최종 파일에는 영향이 없다.

## 폴더별 묶음

| 묶음 | 파일 | 내용 |
|---|---|---|
| A-core | 20 | 최종 빌더(`d1018/g1mix11/build_g1mix11.py`, `options.txt`), G1 보정 라이브러리(`d1011/N4/g1/g1core.py`), v29g·v30b 하네스(`n4lib.py`, `n5lib.py`, `lib_l4.py`), public 점수 모델(`d1016/pub/kriging*.py`, `forensics*.py`) |
| A-env | 16 | 환경 파일 사슬 `run_env_*.sh` (각 파일이 이전 파일을 불러온다; 판독기 이름, md5, public 점수만 담김) |
| A-provenance | 8 | 빌더 생성 이력: `mk_g1mix11.py` ← `build_g1mix7.py` ← `build_g1mix6.py`와 각 `options.txt` |
| B-chain | 43 | 위 표의 v22b ~ v30b 단계, 실험 결과 수집 스크립트 `ingest.py` |
| C-producers | 20 | 캐시된 중간 산출물을 만든 스크립트 (m3b 행 표, L1 후보, L8 입력, 축소 검증 결과, 분할 대안) |
| D-decision | 8 | 제출 판정 도구: public 점수 모델 생성(`mk_pubN.py`), 예측(`kriging31.py`, `fullpred28.py`), 판정(`d1018/judge_slots1018.py`, `judge_bold_fp1018.py`) |

## sha256으로 고정된 파일

아래 파일은 다른 스크립트가 sha256을 확인하므로 한 글자도 바꾸면 안 된다.

| 고정된 파일 | sha256 앞자리 | 확인하는 곳 |
|---|---|---|
| `d1018/g1mix11/options.txt` | bb568132 | `build_g1mix11.py` 19~20행 |
| `d1011/N4/g1/options.txt` | c4bd0675 | `g1core.py` 22, 37행 |
| `d1016/g1mix7/build_g1mix7.py` | 2e22250a | `mk_g1mix11.py` 6행 |
| `d1016/g1mix6/build_g1mix6.py` | 3408758b | `mk_g1mix7.py` 6행 |
| `d1016/g1mix7/options.txt` | 2a7f15a0 | `build_g1mix7.py` 16~17행 |
| `d1016/g1mix6/options.txt` | a585cfbd | `build_g1mix6.py` 12~13행 |
| `d1010/D/A/a_test.py` | be3d25d3 | `a_test.py` 190행, `a_build.py` 59행 (`a_test.sha256`) |
| `d1010/D/A/PREREG.txt` | 123b281d | `a_test.py` 48, 193행, `a_build.py` 37, 62행 (`d1010/D/PREREG.sha256`, 절대 경로 기준) |
| `d1010/L/l_test.py` | 030e51b1 | `l_build.py` 68행 |

## 알려진 문제와 빠진 것

- `d1010/D/A/PREREG.txt`는 v28n 빌드 뒤에 내용이 덧붙여져 지금 sha256이 고정값과 다르다(a5475c1a…). 그래서 `a_test.py`, `a_build.py`는 다시 실행하면 멈춘다. 등록 당시 사본은 찾지 못했다.
- 다음 코드는 남아 있지 않다.
  - v21base의 CNN test 결합(5-fold 평균과 노트북 전체 데이터 학습의 반반).
  - 노트북 GPU 학습 실행 스크립트. 명령만 `notes/handoff/laptop-gpu.md`에 있다.
  - `d1009/J/cleanhead`의 parquet를 캐시로 옮긴 과정(손으로 복사).
  - 최종 파일을 `submissions/`로 옮긴 과정(손으로 복사).
  - V4, FD6t, m3b 계산을 저장소 `src/`로 옮기는 작업(하지 못함).
- `d1011/bold/impl_C4/build_u1.py`에는 미리 정한 점검이 하나 있다. C2 단계의 test 예측 변화 표준편차가 train 변화의 0.5~2.0배 안에 있는지 보고, 벗어나면 C2를 쓰지 않는다. 점검은 통과했고, 이 단계에서 test로 맞춘 값은 없다.

## 이 사본에서 바꾼 것

두 파일에서 test 이미지 한 장에 대한 메모만 지웠다. 둘 다 sha256으로 고정된 파일이 아니고, 나머지 113개는 캐시 원본과 바이트 단위로 같다.

- `d1010/L/PREREG.txt`: test 이미지 한 장의 예측값과 변화량을 적은 문장.
- `d1011/bold/plan_bold/build_v29g.py`: 결과 JSON의 메모에만 들어가는 test 이미지 한 장의 변화량 항목.
