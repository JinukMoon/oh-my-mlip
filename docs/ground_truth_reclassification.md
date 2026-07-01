# Ground Truth: 5-bucket root-cause 재분류 (Phase 0)

> 단일 진실원: 보존된 `.sweep/logs/<env>.verify.stderr.log` (2026-06-28 sweep). 본 문서는 GJC_HANDOFF §3의 4-bucket "추정" 분류를 **실측 stderr 근거**로 5-bucket 재분류한 것이다. 진단축(이 문서) ≠ 복구축(`AGENTS.md §8`) — CR-2.

## 측정 환경
- 호스트: WSL, RTX 4060 Ti 16GB **sm89** (compute cap 8.9), nvcc 12.8, conda 25.7.
- sweep: `scripts/sweep_local.py` (build→verify→delete). **env는 테스트 후 삭제됨** → 현재 `envs/`에 빌드된 env 없음(stderr 로그만 보존).
- 결과 원본: `.sweep/results.md` (4 PASS / 1 cpu_fallback / 13 fail / 2 gated-skip), `.sweep/results.jsonl` (20행).

## 5-bucket 분류표 (진단축 — observability/분류 전용, CR-2)

| env | model | 실측 stderr 핵심 | bucket | 비고 |
|---|---|---|---|---|
| mace | MACE-MPA-0 | (energy+force, GPU PID) | — PASS | tier-2 PASS |
| sevennet | SevenNet-MF-OMPA | (energy+force, GPU PID) | — PASS | tier-2 PASS |
| orb | ORB-v3 | (energy+force, GPU PID) | — PASS | tier-2 PASS |
| chgnet | CHGNet-v0.3.0 | (energy+force, GPU PID) | — PASS | tier-2 PASS |
| deepmd | DPA-3.1-3M-FT | `dpa-3.1-3m-ft.pth does not exist` | **A** weight-fetch | on-demand-hf, 경로 누락 |
| grace | GRACE-2L-OAM | `SavedModel file does not exist` | **A** weight-fetch | TF SavedModel 미fetch |
| pet | PET-OAM-XL | `pet-oam-xl-v1.0.0.ckpt does not exist` | **A** weight-fetch | .ckpt 미fetch (단, .ckpt≠.pt fingerprint 알려진 이슈) |
| dpa4 | DPA-4.0.1-pro-MPtrj | `FileNotFoundError(2)` | **A** weight-fetch | 경로 불명 weight |
| **eqnorm** | Eqnorm-MPtrj | `EOFError('Ran out of input')` | **A** weight-fetch (손상/빈 캐시) | 모듈 import 성공, torch.load가 빈/부분 파일 읽음 → 손상 캐시. **확정** |
| **matris** | MatRIS-10M-OAM | `EOFError('Ran out of input')` | **A** weight-fetch (손상/빈 캐시) | 동일 패턴. **확정** |
| nequip | NequIP-OAM-L | `Ninja is required to load C++ extensions` | **B** ninja | OpenEquivariance .pt2 빌드에 ninja 런타임 미적용 |
| allegro | Allegro-OAM-L | `Failed to initialize zip archive: file open failed` | **B2** .pt2 손상/부재 | ninja와 다른 root cause(.nequip.zip 열기 실패) |
| nequix | Nequix-MP-1 | `No module named 'nequix'` | **C** 모듈 미설치 | 선언됐으나 import 불가 |
| alphanet | AlphaNet-v1-OMA | `No module named 'alphanet'` | **C** 모듈 미설치 | |
| equflash | EquFlashV2 | `No module named 'GGNN'` | **C** 모듈 미설치 | 내부 의존 GGNN 미설치 |
| equiformer_v3 | EqV3-OMatMPtrjSalex | `No module named 'fairchem'` | **C** 모듈 미설치 | fairchem 선언됐으나 미적용 |
| mattersim | MatterSim-v1-5M | (energy+force, GPU PID 없음) | **D** cpu_fallback | device=cuda 미강제 |
| tace | TACE-OAM-L | `NVIDIA driver too old (found 12090)` | **E** 드라이버 skew | **비제어 외부값** — PyTorch CUDA 런타임 > 호스트 드라이버(12.9). CR-1 적용 대상. |
| fairchemv1 | eSEN-30M-OAM | (skipped, HF_TOKEN missing) | gated | license: facebook/OMAT24 |
| uma | UMA-m-1p1-OC20 | (skipped, HF_TOKEN missing) | gated | license: facebook/UMA |

## GJC_HANDOFF 4-bucket "추정" 대비 실측 교정

GJC_HANDOFF §3는 Bucket A에 `grace, deepmd, pet (+ likely nequix, alphanet, dpa4, tace, matris, eqnorm, equflash)`를 추정-편입하고 "weight-fetch가 ~7-9 env를 flip하는 단일 최대 레버"라고 단언했다. **실측 교정:**

- **진짜 버킷 A (weight-fetch)**: deepmd, grace, pet, dpa4, **eqnorm, matris** = **6개** (eqnorm/matris는 `EOFError`=손상/빈 weight 캐시로 A에 합류).
- **추정이 틀린 것**: `nequix`, `alphanet`, `equflash` = 버킷 C(모듈 미설치) — weight-fetch를 고쳐도 안 풀린다.
- **tace**: 버킷 E(드라이버 skew, 비제어) — handoff가 §5에 nvcc 12.8만 적고 드라이버-런타임 호환 매트릭스를 누락했다.
- 결론: weight-fetch는 13 중 6개를 커버하는 **최대 단일 레버가 맞지만**, handoff의 "7-9개"는 C/E를 잘못 편입한 과대평가. Option B는 버킷 A 대표(deepmd)부터 시작해 6개를 일반화하는 게 정합.

## carry-over 신뢰표시 (간판-현실 격차)

`models.json validation` 필드가 **cross-host(internal catbench L40S) carry-over값**임이 실측으로 드러났다:

- `eqnorm` = `validated_sm89` 표기인데 host(4060Ti sm89)에서 weight 없어 `EOFError` fail → arch-validity는 유효할 수 있으나 host에서 미검증.
- `fairchemv1`(eSEN), `uma` = `validated_sm89` 표기인데 gated로 skip.
- `matris`, `tace`, `allegro`, `alphanet` = 정직하게 `gpu_pending` 표기.

**조치 (이번 Phase 0):** `models.json _meta.field_guide.validation`을 2축(arch-validity / host-resource·driver-validity)으로 재정의하고, `_meta.gpu_arch.sm89`에 `RTX 4060 Ti`를 등재 + `host_note`로 16GB VRAM·driver skew 주의를 명시했다. 향후 done 판정은 **carried-over arch 라벨이 아니라 해당 host의 실측 tier-1/tier-2 결과**를 진실원으로 삼는다.

## Codex fix GPU 재검증 — Phase 2로 위임 (Option B 정합)

GJC_HANDOFF §4의 Codex fix(`registry.py` +58 version-name resolve, `fetch.py` +300 weights_source-driven fetch, `_worker.py` stdout→stderr, `envs/allegro.yml` +ninja, `envs/equiformer_v3.yml` +fairchem)는 **코드/recipe에 적용돼 있으나 GPU 미검증**이다. 13개 env를 Phase 0에서 전부 재빌드(env당 153~325s, 전부 삭제됨)하는 것은 승인된 플랜의 **Option B(대표 1개 end-to-end 먼저)를 위반**하고 GPU 시간을 낭비한다.

따라서 Codex fix의 실제 GPU 재검증은 **Phase 2에서 버킷 A 대표 deepmd(DPA-3.1-3M-FT) end-to-end 완주**로 수행하며, 그 완주가 곧 `fetch.py`·`registry.py` version-resolve·recipe fix를 한 번에 GPU 실증하는 골든 패스가 된다. 이는 플랜 §4 Phase 2 / Risks("Codex fix가 GPU에서 일부만 PASS → Option B 선검증")와 정합한다.

## Phase 0 게이트 분기 판정

플랜 §4 Phase 0 게이트: "E버킷이 tace 외로 다수 확장되거나 신규 root cause 발견 시 Option B 대표 선정 재평가". **실측 결과: E버킷은 tace 1개뿐**, 신규 root cause 없음(eqnorm/matris는 기존 A로 흡수). → **게이트 통과. Option B 대표 = deepmd(버킷 A) 유지.**

## GPU 재검증 실측 (Phase 0, 2026-06-30)

Codex fix(registry/fetch/recipe) 적용 상태에서 대표 PASS 모델 mace를 **재빌드+재검증**해 회귀 없음을 GPU로 실증:

| env | install | verify | energy (eV) | tier | 비고 |
|---|---|---|---|---|---|
| mace (MACE-MPA-0) | rc=0 (~270s) | rc=0 | -16.391451 | tier-2 PASS | Codex fix 적용 후 회귀 없음. forces 산출(소형 구조라 max\|force\|=0). **GPU PID 356480 nvidia-smi compute-apps 직접 관측** (`.sweep/phase0/gpu_sample.log`) → tier-2 GPU 실사용 입증. |

- 빌드 산출물: `envs/mace/` (6.8G, `.omm_ready` 센티넬 존재) — Phase 4 플러그인 `/setup` tier-1 증명에 재사용 보존.
- 로그: `.sweep/phase0/mace.{install,verify,runner}.log`.
- 결론: Codex fix는 기존 PASS 모델을 깨지 않음(회귀 0). 나머지 fix(fetch.py weights_source-driven, registry version-resolve)의 fail→PASS flip 실증은 Phase 2 버킷 A 대표 deepmd end-to-end에서 수행(Option B 정합).
