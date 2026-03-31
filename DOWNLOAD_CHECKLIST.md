# Heart Systemic Multimodal — 데이터셋 다운로드 체크리스트

> Codex에게 시킬 때 이 파일을 참조하세요
> 저장 위치: `heart_systemic_multimodal_project/data/raw/`

---

## 1단계: 핵심 (반드시 필요)

### Heart Single-Cell / Single-Nucleus

| # | ID | 질환 | 예상 크기 | 다운로드 명령 |
|---|---|---|---|---|
| 1 | **GSE183852** | Heart Failure | ~2-5 GB | `wget -P data/raw/ "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE183852&format=file"` |
| 2 | **GSE181764** | HCM | ~1-3 GB | `wget -P data/raw/ "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE181764&format=file"` |
| 3 | **GSE109816** | Normal Heart Reference | ~500 MB | `wget -P data/raw/ "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE109816&format=file"` |

### Plasma EV / Proteomics

| # | ID | 설명 | 예상 크기 | 다운로드 방법 |
|---|---|---|---|---|
| 4 | **PXD021371** | Plasma small EV proteome | ~200 MB | PRIDE API → processed results만 |
| 5 | **PXD059929** | ACS plasma EV | ~200 MB | PRIDE API → processed results만 |
| 6 | **PXD060680** | HCM plasma proteomics | ~200 MB | PRIDE API → processed results만 |

**PRIDE 다운로드 명령:**
```bash
# Processed files only (raw .mzML 제외)
python main.py download --dataset PXD021371
python main.py download --dataset PXD059929
python main.py download --dataset PXD060680
```

---

## 2단계: 중요 (공간 분석 + 뇌 참조)

| # | ID | 설명 | 예상 크기 | 다운로드 명령 |
|---|---|---|---|---|
| 7 | **GSE135805** | Cardiomyopathy spatial | ~1-2 GB | `wget -P data/raw/ "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE135805&format=file"` |
| 8 | **GSE290577** | Transplant rejection spatial | ~1-2 GB | `wget -P data/raw/ "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE290577&format=file"` |
| 9 | **E-MTAB-15659** | Brain organoid multiome | ~2-4 GB | BioStudies REST API |

---

## 3단계: 선택 (추가 분석용)

| # | ID | 설명 | 상태 |
|---|---|---|---|
| 10 | **GSE155514** | Cardiac hypertrophy (mouse) | GEO에서 수동 다운로드 |
| 11 | **CELLxGENE failing heart** | 대규모 심부전 아틀라스 | `cellxgene-census` API 또는 웹 다운로드 |
| 12 | Visium heart atlas | Placeholder | 데이터 공개 시 추가 |
| 13 | Liver/Muscle/Adipose ref | Placeholder | Human Cell Atlas에서 선택 |

---

## Codex에게 시킬 전체 명령

```bash
cd ~/heart_systemic_multimodal_project

# 1. 환경 설치
pip install -e .

# 2. 전체 다운로드 (자동)
python main.py download

# 3. 또는 단계별
python main.py download --dataset GSE183852
python main.py download --dataset GSE181764
python main.py download --dataset GSE109816
python main.py download --dataset PXD021371
python main.py download --dataset PXD059929
python main.py download --dataset PXD060680

# 4. 전처리
python main.py preprocess

# 5. 전체 파이프라인
python main.py run-all
```

---

## 수동 다운로드 필요한 경우

GEO 데이터가 자동 다운로드 실패 시:

1. https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE183852 접속
2. "Supplementary file" 섹션에서 h5ad 또는 matrix 파일 다운로드
3. `data/raw/GSE183852/` 폴더에 배치

PRIDE 데이터가 자동 다운로드 실패 시:

1. https://www.ebi.ac.uk/pride/archive/projects/PXD021371 접속
2. "Files" 탭에서 processed results (CSV/TSV) 다운로드
3. `data/raw/PXD021371/` 폴더에 배치

CELLxGENE:

1. https://cellxgene.cziscience.com/collections/283d65eb-dd53-496d-adb7-7570c7caa443 접속
2. "Download" 버튼으로 h5ad 파일 다운로드
3. `data/raw/cellxgene/` 폴더에 배치

---

## 예상 총 용량

- 1단계 (핵심): ~4-9 GB
- 2단계 (중요): ~4-8 GB
- 3단계 (선택): ~5-10 GB
- **총합: ~13-27 GB**

Mac mini 48GB RAM이면 모든 데이터셋 처리 가능. 24GB면 downsample 옵션 사용:
```bash
python main.py preprocess --downsample 50000
```
