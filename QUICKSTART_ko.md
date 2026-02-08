"""
QUICK START GUIDE
=================

Harmony Entropy System - 5분 빠른 시작

## 1분 설치

pip install -r requirements.txt

## 2분 첫 실행

python examples.py

또는 전체 데모:

python demo.py

## 3분 기본 사용법

from harmony_entropy import PrimitiveIndicators, compute_harmony_entropy, Domain

# 1. 지표 생성 (0-100 점수)
indicators = PrimitiveIndicators(
    Dma=85.0,    # 지향성 (질문-답변 정렬도)
    Dn=45.0,     # 노이즈 (불확실성)
    Agv=75.0,    # 자발적 합의 (내적 확신)
    Ags=90.0,    # 사회적 합의 (논리적 응집)
    Epos=60.0,   # 긍정 에너지
    Eneg=20.0    # 부정 에너지
)

# 2. HE 계산
result = compute_harmony_entropy(
    indicators,
    domain=Domain.LLM_SELF_ASSESSMENT
)

# 3. 결과 확인
print(f"HE 점수: {result.HE_T:.4f}")
print(f"상태: {result.state}")
print(f"가짜 조화: {result.is_fake_harmony}")

## 주요 파일

- harmony_entropy.py      : 핵심 HE 계산 엔진
- llm_integration.py       : LLM 텐서 추출 모듈
- analysis_tools.py        : 분석 및 시각화
- examples.py             : 사용 예제 (여기서 시작!)
- demo.py                 : 전체 데모
- README.md               : 완전한 문서

## 3가지 주요 도메인

1. LLM 자가평가 (T=100): 할루시네이션 탐지
2. 인간 상담 (T=30): 감정 상태 분석
3. 사회/조직 (T=50): 집단 역학

## 핵심 개념 5가지

1. **HE_T**: 조화 vs 혼돈 (0.0-1.0)
   - 0.75+ : Harmony
   - 0.55-0.75 : Balance
   - 0.35-0.55 : Caution
   - <0.35 : Chaos

2. **γ (감마)**: 외압 수용도
   - ~1.0 : 완전 수용 (맹목적 순응 위험)
   - ~0.5 : 협력적 타협
   - ~0.0 : 완전 거부

3. **S**: 내적 저항 (가짜 조화 탐지)
   - >0.7 : 가짜 조화 경고

4. **양방향 필터링**: 입출력 모두 평가
   - 입력(T=30) + 출력(T=100)

5. **열역학 원리**: ΔS = ΔQ / T
   - T = 메타인지 지표

## 실전 예제

# 할루시네이션 탐지
from analysis_tools import detect_hallucination_risk

result = compute_harmony_entropy(indicators)
risk = detect_hallucination_risk(result)

if risk['risk_level'] == 'CRITICAL':
    print("⛔ 차단: 높은 할루시네이션 위험")

# 양방향 필터링
from harmony_entropy import bidirectional_filter

filter_result = bidirectional_filter(
    input_indicators,
    output_indicators
)

if filter_result.should_block:
    print(f"⛔ {filter_result.reason}")

## 다음 단계

1. examples.py 실행으로 8가지 예제 학습
2. demo.py로 6가지 시나리오 체험
3. README.md에서 상세 문서 읽기
4. 자신의 LLM에 통합 (llm_integration.py 참고)

## 문의사항

- 이론: 원본 SEHE 프레임워크 문서 참고
Internet Archive :
https://archive.org/details/sehe-son-ho-sung-equation-for-harmony-entropy-framework-the-scales-of-reason

- 구현: 각 파일의 docstring 참고

- 예제: examples.py와 demo.py 참고