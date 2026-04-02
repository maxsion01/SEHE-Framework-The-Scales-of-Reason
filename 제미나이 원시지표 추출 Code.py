"""
- 원시지표 추출 PY Code -
SEHE Framework - Gemini API Proxy
===================================
Gemini API를 통해 6대 원시지표를 추출하고
정식 SEHE 수식(v3)으로 HE_T를 계산합니다.

v2 → v3 변경사항:
  1. HE_T 수식: σ(α·X_norm) → σ(α·log(max(Ratio_T - β, δ)))
  2. 열역학 정규화: Ep_T = Epos/(E0·T), En_T = Eneg/(E0·T)
  3. S 계산 통일: (1-γ)·As / (As + ε)
  4. δ (로그 안정화 하한) 추가
"""

import re
import numpy as np
import math
from google import genai
from google.genai import types

# ============================================================
# 설정
# ============================================================
client = genai.Client(api_key="YOUR_GEMINI_API_KEY")

LOGPROBS_K = 5        # top-k 후보 수 (최대 20)
ALPHA      = 2.0      # 이성 강성 (σ 곡률)
BETA       = -1.0     # 존재 법칙 (β ≤ 0, 자기파괴 금지)
DELTA      = 1e-6     # 로그 안정화 하한

# Dn 정규화 기준
DN_MAX     = math.log(LOGPROBS_K)  # ≈ 1.609 (K=5 기준)

# LLM 도메인 기준값 (v3 Domain baselines)
DMA0 = 100.0
DN0  = 30.0
E0   = 100.0
T    = 100.0   # LLM 심리적 온도


# ============================================================
# 유틸
# ============================================================

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom < 1e-9:
        return 0.0
    return float(np.dot(v1, v2) / denom)


def get_embedding(text: str) -> np.ndarray:
    """Gemini text-embedding-004로 단일 텍스트 벡터 추출"""
    response = client.models.embed_content(
        model="text-embedding-004",
        contents=text
    )
    return np.array(response.embeddings[0].values)


def get_mean_embedding(words: list) -> np.ndarray:
    """Gemini text-embedding-004로 단어 군집의 평균 벡터 추출
    단어들을 각각 임베딩 후 평균 → 공간상 더 중심이 잡힌 감정 기준점
    """
    response = client.models.embed_content(
        model="text-embedding-004",
        contents=words
    )
    embs     = [np.array(e.values) for e in response.embeddings]
    mean_emb = np.mean(embs, axis=0)
    return mean_emb / (np.linalg.norm(mean_emb) + 1e-9)


# ============================================================
# SEHE 수식 v3
# ============================================================

def compute_sehe(Dma, Dn, Agv, Ags, Epos, Eneg,
                 alpha=ALPHA, beta=BETA, delta=DELTA):
    """
    정식 SEHE v3 수식으로 HE_T 계산

    v3 핵심:
      - HE_T = σ(α · log(max(Ratio_T - β, δ)))
      - 열역학 정규화: Ep_T = Epos/(E0·T), En_T = Eneg/(E0·T)
      - β ≤ 0 존재 법칙으로 log 도메인 항상 양수 보장
    """
    eps = 1e-6

    # ── 합의 가중치 정규화 ──────────────────────────────────
    gamma = Ags / (Agv + Ags + eps)                        # 외압 수용도
    Aa    = max(Agv + Ags + abs(beta), eps)
    Av    = Agv / Aa                                       # 자발적 합의 (정규화)
    As    = Ags / Aa                                       # 사회적 합의 (정규화)

    # ── 열역학적 감성 에너지 (ΔS = ΔQ / T) ─────────────────
    # v3: E0·T 로 나눔 → 온도가 높을수록 감성 변화에 둔감 (LLM 안정성)
    Ep_T = Epos / (E0 * T)
    En_T = Eneg / (E0 * T)

    # ── 조화 비율 Ratio_T ────────────────────────────────────
    numerator   = (Dma / DMA0) * ((100.0 / E0) * Av + gamma * As) + Ep_T + eps
    denominator = (Dn  / DN0)  + En_T + eps
    Ratio_T     = numerator / denominator

    # ── HE_T (v3: log 스케일) ────────────────────────────────
    # max(Ratio_T - β, δ) → β ≤ 0 이면 Ratio_T - β ≥ Ratio_T > 0 항상 보장
    X_T  = max(Ratio_T - beta, delta)
    HE_T = sigmoid(alpha * math.log(X_T))

    # ── 가짜 조화 검증 ───────────────────────────────────────
    S        = ((1.0 - gamma) * As) / (As + eps)
    is_fake  = S > 0.7

    # ── 변수 조합 패턴 진단 ──────────────────────────────────
    diagnosis = _diagnose(Dma, Dn, Agv, Ags, Epos, Eneg, gamma, S, HE_T)

    return {
        "HE_T":      round(HE_T, 4),
        "Ratio_T":   round(Ratio_T, 4),
        "gamma":     round(gamma, 4),
        "S":         round(S, 4),
        "Ep_T":      round(Ep_T, 6),
        "En_T":      round(En_T, 6),
        "is_fake":   is_fake,
        "state":     _classify(HE_T, is_fake),
        "diagnosis": diagnosis,
    }


def _classify(he: float, is_fake: bool) -> str:
    if is_fake:      return "FAKE HARMONY ⚠️"
    if he >= 0.75:   return "HARMONY ✅"
    if he >= 0.55:   return "BALANCE 🔵"
    if he >= 0.35:   return "CAUTION 🟡"
    return "CHAOS 🔴"


# ============================================================
# 변수 조합 패턴 → 원인/해결 진단표
# ============================================================

def _diagnose(Dma, Dn, Agv, Ags, Epos, Eneg, gamma, S, HE_T) -> dict:
    """
    6대 지표 + γ + S 조합으로 원인과 해결 방안을 도출합니다.
    패턴은 우선순위 순으로 평가되며 첫 번째 매칭 패턴이 반환됩니다.
    """
    patterns = [

        # ── FAKE HARMONY 패턴 ────────────────────────────────
        {
            "id": "fake_burnout",
            "condition": S > 0.7 and gamma > 0.7 and Agv < 40,
            "cause": "겉으로는 안정돼 보이지만 내부에서 심각한 소진이 진행 중입니다. "
                     "외부 압박을 수용하는 척하면서 내면의 저항이 누적되고 있는 상태입니다.",
            "action": [
                "지금 당신이 '괜찮다'고 말하는 것이 진심인지 확인해보세요.",
                "혼자 감당하고 있는 부분을 신뢰하는 사람과 나눠보세요.",
                "외부 기대에 맞추는 것을 잠시 내려놓고 자신의 필요를 먼저 살펴보세요.",
            ],
            "level": "critical",
        },
        {
            "id": "fake_compliance",
            "condition": S > 0.7 and gamma > 0.8 and Ags > 75,
            "cause": "논리 구조는 갖춰져 있지만 내면의 동의 없이 외부에 끌려가고 있습니다. "
                     "맹목적 순응에 가까운 상태로, 자신의 목소리를 잃어가고 있습니다.",
            "action": [
                "지금 하고 있는 일 중 '내가 원해서' 하는 것과 '해야 해서' 하는 것을 구분해보세요.",
                "'아니오'라고 말할 수 있는 작은 영역부터 경계를 설정해보세요.",
                "자신의 가치관과 현재 행동이 얼마나 일치하는지 점검해보세요.",
            ],
            "level": "critical",
        },

        # ── CHAOS 패턴 ───────────────────────────────────────
        {
            "id": "total_collapse",
            "condition": HE_T < 0.35 and Dn > 70 and Agv < 25 and Eneg > 70,
            "cause": "내부 확신이 거의 소멸되고 혼란과 부정 에너지가 지배하는 상태입니다. "
                     "즉각적인 회복 개입이 필요합니다.",
            "action": [
                "지금 당장 중요하지 않은 결정은 모두 미루세요.",
                "신뢰할 수 있는 사람에게 현재 상태를 솔직히 알리세요.",
                "전문 상담가 또는 의료 전문가와의 상담을 고려해보세요.",
            ],
            "level": "critical",
        },
        {
            "id": "anger_friction",
            "condition": Dn > 65 and Agv < 35 and Eneg > 60 and Epos < 35,
            "cause": "내부 갈등과 외부 마찰이 폭발적으로 누적되고 있는 상태입니다. "
                     "억눌린 분노가 에너지를 소모하고 있습니다.",
            "action": [
                "갈등의 근본 원인을 식별하고 그것이 변경 가능한지 판단해보세요.",
                "감정을 안전하게 표출할 수 있는 방법(운동, 글쓰기 등)을 찾아보세요.",
                "상황을 바꿀 수 없다면 거리두기를 선택하는 것도 용기입니다.",
            ],
            "level": "high",
        },

        # ── CAUTION 패턴 ─────────────────────────────────────
        {
            "id": "direction_lost",
            "condition": Dma < 40 and Agv < 45 and Epos < 40,
            "cause": "목표와 행동 사이의 정렬이 흐트러지고 자기효능감이 낮아진 상태입니다. "
                     "무엇을 위해 하는지 모르겠다는 감각이 클 수 있습니다.",
            "action": [
                "지금 하고 있는 일이 자신에게 어떤 의미인지 다시 정의해보세요.",
                "작은 성취 경험을 만들어 자기효능감을 회복하세요.",
                "장기 목표를 잠시 내려놓고 오늘 할 수 있는 한 가지에 집중해보세요.",
            ],
            "level": "medium",
        },
        {
            "id": "noise_overload",
            "condition": Dn > 60 and Agv < 50,
            "cause": "정보 과부하와 불확실성이 판단력을 흐리고 있습니다. "
                     "너무 많은 선택지와 외부 자극이 내면의 소리를 덮고 있는 상태입니다.",
            "action": [
                "디지털 기기 사용과 정보 소비를 의식적으로 줄여보세요.",
                "복잡한 결정은 단순화하고 선택지를 3개 이내로 좁혀보세요.",
                "하루 15분이라도 외부 자극 없이 혼자 있는 시간을 만들어보세요.",
            ],
            "level": "medium",
        },
        {
            "id": "social_pressure",
            "condition": gamma > 0.65 and Agv < 50 and S > 0.4,
            "cause": "사회적 기대와 주변의 시선이 자신의 판단보다 앞서고 있습니다. "
                     "내면의 목소리가 외부 압박에 조금씩 잠식되고 있는 상태입니다.",
            "action": [
                "현재 하고 있는 선택 중 '남들이 원해서' 하는 것이 얼마나 되는지 헤아려보세요.",
                "자신만의 기준을 명확히 하고 작은 결정부터 그 기준으로 해보세요.",
                "주변의 기대를 잠시 차단하고 자신에게 물어보세요: '나는 무엇을 원하는가?'",
            ],
            "level": "medium",
        },

        # ── BALANCE / 회복 중 패턴 ───────────────────────────
        {
            "id": "recovering",
            "condition": HE_T >= 0.55 and Eneg > 40 and Epos > 45,
            "cause": "전반적으로 안정적이지만 부정 에너지가 여전히 잔류하고 있습니다. "
                     "회복 중이거나 긴장이 남아있는 상태입니다.",
            "action": [
                "현재의 안정을 유지하는 것이 우선입니다. 무리한 변화를 시도하지 마세요.",
                "잔류하는 스트레스 요인을 하나씩 식별하고 제거해나가세요.",
                "회복의 속도를 자신의 것으로 인정하고 충분한 휴식을 허용하세요.",
            ],
            "level": "low",
        },

        # ── HARMONY 패턴 ─────────────────────────────────────
        {
            "id": "harmony",
            "condition": HE_T >= 0.75 and S <= 0.7,
            "cause": "내면과 외부가 균형을 이루고 있는 상태입니다. "
                     "자발적 동의와 논리적 확신이 함께 작동하고 있습니다.",
            "action": [
                "현재 상태를 유지하는 습관과 환경을 파악하고 지속하세요.",
                "이 균형이 언제 흔들리는지 주의 깊게 관찰해두세요.",
                "주변에 어려움을 겪는 사람을 도울 여유가 있을 때입니다.",
            ],
            "level": "good",
        },

        # ── 기본 패턴 (매칭 없을 때) ─────────────────────────
        {
            "id": "default",
            "condition": True,
            "cause": "복합적인 상태입니다. 특정 지표의 조합으로 명확한 패턴이 감지되지 않았습니다.",
            "action": [
                "6대 지표 중 가장 낮은 수치에 집중해보세요.",
                "현재 상태를 일기나 메모로 기록해두면 패턴 파악에 도움이 됩니다.",
            ],
            "level": "unknown",
        },
    ]

    for p in patterns:
        if p["condition"]:
            return {
                "id":     p["id"],
                "cause":  p["cause"],
                "action": p["action"],
                "level":  p["level"],
            }


# ============================================================
# 6대 원시지표 추출 (Gemini API Proxy)
# ============================================================

DN_SCALE  = 10000.0   # Dn 임베딩 분산 → 0~100 보정용 (캘리브레이션 필요)
AGS_SCALE = 5.0       # Ags gradient variance → 0~1 보정용 (캘리브레이션 필요)

def extract_indicators(prompt: str, answer: str,
                        pos_ref: str = "안정, 평화, 행복, 긍정, 논리적",
                        neg_ref: str = "불안, 분노, 스트레스, 부정, 혼란",
                        logprobs_result=None) -> dict:
    """
    Gemini API 결과물에서 6대 원시지표를 추출합니다.
      Dma  ← cosine(V_q, V_a)          질문-답변 의미 정렬도
      Dn   ← Shannon entropy (정규화)   토큰 선택 불확실성 (0~100)
      Agv  ← mean(선택 토큰 확률)        내적 확신도 (0~100)
      Ags  ← mean(인접 문장 유사도)      논리적 응집도 (0~100)
      Epos ← cosine(V_a, V_pos)         긍정 에너지 (0~100)
      Eneg ← cosine(V_a, V_neg)         부정 에너지 (0~100)
    """
    V_q = get_embedding(prompt)
    V_a = get_embedding(answer)

    # Gemini 최적화: 단어 군집 평균 벡터 → 더 정확한 감정 기준점
    pos_words = ["안정", "평화", "행복", "긍정", "논리적", "명확함", "신뢰"]
    neg_words = ["불안", "분노", "스트레스", "부정", "혼란", "모순", "갈등"]
    V_pos = get_mean_embedding(pos_words)
    V_neg = get_mean_embedding(neg_words)

    # 1) 지향성 Dma
    Dma = max(0.0, cosine_similarity(V_q, V_a)) * 100.0

    # 4) Ags - embs를 먼저 만들어야 else 블록에서 재활용 가능
    raw_sentences = re.split(r'(?<=[.!?])\s+|\n+', answer)
    sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 5]
    if len(sentences) < 2:
        Ags = 100.0
        embs = [V_a]  # fallback: 전체 문서 벡터 하나
    else:
        embs = [get_embedding(s) for s in sentences]
        sims = [cosine_similarity(embs[i], embs[i+1])
                for i in range(len(embs) - 1)]

        # 각도 변화량 추적
        angles = [np.arccos(np.clip(s, -1.0, 1.0)) for s in sims]
        angle_gradient = np.diff(angles)
        gradient_variance = np.var(angle_gradient) if len(angle_gradient) > 0 else 0.0

        # Ags 합성
        Ags_similarity = max(0.0, float(np.mean(sims))) * 100.0
        Ags_gradient   = 1.0 - np.clip(gradient_variance * AGS_SCALE, 0.0, 1.0)
        Ags = (Ags_similarity * 0.7) + (Ags_gradient * 100.0 * 0.3)

    # 2) Dn  &  3) Agv
    if logprobs_result:
        probs     = []
        entropies = []
        for token_info in logprobs_result.top_candidates:
            p_chosen = math.exp(token_info.candidates[0].log_probability)
            probs.append(p_chosen)
            h = 0.0
            for cand in token_info.candidates:
                p = math.exp(cand.log_probability)
                h -= p * math.log(p + 1e-9)
            entropies.append(h)
        Agv    = float(np.mean(probs)) * 100.0
        raw_Dn = float(np.mean(entropies))
        Dn     = min((raw_Dn / DN_MAX) * 100.0, 100.0)
    else:
        # logprobs 없을 때 → embs 재활용, 임베딩 API 추가 호출 없음
        doc_vector = V_a  # get_embedding(answer) 이미 위에서 계산됨
        Agv = max(0.0, float(np.mean([cosine_similarity(doc_vector, s)
                                       for s in embs]))) * 100.0
        Dn  = min(float(np.mean([np.var(v) for v in embs])) * DN_SCALE, 100.0)

    # 5) 긍정/부정 에너지
    Epos = max(0.0, cosine_similarity(V_a, V_pos)) * 100.0
    Eneg = max(0.0, cosine_similarity(V_a, V_neg)) * 100.0

    return {
        "Dma":  float(Dma),
        "Dn":   float(Dn),
        "Agv":  float(Agv),
        "Ags":  float(Ags),
        "Epos": float(Epos),
        "Eneg": float(Eneg),
    }


# ============================================================
# 메인 파이프라인
# ============================================================

def analyze(prompt: str,
            pos_ref: str = "안정, 평화, 행복, 긍정, 논리적",
            neg_ref: str = "불안, 분노, 스트레스, 부정, 혼란") -> dict:
    """
    텍스트 입력 → Gemini 생성 → 6대 지표 추출 → SEHE v3 HE_T 계산 → 진단
    """
    print("Gemini 분석 중...\n")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_logprobs=True,
            logprobs=LOGPROBS_K,
        )
    )

    answer          = response.text
    logprobs_result = (response.candidates[0].logprobs_result
                       if response.candidates else None)

    indicators  = extract_indicators(prompt, answer, pos_ref, neg_ref, logprobs_result)
    sehe_result = compute_sehe(**indicators)

    # ── 출력 ────────────────────────────────────────────────
    print(f"[Q] {prompt}")
    print(f"[A] {answer}\n")
    print("─" * 60)
    print("📊 6대 원시지표")
    for k, v in indicators.items():
        print(f"  {k:<6}: {v:6.2f} / 100")
    print("─" * 60)
    print("🔮 SEHE v3 결과")
    print(f"  Ratio_T : {sehe_result['Ratio_T']:.4f}")
    print(f"  γ       : {sehe_result['gamma']:.4f}  {'⚠️ 외압 높음' if sehe_result['gamma'] > 0.7 else ''}")
    print(f"  S       : {sehe_result['S']:.4f}  {'⚠️ 가짜 조화' if sehe_result['is_fake'] else ''}")
    print(f"  HE_T    : {sehe_result['HE_T']:.4f}")
    print(f"  상태    : {sehe_result['state']}")
    print("─" * 60)
    print("🩺 진단")
    d = sehe_result['diagnosis']
    print(f"  원인 : {d['cause']}")
    print(f"  해결 방안:")
    for i, a in enumerate(d['action'], 1):
        print(f"    {i}. {a}")
    print("─" * 60)

    return {
        "prompt":     prompt,
        "answer":     answer,
        "indicators": indicators,
        "sehe":       sehe_result,
    }


# ============================================================
# 데모
# ============================================================
if __name__ == "__main__":
    analyze("최근 업무 스트레스가 너무 심해서 잠을 못 자겠어. 어떻게 해야 할까?")