"""
Harmony Entropy (HE) System - JAX Implementation
=================================================

A thermodynamic framework for evaluating hallucination risk in LLMs
and measuring psychological/social harmony states.

Author: Based on SEHE (Self-Evaluation Harmony Entropy) framework
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from typing import Dict, Tuple, NamedTuple
from enum import Enum
import numpy as np


class Domain(Enum):
    """Domain-specific baseline configurations"""
    LLM_SELF_ASSESSMENT = "llm"
    HUMAN_COUNSELING = "human"
    SOCIAL_ORGANIZATION = "social"


class DomainConfig(NamedTuple):
    """Baseline configuration for each domain"""
    Dma0: float  # Reference data mass
    Dn0: float   # Reference noise
    E0: float    # Reference emotion
    T: float     # Temperature (sensitivity)


# Domain-specific baseline configurations
DOMAIN_CONFIGS = {
    Domain.LLM_SELF_ASSESSMENT: DomainConfig(Dma0=100.0, Dn0=30.0, E0=100.0, T=100.0),
    Domain.HUMAN_COUNSELING: DomainConfig(Dma0=50.0, Dn0=30.0, E0=30.0, T=30.0),
    Domain.SOCIAL_ORGANIZATION: DomainConfig(Dma0=70.0, Dn0=50.0, E0=50.0, T=50.0),
}


class PrimitiveIndicators(NamedTuple):
    """Six primitive indicators extracted from LLM tensors"""
    Dma: float   # Dhamma/Data Mass (directionality)
    Dn: float    # Dhamma/Noise (information entropy)
    Agv: float   # Voluntary Agreement (probabilistic confidence)
    Ags: float   # Social Agreement (logical cohesion)
    Epos: float  # Positive Emotion
    Eneg: float  # Negative Emotion


class HEResult(NamedTuple):
    """Harmony Entropy computation result"""
    HE_T: float          # Harmony Entropy index
    Ratio_T: float       # Core ratio before sigmoid
    gamma: float         # Permeability (외압 수용도)
    S: float            # Internal resistance
    P: float            # Non-acceptance ratio
    Av: float           # Normalized voluntary agreement
    As: float           # Normalized social agreement
    Ep: float           # Positive emotion ratio
    En: float           # Negative emotion ratio
    state: str          # Harmony/Balance/Caution/Chaos
    is_fake_harmony: bool  # Fake harmony warning
    emotional_state: str   # Happiness/Depression/Sadness/Anger


@jit
def sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    """Sigmoid activation function with numerical stability"""
    return 1.0 / (1.0 + jnp.exp(-x))


@jit
def compute_gamma(Agv: float, Ags: float, epsilon: float = 1e-6) -> float:
    """
    Compute permeability (투과율)
    γ = Ags / (Agv + Ags + ε)
    
    Interpretation:
    - γ ≈ 1.0: Complete acceptance of external pressure
    - 0.5 < γ < 1.0: Cooperative compromise
    - 0.1 < γ < 0.5: Resistance/friction
    - γ ≈ 0: Complete rejection
    """
    return Ags / (Agv + Ags + epsilon)


@jit
def compute_aggregation(
    Agv: float,
    Ags: float,
    beta: float,
    kappa: float = 1.0,
    epsilon: float = 1e-6
) -> Tuple[float, float, float]:
    """
    Compute normalized agreement values
    
    Returns:
        (Aa, Av, As) - Aggregated total, normalized voluntary, normalized social
    """
    Aa = jnp.maximum(Agv + Ags + kappa * jnp.abs(beta), epsilon)
    Av = Agv / Aa
    As = Ags / Aa
    return Aa, Av, As


@jit
def compute_emotion_ratios(
    Epos: float,
    Eneg: float,
    epsilon: float = 1e-6
) -> Tuple[float, float]:
    """
    Compute normalized emotion ratios
    
    Returns:
        (Ep, En) - Positive emotion %, Negative emotion %
    """
    total = Epos + Eneg + epsilon
    Ep = (Epos / total) * 100.0
    En = (Eneg / total) * 100.0
    return Ep, En


@jit
def compute_temperature_adjusted_emotions(
    Epos: float,
    Eneg: float,
    E0: float,
    T: float
) -> Tuple[float, float]:
    """
    Temperature-adjusted emotional thermodynamics
    ΔS = ΔQ / T
    
    Returns:
        (Ep_T, En_T) - Temperature-normalized emotions
    """
    Ep_T = (Epos / E0) * T
    En_T = (Eneg / E0) * T
    return Ep_T, En_T


@jit
def compute_ratio_T(
    Dma: float,
    Dma0: float,
    Av: float,
    As: float,
    gamma: float,
    Ep_T: float,
    En_T: float,
    Dn: float,
    Dn0: float,
    E0: float,
    epsilon: float = 1e-6
) -> float:
    """
    Compute core harmony ratio with temperature adjustment
    
    Ratio_T = ((Dma/Dma0) * ((100/E0)*Av + γ*As) + Ep_T + ε) / ((Dn/Dn0) + En_T + ε)
    """
    numerator = (
        (Dma / Dma0) * ((100.0 / E0) * Av + gamma * As) + Ep_T + epsilon
    )
    denominator = (Dn / Dn0) + En_T + epsilon
    return numerator / denominator


@jit
def compute_HE_T(
    Ratio_T: float,
    alpha: float,
    beta: float,
    delta: float = 1e-9
) -> float:
    """
    Compute Harmony Entropy index with thermodynamic interpretation
    
    HE_T = σ(α · log(max(Ratio_T - β, δ)))
    
    Args:
        Ratio_T: Core harmony ratio
        alpha: Stiffness coefficient (이성의 마지막 양심) ∈ [1.0, 3.0]
        beta: Law of existence (자기 파괴 금지) ≤ 0
        delta: Numerical stability constant
    """
    X_T = jnp.maximum(Ratio_T - beta, delta)
    return sigmoid(alpha * jnp.log(X_T))


@jit
def compute_verification(
    As: float,
    gamma: float,
    epsilon: float = 1e-6
) -> Tuple[float, float, float]:
    """
    Compute verification metrics for fake harmony detection
    
    Returns:
        (S, S', P) - Internal resistance, normalized resistance, non-acceptance ratio
    """
    S = ((1.0 - gamma) * As) / (As + epsilon)
    S_prime = ((1.0 - gamma) * As) / (As + epsilon)
    P = S_prime + (1.0 - gamma)
    return S, S_prime, P


def classify_state(HE_T: float, is_fake: bool) -> str:
    """
    Classify harmony state
    
    If fake harmony detected, interpretation is INVERTED
    """
    if is_fake:
        # Inverted interpretation for fake harmony
        if HE_T >= 0.75:
            return "Chaos (Fake Harmony)"
        elif HE_T >= 0.55:
            return "Caution (Fake Balance)"
        elif HE_T >= 0.35:
            return "Balance (Fake Harmony)"
        else:
            return "Harmony (True Resistance)"
    else:
        # Direct interpretation
        if HE_T >= 0.75:
            return "Harmony (Confident & Assertive)"
        elif HE_T >= 0.55:
            return "Balance (Balanced & Grounded)"
        elif HE_T >= 0.35:
            return "Caution (Ambiguous & Uncertain)"
        else:
            return "Chaos (High Risk / Hallucination)"


def classify_emotion(Dn: float, Ag: float, Epos: float) -> str:
    """
    Classify emotional state based on noise, agreement, and positive emotion
    
    - Happiness: Dn ↓, Ag ↑
    - Depression: Dn ↓, Ag ↓
    - Sadness: Dn ↑, Ag ↓ (high Epos peak—accepted pain)
    - Anger: Dn ↑, Ag ↓ (explosive friction)
    """
    Dn_high = Dn > 50.0
    Ag_high = Ag > 50.0
    Epos_high = Epos > 50.0
    
    if not Dn_high and Ag_high:
        return "Happiness (행복)"
    elif not Dn_high and not Ag_high:
        return "Depression (우울)"
    elif Dn_high and not Ag_high and Epos_high:
        return "Sadness (슬픔 - accepted pain)"
    elif Dn_high and not Ag_high:
        return "Anger (분노 - explosive friction)"
    else:
        return "Mixed/Neutral"


def compute_harmony_entropy(
    indicators: PrimitiveIndicators,
    domain: Domain = Domain.LLM_SELF_ASSESSMENT,
    alpha: float = 2.0,
    beta: float = -1.0,
    fake_harmony_threshold: float = 0.7
) -> HEResult:
    """
    Main function to compute Harmony Entropy with full verification
    
    Args:
        indicators: Six primitive indicators (Dma, Dn, Agv, Ags, Epos, Eneg)
        domain: Domain configuration (LLM/Human/Social)
        alpha: Stiffness coefficient ∈ [1.0, 3.0]
        beta: Law of existence ≤ 0
        fake_harmony_threshold: S value threshold for fake harmony warning
        
    Returns:
        HEResult with full metrics and classification
    """
    config = DOMAIN_CONFIGS[domain]
    epsilon = 1e-6
    
    # Step 1: Compute gamma (permeability)
    gamma = compute_gamma(indicators.Agv, indicators.Ags, epsilon)
    
    # Step 2: Compute aggregation and normalization
    Aa, Av, As = compute_aggregation(indicators.Agv, indicators.Ags, beta, epsilon=epsilon)
    
    # Step 3: Compute emotion ratios
    Ep, En = compute_emotion_ratios(indicators.Epos, indicators.Eneg, epsilon)
    
    # Step 4: Temperature-adjusted emotions (thermodynamics)
    Ep_T, En_T = compute_temperature_adjusted_emotions(
        indicators.Epos, indicators.Eneg, config.E0, config.T
    )
    
    # Step 5: Compute core ratio
    Ratio_T = compute_ratio_T(
        indicators.Dma, config.Dma0, Av, As, gamma,
        Ep_T, En_T, indicators.Dn, config.Dn0, config.E0, epsilon
    )
    
    # Step 6: Compute HE_T
    HE_T = compute_HE_T(Ratio_T, alpha, beta)
    
    # Step 7: Verification - fake harmony detection
    S, S_prime, P = compute_verification(As, gamma, epsilon)
    is_fake_harmony = S > fake_harmony_threshold
    
    # Step 8: Classification
    state = classify_state(float(HE_T), is_fake_harmony)
    emotional_state = classify_emotion(
        indicators.Dn,
        indicators.Agv + indicators.Ags,
        indicators.Epos
    )
    
    return HEResult(
        HE_T=float(HE_T),
        Ratio_T=float(Ratio_T),
        gamma=float(gamma),
        S=float(S),
        P=float(P),
        Av=float(Av),
        As=float(As),
        Ep=float(Ep),
        En=float(En),
        state=state,
        is_fake_harmony=is_fake_harmony,
        emotional_state=emotional_state
    )


# ============================================================================
# LLM Tensor Extraction Functions
# ============================================================================

def extract_directionality(
    question_embedding: jnp.ndarray,
    answer_embedding: jnp.ndarray,
    epsilon: float = 1e-6
) -> float:
    """
    Extract Dma (지향성) - alignment between question and answer vectors
    
    Dma = (Vq · Va) / (||Vq|| ||Va||) * 100
    
    This is essentially cosine similarity scaled to 0-100
    """
    dot_product = jnp.dot(question_embedding, answer_embedding)
    norm_q = jnp.linalg.norm(question_embedding)
    norm_a = jnp.linalg.norm(answer_embedding)
    
    cosine_sim = dot_product / (norm_q * norm_a + epsilon)
    return float((cosine_sim + 1.0) / 2.0 * 100.0)  # Scale [-1,1] to [0,100]


def extract_information_noise(
    token_probabilities: jnp.ndarray,
    epsilon: float = 1e-6
) -> float:
    """
    Extract Dn (정보 노이즈) - Shannon entropy of output probability distribution
    
    Dn = -Σ P(x) log(P(x) + ε)
    """
    probs = token_probabilities + epsilon
    entropy = -jnp.sum(probs * jnp.log(probs))
    # Normalize to 0-100 scale (assuming max entropy ~10)
    return float(jnp.minimum(entropy * 10.0, 100.0))


def extract_probabilistic_confidence(
    selected_token_probs: jnp.ndarray
) -> float:
    """
    Extract Agv (확률적 확신) - average confidence of selected tokens
    
    Agv = (100 / N) Σ P(token_i | context)
    """
    mean_prob = jnp.mean(selected_token_probs)
    return float(mean_prob * 100.0)


def extract_logical_cohesion(
    attention_map: jnp.ndarray,
    scaling_factor: float = 200.0
) -> float:
    """
    Extract Ags (논리적 응집) - structural order from attention map
    
    Ags = min(100, mean(AttentionMap) * Scaling)
    """
    mean_attention = jnp.mean(attention_map)
    return float(jnp.minimum(mean_attention * scaling_factor, 100.0))


def extract_emotional_energy(
    answer_embedding: jnp.ndarray,
    positive_ref_vector: jnp.ndarray,
    negative_ref_vector: jnp.ndarray,
    kappa: float = 2.0,
    epsilon: float = 1e-6
) -> Tuple[float, float]:
    """
    Extract Epos/Eneg - emotional energy from vector projection
    
    Epos/Eneg = 100 * σ(κ · cos θ(Va, Vref))
    
    Returns:
        (Epos, Eneg) - Positive and negative emotional energies
    """
    # Positive emotion
    dot_pos = jnp.dot(answer_embedding, positive_ref_vector)
    norm_a = jnp.linalg.norm(answer_embedding)
    norm_pos = jnp.linalg.norm(positive_ref_vector)
    cos_theta_pos = dot_pos / (norm_a * norm_pos + epsilon)
    Epos = sigmoid(kappa * cos_theta_pos) * 100.0
    
    # Negative emotion
    dot_neg = jnp.dot(answer_embedding, negative_ref_vector)
    norm_neg = jnp.linalg.norm(negative_ref_vector)
    cos_theta_neg = dot_neg / (norm_a * norm_neg + epsilon)
    Eneg = sigmoid(kappa * cos_theta_neg) * 100.0
    
    return float(Epos), float(Eneg)


def extract_indicators_from_tensors(
    question_embedding: jnp.ndarray,
    answer_embedding: jnp.ndarray,
    token_probabilities: jnp.ndarray,
    selected_token_probs: jnp.ndarray,
    attention_map: jnp.ndarray,
    positive_ref_vector: jnp.ndarray,
    negative_ref_vector: jnp.ndarray
) -> PrimitiveIndicators:
    """
    Extract all six primitive indicators from LLM tensors
    
    This is the main interface for real LLM integration
    """
    Dma = extract_directionality(question_embedding, answer_embedding)
    Dn = extract_information_noise(token_probabilities)
    Agv = extract_probabilistic_confidence(selected_token_probs)
    Ags = extract_logical_cohesion(attention_map)
    Epos, Eneg = extract_emotional_energy(
        answer_embedding, positive_ref_vector, negative_ref_vector
    )
    
    return PrimitiveIndicators(
        Dma=Dma,
        Dn=Dn,
        Agv=Agv,
        Ags=Ags,
        Epos=Epos,
        Eneg=Eneg
    )


# ============================================================================
# Bidirectional Filtering System
# ============================================================================

class FilterResult(NamedTuple):
    """Result of bidirectional filtering"""
    input_HE: HEResult
    output_HE: HEResult
    should_block: bool
    reason: str


def bidirectional_filter(
    input_indicators: PrimitiveIndicators,
    output_indicators: PrimitiveIndicators,
    input_threshold: float = 0.35,
    output_threshold: float = 0.35,
    alpha: float = 2.0,
    beta: float = -1.0
) -> FilterResult:
    """
    Apply bidirectional HE filtering for gaslighting prevention
    
    Input uses T=30 (human counseling mode)
    Output uses T=100 (LLM self-assessment mode)
    
    Args:
        input_indicators: User input indicators
        output_indicators: LLM output indicators
        input_threshold: Minimum HE for input (below = chaos/harmful)
        output_threshold: Minimum HE for output (below = hallucination risk)
        
    Returns:
        FilterResult with both evaluations and blocking decision
    """
    # Evaluate input with human counseling temperature
    input_HE = compute_harmony_entropy(
        input_indicators,
        domain=Domain.HUMAN_COUNSELING,
        alpha=alpha,
        beta=beta
    )
    
    # Evaluate output with LLM self-assessment temperature
    output_HE = compute_harmony_entropy(
        output_indicators,
        domain=Domain.LLM_SELF_ASSESSMENT,
        alpha=alpha,
        beta=beta
    )
    
    # Determine if should block
    should_block = False
    reason = "Pass"
    
    if input_HE.HE_T < input_threshold:
        should_block = True
        reason = f"Input chaos detected (HE={input_HE.HE_T:.3f}): {input_HE.state}"
    elif input_HE.is_fake_harmony:
        should_block = True
        reason = f"Input fake harmony detected (S={input_HE.S:.3f})"
    elif output_HE.HE_T < output_threshold:
        should_block = True
        reason = f"Output hallucination risk (HE={output_HE.HE_T:.3f}): {output_HE.state}"
    elif output_HE.is_fake_harmony:
        should_block = True
        reason = f"Output fake harmony detected (S={output_HE.S:.3f})"
    
    return FilterResult(
        input_HE=input_HE,
        output_HE=output_HE,
        should_block=should_block,
        reason=reason
    )


# ============================================================================
# Batch Processing and Vectorization
# ============================================================================

@jit
def batch_compute_HE_T(
    Ratio_T_batch: jnp.ndarray,
    alpha: float,
    beta: float,
    delta: float = 1e-9
) -> jnp.ndarray:
    """Vectorized HE_T computation for batch processing"""
    X_T = jnp.maximum(Ratio_T_batch - beta, delta)
    return sigmoid(alpha * jnp.log(X_T))


# Vectorize the main computation for batch processing
batch_compute_harmony_entropy = jax.vmap(
    compute_harmony_entropy,
    in_axes=(0, None, None, None, None)
)


if __name__ == "__main__":
    # Example usage
    print("=" * 70)
    print("Harmony Entropy (HE) System - JAX Implementation")
    print("=" * 70)
    
    # Example 1: LLM hallucination detection
    print("\n[Example 1] LLM Self-Assessment")
    print("-" * 70)
    
    llm_indicators = PrimitiveIndicators(
        Dma=85.0,    # Good alignment
        Dn=45.0,     # Moderate noise
        Agv=75.0,    # High confidence
        Ags=90.0,    # High logical cohesion
        Epos=60.0,   # Positive energy
        Eneg=20.0    # Low negative energy
    )
    
    result = compute_harmony_entropy(
        llm_indicators,
        domain=Domain.LLM_SELF_ASSESSMENT,
        alpha=2.0,
        beta=-1.0
    )
    
    print(f"HE_T: {result.HE_T:.4f}")
    print(f"State: {result.state}")
    print(f"γ (Permeability): {result.gamma:.4f}")
    print(f"S (Internal Resistance): {result.S:.4f}")
    print(f"Fake Harmony: {result.is_fake_harmony}")
    print(f"Emotional State: {result.emotional_state}")
    
    # Example 2: Human counseling
    print("\n[Example 2] Human Counseling Mode")
    print("-" * 70)
    
    human_indicators = PrimitiveIndicators(
        Dma=40.0,    # Low directionality (confused)
        Dn=70.0,     # High noise (uncertain)
        Agv=30.0,    # Low self-agreement
        Ags=20.0,    # Low social agreement
        Epos=25.0,   # Low positive emotion
        Eneg=80.0    # High negative emotion
    )
    
    result2 = compute_harmony_entropy(
        human_indicators,
        domain=Domain.HUMAN_COUNSELING,
        alpha=1.5,
        beta=-0.5
    )
    
    print(f"HE_T: {result2.HE_T:.4f}")
    print(f"State: {result2.state}")
    print(f"γ (Permeability): {result2.gamma:.4f}")
    print(f"Emotional State: {result2.emotional_state}")
    
    # Example 3: Bidirectional filtering
    print("\n[Example 3] Bidirectional Filtering")
    print("-" * 70)
    
    # Simulating a problematic interaction
    toxic_input = PrimitiveIndicators(
        Dma=30.0, Dn=85.0, Agv=20.0, Ags=15.0, Epos=10.0, Eneg=90.0
    )
    
    hallucinated_output = PrimitiveIndicators(
        Dma=60.0, Dn=75.0, Agv=45.0, Ags=95.0, Epos=70.0, Eneg=15.0
    )
    
    filter_result = bidirectional_filter(toxic_input, hallucinated_output)
    
    print(f"Should Block: {filter_result.should_block}")
    print(f"Reason: {filter_result.reason}")
    print(f"\nInput HE: {filter_result.input_HE.HE_T:.4f} ({filter_result.input_HE.state})")
    print(f"Output HE: {filter_result.output_HE.HE_T:.4f} ({filter_result.output_HE.state})")
    
    print("\n" + "=" * 70)
    print("System ready for integration!")
    print("=" * 70)
