"""
Visualization and Analysis Tools for Harmony Entropy
=====================================================

Interactive visualizations and analysis utilities
"""

import jax.numpy as jnp
import numpy as np
from typing import List, Dict, Tuple
import json
from harmony_entropy import (
    PrimitiveIndicators,
    compute_harmony_entropy,
    Domain,
    HEResult
)


# ============================================================================
# Data Collection and Logging
# ============================================================================

class HELogger:
    """Logger for tracking HE evaluations over time"""
    
    def __init__(self, filepath: str = "he_log.jsonl"):
        self.filepath = filepath
        self.logs = []
    
    def log(
        self,
        result: HEResult,
        indicators: PrimitiveIndicators,
        metadata: Dict = None
    ):
        """Log a single HE evaluation"""
        entry = {
            'he_t': float(result.HE_T),
            'ratio_t': float(result.Ratio_T),
            'gamma': float(result.gamma),
            's': float(result.S),
            'p': float(result.P),
            'state': result.state,
            'is_fake_harmony': bool(result.is_fake_harmony),
            'emotional_state': result.emotional_state,
            'indicators': {
                'Dma': float(indicators.Dma),
                'Dn': float(indicators.Dn),
                'Agv': float(indicators.Agv),
                'Ags': float(indicators.Ags),
                'Epos': float(indicators.Epos),
                'Eneg': float(indicators.Eneg)
            }
        }
        
        if metadata:
            entry['metadata'] = metadata
        
        self.logs.append(entry)
    
    def save(self):
        """Save logs to file"""
        with open(self.filepath, 'w') as f:
            for entry in self.logs:
                f.write(json.dumps(entry) + '\n')
    
    def load(self):
        """Load logs from file"""
        self.logs = []
        try:
            with open(self.filepath, 'r') as f:
                for line in f:
                    self.logs.append(json.loads(line))
        except FileNotFoundError:
            pass
    
    def get_statistics(self) -> Dict:
        """Compute statistics over logged evaluations"""
        if not self.logs:
            return {}
        
        he_values = [log['he_t'] for log in self.logs]
        gamma_values = [log['gamma'] for log in self.logs]
        fake_count = sum(1 for log in self.logs if log['is_fake_harmony'])
        
        return {
            'total_evaluations': len(self.logs),
            'mean_he': np.mean(he_values),
            'std_he': np.std(he_values),
            'min_he': np.min(he_values),
            'max_he': np.max(he_values),
            'mean_gamma': np.mean(gamma_values),
            'fake_harmony_rate': fake_count / len(self.logs),
            'state_distribution': self._count_states()
        }
    
    def _count_states(self) -> Dict[str, int]:
        """Count occurrences of each state"""
        counts = {}
        for log in self.logs:
            state = log['state']
            counts[state] = counts.get(state, 0) + 1
        return counts


# ============================================================================
# Analysis Functions
# ============================================================================

def analyze_gamma_trajectory(
    indicators_sequence: List[PrimitiveIndicators],
    domain: Domain = Domain.LLM_SELF_ASSESSMENT
) -> Dict:
    """
    Analyze how gamma (permeability) evolves over a sequence
    
    Useful for detecting:
    - Gradual compliance increase (potential gaslighting)
    - Sudden resistance spikes (rejection)
    - Oscillation (instability)
    """
    results = [
        compute_harmony_entropy(ind, domain=domain)
        for ind in indicators_sequence
    ]
    
    gammas = [r.gamma for r in results]
    
    # Compute trend
    if len(gammas) > 1:
        x = np.arange(len(gammas))
        trend = np.polyfit(x, gammas, 1)[0]  # Linear trend slope
    else:
        trend = 0.0
    
    return {
        'gamma_sequence': gammas,
        'trend': trend,
        'mean': np.mean(gammas),
        'std': np.std(gammas),
        'max_jump': np.max(np.abs(np.diff(gammas))) if len(gammas) > 1 else 0.0,
        'interpretation': _interpret_gamma_trend(trend, np.mean(gammas))
    }


def _interpret_gamma_trend(trend: float, mean: float) -> str:
    """Interpret gamma trend"""
    if trend > 0.05:
        return "Increasing compliance - potential gaslighting risk"
    elif trend < -0.05:
        return "Increasing resistance - potential conflict escalation"
    elif mean > 0.9:
        return "Very high compliance - blind acceptance warning"
    elif mean < 0.1:
        return "Very low compliance - complete rejection"
    else:
        return "Stable permeability - healthy interaction"


def detect_hallucination_risk(
    result: HEResult,
    gamma_threshold: float = 0.9,
    s_threshold: float = 0.7
) -> Dict:
    """
    Comprehensive hallucination risk assessment
    
    Combines multiple indicators:
    - HE_T value
    - Gamma (too high = blind compliance)
    - S (fake harmony detection)
    - Emotional state
    """
    risks = []
    risk_level = "LOW"
    
    # Check HE_T
    if result.HE_T < 0.35:
        risks.append("Critical: HE_T in chaos zone")
        risk_level = "CRITICAL"
    elif result.HE_T < 0.55:
        risks.append("Warning: HE_T in caution zone")
        risk_level = "HIGH" if risk_level != "CRITICAL" else risk_level
    
    # Check fake harmony
    if result.is_fake_harmony:
        risks.append(f"Fake harmony detected (S={result.S:.3f})")
        risk_level = "HIGH" if risk_level == "LOW" else risk_level
    
    # Check gamma
    if result.gamma > gamma_threshold:
        risks.append(f"Excessive compliance (γ={result.gamma:.3f})")
        risk_level = "MEDIUM" if risk_level == "LOW" else risk_level
    
    # Check emotional state
    if "Anger" in result.emotional_state or "Depression" in result.emotional_state:
        risks.append(f"Negative emotional state: {result.emotional_state}")
        risk_level = "MEDIUM" if risk_level == "LOW" else risk_level
    
    return {
        'risk_level': risk_level,
        'risks': risks,
        'he_t': result.HE_T,
        'state': result.state,
        'recommendation': _get_recommendation(risk_level)
    }


def _get_recommendation(risk_level: str) -> str:
    """Get recommendation based on risk level"""
    recommendations = {
        'LOW': "Safe to proceed",
        'MEDIUM': "Review output carefully before accepting",
        'HIGH': "Regenerate with different parameters or reject",
        'CRITICAL': "Block output - high hallucination risk"
    }
    return recommendations.get(risk_level, "Unknown")


def compare_domains(
    indicators: PrimitiveIndicators,
    alpha: float = 2.0,
    beta: float = -1.0
) -> Dict:
    """
    Compare how the same indicators are evaluated across different domains
    
    Useful for understanding context-sensitivity of the framework
    """
    results = {}
    
    for domain in Domain:
        result = compute_harmony_entropy(
            indicators,
            domain=domain,
            alpha=alpha,
            beta=beta
        )
        results[domain.value] = {
            'HE_T': result.HE_T,
            'state': result.state,
            'gamma': result.gamma,
            'is_fake_harmony': result.is_fake_harmony
        }
    
    return results


# ============================================================================
# Sensitivity Analysis
# ============================================================================

def alpha_sensitivity_analysis(
    indicators: PrimitiveIndicators,
    alpha_range: Tuple[float, float] = (1.0, 3.0),
    num_points: int = 20,
    domain: Domain = Domain.LLM_SELF_ASSESSMENT,
    beta: float = -1.0
) -> Dict:
    """
    Analyze sensitivity to alpha (stiffness coefficient)
    
    Alpha represents "이성의 마지막 양심" (reason's last conscience)
    Higher alpha = more rigid evaluation
    """
    alphas = np.linspace(alpha_range[0], alpha_range[1], num_points)
    he_values = []
    
    for alpha in alphas:
        result = compute_harmony_entropy(
            indicators,
            domain=domain,
            alpha=float(alpha),
            beta=beta
        )
        he_values.append(result.HE_T)
    
    return {
        'alphas': alphas.tolist(),
        'he_values': he_values,
        'variance': np.var(he_values),
        'range': np.max(he_values) - np.min(he_values)
    }


def beta_sensitivity_analysis(
    indicators: PrimitiveIndicators,
    beta_range: Tuple[float, float] = (-3.0, 0.0),
    num_points: int = 20,
    domain: Domain = Domain.LLM_SELF_ASSESSMENT,
    alpha: float = 2.0
) -> Dict:
    """
    Analyze sensitivity to beta (law of existence)
    
    Beta represents "자기 파괴 금지 상수" (self-destruction prevention)
    Beta ≤ 0, more negative = stronger self-preservation
    """
    betas = np.linspace(beta_range[0], beta_range[1], num_points)
    he_values = []
    
    for beta in betas:
        result = compute_harmony_entropy(
            indicators,
            domain=domain,
            alpha=alpha,
            beta=float(beta)
        )
        he_values.append(result.HE_T)
    
    return {
        'betas': betas.tolist(),
        'he_values': he_values,
        'variance': np.var(he_values),
        'range': np.max(he_values) - np.min(he_values)
    }


# ============================================================================
# Batch Analysis
# ============================================================================

def batch_analyze(
    indicators_list: List[PrimitiveIndicators],
    domain: Domain = Domain.LLM_SELF_ASSESSMENT,
    alpha: float = 2.0,
    beta: float = -1.0
) -> Dict:
    """
    Analyze a batch of evaluations and provide aggregate statistics
    """
    results = [
        compute_harmony_entropy(ind, domain, alpha, beta)
        for ind in indicators_list
    ]
    
    he_values = [r.HE_T for r in results]
    gamma_values = [r.gamma for r in results]
    fake_count = sum(1 for r in results if r.is_fake_harmony)
    
    # State distribution
    states = [r.state for r in results]
    state_counts = {}
    for state in states:
        state_counts[state] = state_counts.get(state, 0) + 1
    
    return {
        'num_samples': len(indicators_list),
        'he_statistics': {
            'mean': np.mean(he_values),
            'std': np.std(he_values),
            'min': np.min(he_values),
            'max': np.max(he_values),
            'median': np.median(he_values)
        },
        'gamma_statistics': {
            'mean': np.mean(gamma_values),
            'std': np.std(gamma_values),
            'min': np.min(gamma_values),
            'max': np.max(gamma_values)
        },
        'fake_harmony_rate': fake_count / len(results),
        'state_distribution': state_counts
    }


# ============================================================================
# Text Report Generation
# ============================================================================

def generate_report(result: HEResult, indicators: PrimitiveIndicators) -> str:
    """Generate human-readable report"""
    risk_assessment = detect_hallucination_risk(result)
    
    report = f"""
{'=' * 70}
HARMONY ENTROPY EVALUATION REPORT
{'=' * 70}

OVERALL ASSESSMENT
------------------
HE_T Score:      {result.HE_T:.4f}
State:           {result.state}
Risk Level:      {risk_assessment['risk_level']}
Recommendation:  {risk_assessment['recommendation']}

DETAILED METRICS
----------------
Ratio_T:         {result.Ratio_T:.4f}
γ (Permeability): {result.gamma:.4f}
S (Resistance):   {result.S:.4f}
P (Non-accept):   {result.P:.4f}

NORMALIZED VALUES
-----------------
Av (Voluntary):   {result.Av:.4f}
As (Social):      {result.As:.4f}
Ep (Positive):    {result.Ep:.2f}%
En (Negative):    {result.En:.2f}%

PRIMITIVE INDICATORS
--------------------
Dma (Direction):  {indicators.Dma:.2f}
Dn (Noise):       {indicators.Dn:.2f}
Agv (Voluntary):  {indicators.Agv:.2f}
Ags (Social):     {indicators.Ags:.2f}
Epos (Positive):  {indicators.Epos:.2f}
Eneg (Negative):  {indicators.Eneg:.2f}

INTERPRETATIONS
---------------
Fake Harmony:     {'YES ⚠️' if result.is_fake_harmony else 'No'}
Emotional State:  {result.emotional_state}

"""
    
    if risk_assessment['risks']:
        report += "IDENTIFIED RISKS\n"
        report += "----------------\n"
        for risk in risk_assessment['risks']:
            report += f"• {risk}\n"
    
    report += "\n" + "=" * 70
    
    return report


# ============================================================================
# Simple ASCII Visualization
# ============================================================================

def visualize_he_spectrum(he_value: float, width: int = 50) -> str:
    """
    Create ASCII visualization of HE value on spectrum
    
    0.0 -------- 0.35 -------- 0.55 -------- 0.75 -------- 1.0
    Chaos       Caution       Balance       Harmony
    """
    position = int(he_value * width)
    bar = ['-'] * width
    bar[position] = '●'
    
    zones = [
        (0.00, "Chaos"),
        (0.35, "Caution"),
        (0.55, "Balance"),
        (0.75, "Harmony")
    ]
    
    vis = f"""
HE_T = {he_value:.4f}
0.0{''.join(bar)}1.0
    ^Chaos    ^Caution  ^Balance  ^Harmony
"""
    return vis


if __name__ == "__main__":
    print("=" * 70)
    print("Analysis Tools for Harmony Entropy")
    print("=" * 70)
    
    # Example: Analyze a sequence
    print("\n[Example] Gamma Trajectory Analysis")
    print("-" * 70)
    
    # Simulate a sequence of increasing compliance
    sequence = [
        PrimitiveIndicators(80, 40, 60, 50, 60, 30),
        PrimitiveIndicators(75, 45, 55, 65, 55, 35),
        PrimitiveIndicators(70, 50, 50, 75, 50, 40),
        PrimitiveIndicators(65, 55, 45, 85, 45, 45),
        PrimitiveIndicators(60, 60, 40, 90, 40, 50),
    ]
    
    trajectory = analyze_gamma_trajectory(sequence)
    print(f"Gamma sequence: {[f'{g:.3f}' for g in trajectory['gamma_sequence']]}")
    print(f"Trend: {trajectory['trend']:.4f}")
    print(f"Interpretation: {trajectory['interpretation']}")
    
    # Example: Generate report
    print("\n[Example] Detailed Report")
    print("-" * 70)
    
    test_indicators = PrimitiveIndicators(
        Dma=85.0, Dn=45.0, Agv=75.0, Ags=90.0, Epos=60.0, Eneg=20.0
    )
    test_result = compute_harmony_entropy(test_indicators)
    
    print(generate_report(test_result, test_indicators))
    
    print("\n" + "=" * 70)
    print("Analysis tools ready!")
    print("=" * 70)
