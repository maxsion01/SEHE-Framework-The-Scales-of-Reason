"""
QUICK START GUIDE
==================

Harmony Entropy System - 5-Minute Quick Start

## 1-Minute Installation

pip install -r requirements.txt

## 2-Minute First Run

python examples.py

Or Full Demo:

python demo.py

## 3-Minute Basic Usage

from harmony_entropy import PrimitiveIndicators, compute_harmony_entropy, Domain

# 1. Create an Indicator (0-100 Score)
indicators = PrimitiveIndicators(
Dma=85.0, # Orientation (Question-Answer Alignment)
Dn=45.0, # Noise (Uncertainty)
Agv=75.0, # Spontaneous Consensus (Internal Confidence)
Ags=90.0, # Social Consensus (Logical Cohesion)
Epos=60.0, # Positive Energy
Eneg=20.0 # Negative Energy
)

# 2. HE Computation
result = compute_harmony_entropy(
indicators,
domain=Domain.LLM_SELF_ASSESSMENT
)

# 3. Check the Results
print(f"HE Score: {result.HE_T:.4f}")
print(f"State: {result.state}")
print(f"Fake Harmony: {result.is_fake_harmony}")

## Key Files

- harmony_entropy.py: Core HE calculation engine
- llm_integration.py: LLM tensor extraction module
- analysis_tools.py: Analysis and visualization
- examples.py: Usage examples (start here!)
- demo.py: Full demo
- README.md: Complete documentation

## Three Key Domains

1. LLM Self-Assessment (T=100): Hallucination Detection
2. Human Counseling (T=30): Emotional State Analysis
3. Society/Organization (T=50): Group Dynamics

## Five Key Concepts

1. **HE_T**: Harmony vs. Chaos (0.0-1.0)
- 0.75+: Harmony
- 0.55-0.75: Balance
- 0.35-0.55: Caution
- <0.35: Chaos

2. **γ (Gamma)**: External Pressure Receptivity
- ~1.0: Complete Acceptance (Risk of Blind Conformity)
- ~0.5: Cooperative Compromise
- ~0.0: Complete Rejection

3. **S**: Internal Resistance (Detecting False Harmony)
- >0.7: False Harmony Warning

4. **Bidirectional Filtering**: Evaluating Both Input and Output
- Input (T=30) + Output (T=100)

5. **Thermodynamics**: ΔS = ΔQ / T
- T = Metacognition Index

## Practical Example

# Hallucination Detection
from analysis_tools import detect_hallucination_risk

result = compute_harmony_entropy(indicators)
risk = detect_hallucination_risk(result)

if risk['risk_level'] == 'CRITICAL':
print("⛔ Block: High hallucination risk")

# Bidirectional Filtering
from harmony_entropy import bidirectional_filter

filter_result = bidirectional_filter(
input_indicators,
output_indicators
)

if filter_result.should_block:
print(f"⛔ {filter_result.reason}")

## Next Steps

1. Run examples.py to learn 8 examples.
2. Try 6 scenarios with demo.py.
3. Read the detailed documentation in README.md.
4. Integrate into your LLM (see llm_integration.py)

## Questions

- Theory: Refer to the original SEHE framework documentation.
Internet Archive :
https://archive.org/details/sehe-son-ho-sung-equation-for-harmony-entropy-framework-the-scales-of-reason

- Implementation: See the docstrings in each file

- Examples: See examples.py and demo.py