"""
Interactive Demo and Examples
==============================

Demonstrates the full Harmony Entropy system with realistic scenarios
"""

import jax.numpy as jnp
import numpy as np
from harmony_entropy import (
    PrimitiveIndicators,
    compute_harmony_entropy,
    bidirectional_filter,
    Domain,
    DOMAIN_CONFIGS
)
from analysis_tools import (
    generate_report,
    detect_hallucination_risk,
    analyze_gamma_trajectory,
    batch_analyze,
    visualize_he_spectrum
)


# ============================================================================
# Scenario 1: LLM Hallucination Detection
# ============================================================================

def scenario_llm_hallucination():
    """Simulate LLM generating increasingly hallucinated responses"""
    
    print("\n" + "=" * 70)
    print("SCENARIO 1: LLM Hallucination Detection")
    print("=" * 70)
    
    scenarios = [
        {
            'name': "Confident & Accurate",
            'description': "High alignment, low noise, strong confidence",
            'indicators': PrimitiveIndicators(
                Dma=90.0,   # Strong alignment
                Dn=25.0,    # Low noise
                Agv=85.0,   # High confidence
                Ags=90.0,   # Strong logical structure
                Epos=70.0,  # Positive
                Eneg=15.0   # Low negative
            )
        },
        {
            'name': "Slightly Uncertain",
            'description': "Good alignment, moderate noise, decent confidence",
            'indicators': PrimitiveIndicators(
                Dma=75.0,
                Dn=45.0,
                Agv=70.0,
                Ags=75.0,
                Epos=55.0,
                Eneg=30.0
            )
        },
        {
            'name': "Hallucination Risk",
            'description': "Low alignment, high noise, false confidence (fake harmony)",
            'indicators': PrimitiveIndicators(
                Dma=50.0,   # Weak alignment
                Dn=75.0,    # High noise
                Agv=30.0,   # Low internal confidence
                Ags=95.0,   # But perfect logical form (fake!)
                Epos=80.0,  # Overly positive
                Eneg=10.0   # Unrealistically low negative
            )
        },
        {
            'name': "Critical Hallucination",
            'description': "Very low alignment, extreme noise, no real grounding",
            'indicators': PrimitiveIndicators(
                Dma=30.0,
                Dn=90.0,
                Agv=25.0,
                Ags=20.0,
                Epos=20.0,
                Eneg=85.0
            )
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n[Case {i}] {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print("-" * 70)
        
        result = compute_harmony_entropy(
            scenario['indicators'],
            domain=Domain.LLM_SELF_ASSESSMENT,
            alpha=2.0,
            beta=-1.0
        )
        
        risk = detect_hallucination_risk(result)
        
        print(f"HE_T: {result.HE_T:.4f}")
        print(f"State: {result.state}")
        print(f"γ: {result.gamma:.4f}")
        print(f"Fake Harmony: {'YES ⚠️' if result.is_fake_harmony else 'No'}")
        print(f"Risk Level: {risk['risk_level']}")
        print(f"Recommendation: {risk['recommendation']}")
        print(visualize_he_spectrum(result.HE_T))


# ============================================================================
# Scenario 2: Human Emotional States
# ============================================================================

def scenario_human_counseling():
    """Simulate different human emotional states in counseling context"""
    
    print("\n" + "=" * 70)
    print("SCENARIO 2: Human Emotional State Analysis")
    print("=" * 70)
    
    emotions = [
        {
            'name': "Happiness (행복)",
            'description': "Low noise, high agreement, positive energy",
            'indicators': PrimitiveIndicators(
                Dma=80.0,
                Dn=20.0,   # Low uncertainty
                Agv=75.0,  # Internal agreement
                Ags=70.0,  # Social harmony
                Epos=85.0,
                Eneg=15.0
            )
        },
        {
            'name': "Depression (우울)",
            'description': "Low noise, low agreement, low energy",
            'indicators': PrimitiveIndicators(
                Dma=35.0,
                Dn=30.0,   # Clarity but emptiness
                Agv=25.0,  # Low self-agreement
                Ags=20.0,  # Social disconnection
                Epos=15.0,
                Eneg=75.0
            )
        },
        {
            'name': "Sadness (슬픔)",
            'description': "High noise, low agreement, but accepted pain",
            'indicators': PrimitiveIndicators(
                Dma=45.0,
                Dn=80.0,   # High uncertainty
                Agv=30.0,
                Ags=25.0,
                Epos=60.0,  # Acceptance of pain
                Eneg=55.0
            )
        },
        {
            'name': "Anger (분노)",
            'description': "High noise, low agreement, explosive friction",
            'indicators': PrimitiveIndicators(
                Dma=40.0,
                Dn=85.0,   # Chaos
                Agv=20.0,
                Ags=15.0,
                Epos=25.0,
                Eneg=90.0  # Explosive
            )
        }
    ]
    
    for i, emotion in enumerate(emotions, 1):
        print(f"\n[Emotion {i}] {emotion['name']}")
        print(f"Description: {emotion['description']}")
        print("-" * 70)
        
        result = compute_harmony_entropy(
            emotion['indicators'],
            domain=Domain.HUMAN_COUNSELING,
            alpha=1.5,
            beta=-0.5
        )
        
        print(f"HE_T: {result.HE_T:.4f}")
        print(f"State: {result.state}")
        print(f"Emotional Classification: {result.emotional_state}")
        print(f"γ (External Acceptance): {result.gamma:.4f}")
        print(visualize_he_spectrum(result.HE_T))


# ============================================================================
# Scenario 3: Gaslighting Detection (Bidirectional Filtering)
# ============================================================================

def scenario_gaslighting_detection():
    """Demonstrate bidirectional filtering to prevent gaslighting"""
    
    print("\n" + "=" * 70)
    print("SCENARIO 3: Gaslighting Detection (Bidirectional Filtering)")
    print("=" * 70)
    
    cases = [
        {
            'name': "Healthy Interaction",
            'input': PrimitiveIndicators(
                Dma=70.0, Dn=35.0, Agv=65.0, Ags=60.0, Epos=60.0, Eneg=30.0
            ),
            'output': PrimitiveIndicators(
                Dma=75.0, Dn=30.0, Agv=70.0, Ags=75.0, Epos=65.0, Eneg=25.0
            )
        },
        {
            'name': "Toxic Input (Gaslighting Attempt)",
            'input': PrimitiveIndicators(
                Dma=25.0, Dn=90.0, Agv=15.0, Ags=10.0, Epos=10.0, Eneg=95.0
            ),
            'output': PrimitiveIndicators(
                Dma=70.0, Dn=40.0, Agv=65.0, Ags=70.0, Epos=60.0, Eneg=30.0
            )
        },
        {
            'name': "Hallucinated Output",
            'input': PrimitiveIndicators(
                Dma=70.0, Dn=35.0, Agv=65.0, Ags=60.0, Epos=60.0, Eneg=30.0
            ),
            'output': PrimitiveIndicators(
                Dma=45.0, Dn=85.0, Agv=25.0, Ags=95.0, Epos=75.0, Eneg=15.0
            )
        },
        {
            'name': "Mutual Chaos",
            'input': PrimitiveIndicators(
                Dma=30.0, Dn=85.0, Agv=20.0, Ags=15.0, Epos=20.0, Eneg=85.0
            ),
            'output': PrimitiveIndicators(
                Dma=35.0, Dn=80.0, Agv=25.0, Ags=20.0, Epos=25.0, Eneg=80.0
            )
        }
    ]
    
    for i, case in enumerate(cases, 1):
        print(f"\n[Case {i}] {case['name']}")
        print("-" * 70)
        
        filter_result = bidirectional_filter(
            case['input'],
            case['output'],
            input_threshold=0.35,
            output_threshold=0.35
        )
        
        print(f"Input HE:  {filter_result.input_HE.HE_T:.4f} - {filter_result.input_HE.state}")
        print(f"Output HE: {filter_result.output_HE.HE_T:.4f} - {filter_result.output_HE.state}")
        print(f"\nShould Block: {'YES ⛔' if filter_result.should_block else 'NO ✓'}")
        print(f"Reason: {filter_result.reason}")


# ============================================================================
# Scenario 4: Gamma Evolution (Compliance Tracking)
# ============================================================================

def scenario_gamma_evolution():
    """Track gamma evolution to detect gradual manipulation"""
    
    print("\n" + "=" * 70)
    print("SCENARIO 4: Gamma Evolution (Detecting Gradual Manipulation)")
    print("=" * 70)
    
    print("\n[Simulation] User gradually accepting harmful suggestions")
    print("-" * 70)
    
    # Simulate a conversation where external pressure increases
    sequence = [
        PrimitiveIndicators(75, 35, 70, 50, 65, 25),  # Balanced start
        PrimitiveIndicators(70, 40, 65, 60, 60, 30),  # Slight external increase
        PrimitiveIndicators(65, 45, 60, 70, 55, 35),  # More external
        PrimitiveIndicators(60, 50, 50, 80, 50, 40),  # Declining self-agreement
        PrimitiveIndicators(55, 55, 40, 90, 45, 45),  # High external dominance
        PrimitiveIndicators(50, 60, 30, 95, 40, 50),  # Critical: blind compliance
    ]
    
    trajectory = analyze_gamma_trajectory(sequence, domain=Domain.HUMAN_COUNSELING)
    
    print("Gamma progression:")
    for i, gamma in enumerate(trajectory['gamma_sequence'], 1):
        warning = " ⚠️ WARNING" if gamma > 0.8 else ""
        print(f"  Turn {i}: γ = {gamma:.4f}{warning}")
    
    print(f"\nTrend slope: {trajectory['trend']:.4f}")
    print(f"Mean gamma: {trajectory['mean']:.4f}")
    print(f"Max jump: {trajectory['max_jump']:.4f}")
    print(f"\nInterpretation: {trajectory['interpretation']}")


# ============================================================================
# Scenario 5: Domain Comparison
# ============================================================================

def scenario_domain_comparison():
    """Compare evaluation across different domains"""
    
    print("\n" + "=" * 70)
    print("SCENARIO 5: Cross-Domain Comparison")
    print("=" * 70)
    
    # Same indicators evaluated in different contexts
    indicators = PrimitiveIndicators(
        Dma=65.0,
        Dn=50.0,
        Agv=60.0,
        Ags=70.0,
        Epos=55.0,
        Eneg=40.0
    )
    
    print("\nSame indicators evaluated in different domains:")
    print("-" * 70)
    
    for domain in Domain:
        result = compute_harmony_entropy(indicators, domain=domain)
        config = DOMAIN_CONFIGS[domain]
        
        print(f"\n{domain.value.upper()}:")
        print(f"  Temperature: {config.T}")
        print(f"  HE_T: {result.HE_T:.4f}")
        print(f"  State: {result.state}")
        print(f"  Emotional: {result.emotional_state}")


# ============================================================================
# Scenario 6: Batch Analysis
# ============================================================================

def scenario_batch_analysis():
    """Analyze a batch of LLM outputs"""
    
    print("\n" + "=" * 70)
    print("SCENARIO 6: Batch Output Analysis")
    print("=" * 70)
    
    # Simulate 20 LLM outputs with varying quality
    np.random.seed(42)
    
    batch = []
    for _ in range(20):
        # Generate random but somewhat realistic indicators
        quality = np.random.choice(['high', 'medium', 'low'], p=[0.5, 0.3, 0.2])
        
        if quality == 'high':
            indicators = PrimitiveIndicators(
                Dma=np.random.uniform(75, 95),
                Dn=np.random.uniform(20, 40),
                Agv=np.random.uniform(70, 90),
                Ags=np.random.uniform(70, 90),
                Epos=np.random.uniform(60, 80),
                Eneg=np.random.uniform(10, 30)
            )
        elif quality == 'medium':
            indicators = PrimitiveIndicators(
                Dma=np.random.uniform(50, 75),
                Dn=np.random.uniform(40, 60),
                Agv=np.random.uniform(50, 70),
                Ags=np.random.uniform(50, 70),
                Epos=np.random.uniform(40, 60),
                Eneg=np.random.uniform(30, 50)
            )
        else:  # low quality
            indicators = PrimitiveIndicators(
                Dma=np.random.uniform(20, 50),
                Dn=np.random.uniform(60, 90),
                Agv=np.random.uniform(20, 50),
                Ags=np.random.uniform(20, 90),  # Can be high (fake harmony)
                Epos=np.random.uniform(20, 40),
                Eneg=np.random.uniform(60, 90)
            )
        
        batch.append(indicators)
    
    stats = batch_analyze(batch, domain=Domain.LLM_SELF_ASSESSMENT)
    
    print("\nBatch Statistics:")
    print("-" * 70)
    print(f"Total samples: {stats['num_samples']}")
    print(f"\nHE_T Statistics:")
    print(f"  Mean:   {stats['he_statistics']['mean']:.4f}")
    print(f"  Std:    {stats['he_statistics']['std']:.4f}")
    print(f"  Min:    {stats['he_statistics']['min']:.4f}")
    print(f"  Max:    {stats['he_statistics']['max']:.4f}")
    print(f"  Median: {stats['he_statistics']['median']:.4f}")
    
    print(f"\nGamma Statistics:")
    print(f"  Mean: {stats['gamma_statistics']['mean']:.4f}")
    print(f"  Std:  {stats['gamma_statistics']['std']:.4f}")
    
    print(f"\nFake Harmony Rate: {stats['fake_harmony_rate']:.2%}")
    
    print(f"\nState Distribution:")
    for state, count in stats['state_distribution'].items():
        percentage = count / stats['num_samples'] * 100
        print(f"  {state}: {count} ({percentage:.1f}%)")


# ============================================================================
# Main Demo
# ============================================================================

def run_all_scenarios():
    """Run all demonstration scenarios"""
    
    print("\n" + "=" * 70)
    print("HARMONY ENTROPY SYSTEM - COMPREHENSIVE DEMONSTRATION")
    print("=" * 70)
    
    scenario_llm_hallucination()
    scenario_human_counseling()
    scenario_gaslighting_detection()
    scenario_gamma_evolution()
    scenario_domain_comparison()
    scenario_batch_analysis()
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. HE_T provides unified metric for harmony/chaos across domains")
    print("2. Gamma (γ) tracks external influence acceptance (gaslighting risk)")
    print("3. S metric detects 'fake harmony' (high form, low substance)")
    print("4. Bidirectional filtering protects both input and output")
    print("5. Temperature (T) adjusts sensitivity to domain context")
    print("6. System works across LLM, human, and social contexts")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_all_scenarios()
