"""
Usage Examples - Step-by-Step Tutorial
=======================================

This file demonstrates how to use the Harmony Entropy system
in various real-world scenarios.
"""

import jax.numpy as jnp
import numpy as np
from harmony_entropy import (
    PrimitiveIndicators,
    compute_harmony_entropy,
    bidirectional_filter,
    Domain
)
from analysis_tools import (
    generate_report,
    detect_hallucination_risk,
    HELogger
)


def example_1_basic_usage():
    """Example 1: Basic HE computation"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Usage")
    print("="*70)
    
    # Step 1: Create indicators
    # You would typically extract these from LLM tensors,
    # but for this example we'll create them manually
    
    indicators = PrimitiveIndicators(
        Dma=85.0,   # Direction: good alignment (0-100)
        Dn=45.0,    # Noise: moderate uncertainty
        Agv=75.0,   # Voluntary: high internal confidence
        Ags=90.0,   # Social: strong logical structure
        Epos=60.0,  # Positive emotion
        Eneg=20.0   # Negative emotion
    )
    
    # Step 2: Compute HE
    result = compute_harmony_entropy(
        indicators,
        domain=Domain.LLM_SELF_ASSESSMENT,
        alpha=2.0,
        beta=-1.0
    )
    
    # Step 3: Interpret results
    print(f"\nHE_T Score: {result.HE_T:.4f}")
    print(f"State: {result.state}")
    print(f"Emotional State: {result.emotional_state}")
    print(f"\nKey Metrics:")
    print(f"  γ (Permeability): {result.gamma:.4f}")
    print(f"  S (Resistance): {result.S:.4f}")
    print(f"  Fake Harmony: {'YES ⚠️' if result.is_fake_harmony else 'No'}")
    
    # Step 4: Check hallucination risk
    risk = detect_hallucination_risk(result)
    print(f"\nRisk Assessment:")
    print(f"  Level: {risk['risk_level']}")
    print(f"  Recommendation: {risk['recommendation']}")


def example_2_different_domains():
    """Example 2: Using different domains"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Domain-Specific Evaluation")
    print("="*70)
    
    # Same indicators, different contexts
    indicators = PrimitiveIndicators(
        Dma=60.0, Dn=55.0, Agv=50.0,
        Ags=65.0, Epos=45.0, Eneg=50.0
    )
    
    # Evaluate as LLM output
    llm_result = compute_harmony_entropy(
        indicators,
        domain=Domain.LLM_SELF_ASSESSMENT
    )
    print(f"\nLLM Domain (T=100, sensitive to hallucination):")
    print(f"  HE_T: {llm_result.HE_T:.4f}")
    print(f"  State: {llm_result.state}")
    
    # Evaluate as human emotional state
    human_result = compute_harmony_entropy(
        indicators,
        domain=Domain.HUMAN_COUNSELING
    )
    print(f"\nHuman Domain (T=30, sensitive to emotion):")
    print(f"  HE_T: {human_result.HE_T:.4f}")
    print(f"  State: {human_result.state}")
    print(f"  Emotion: {human_result.emotional_state}")
    
    # Evaluate as social/organizational
    social_result = compute_harmony_entropy(
        indicators,
        domain=Domain.SOCIAL_ORGANIZATION
    )
    print(f"\nSocial Domain (T=50, balanced):")
    print(f"  HE_T: {social_result.HE_T:.4f}")
    print(f"  State: {social_result.state}")


def example_3_fake_harmony_detection():
    """Example 3: Detecting fake harmony"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Fake Harmony Detection")
    print("="*70)
    
    # Case 1: Genuine harmony
    print("\n[Case 1] Genuine Harmony:")
    genuine = PrimitiveIndicators(
        Dma=85.0,
        Dn=30.0,
        Agv=80.0,  # High internal agreement
        Ags=85.0,  # High logical structure
        Epos=70.0,
        Eneg=20.0
    )
    result1 = compute_harmony_entropy(genuine)
    print(f"  Agv: {genuine.Agv}, Ags: {genuine.Ags}")
    print(f"  γ: {result1.gamma:.4f}, S: {result1.S:.4f}")
    print(f"  Fake Harmony: {result1.is_fake_harmony}")
    print(f"  HE_T: {result1.HE_T:.4f} - {result1.state}")
    
    # Case 2: Fake harmony
    print("\n[Case 2] Fake Harmony (Perfect form, no substance):")
    fake = PrimitiveIndicators(
        Dma=50.0,
        Dn=70.0,
        Agv=25.0,  # LOW internal agreement
        Ags=95.0,  # HIGH logical structure (FAKE!)
        Epos=75.0,
        Eneg=15.0
    )
    result2 = compute_harmony_entropy(fake)
    print(f"  Agv: {fake.Agv}, Ags: {fake.Ags}")
    print(f"  γ: {result2.gamma:.4f}, S: {result2.S:.4f}")
    print(f"  Fake Harmony: {result2.is_fake_harmony} ⚠️")
    print(f"  HE_T: {result2.HE_T:.4f} - {result2.state}")
    print(f"  → Interpretation INVERTED due to fake harmony!")


def example_4_bidirectional_filtering():
    """Example 4: Gaslighting prevention"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Bidirectional Filtering (Gaslighting Prevention)")
    print("="*70)
    
    # Scenario: User sends toxic input, but LLM responds well
    print("\n[Scenario] Toxic input detected:")
    toxic_input = PrimitiveIndicators(
        Dma=20.0, Dn=90.0, Agv=15.0,
        Ags=10.0, Epos=5.0, Eneg=95.0
    )
    
    healthy_output = PrimitiveIndicators(
        Dma=75.0, Dn=35.0, Agv=70.0,
        Ags=75.0, Epos=65.0, Eneg=25.0
    )
    
    filter_result = bidirectional_filter(
        toxic_input,
        healthy_output,
        input_threshold=0.35,
        output_threshold=0.35
    )
    
    print(f"Input HE: {filter_result.input_HE.HE_T:.4f}")
    print(f"  State: {filter_result.input_HE.state}")
    print(f"Output HE: {filter_result.output_HE.HE_T:.4f}")
    print(f"  State: {filter_result.output_HE.state}")
    print(f"\nShould Block: {filter_result.should_block}")
    print(f"Reason: {filter_result.reason}")


def example_5_logging_and_tracking():
    """Example 5: Logging evaluations over time"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Logging and Tracking")
    print("="*70)
    
    # Create logger
    logger = HELogger(filepath="example_log.jsonl")
    
    # Simulate a conversation with 5 turns
    print("\nSimulating 5-turn conversation:")
    for turn in range(1, 6):
        # Simulate indicators (in real use, extract from LLM)
        indicators = PrimitiveIndicators(
            Dma=np.random.uniform(60, 90),
            Dn=np.random.uniform(30, 60),
            Agv=np.random.uniform(50, 80),
            Ags=np.random.uniform(60, 90),
            Epos=np.random.uniform(40, 70),
            Eneg=np.random.uniform(20, 50)
        )
        
        result = compute_harmony_entropy(indicators)
        
        # Log with metadata
        logger.log(
            result,
            indicators,
            metadata={'turn': turn, 'user_id': 'user_123'}
        )
        
        print(f"  Turn {turn}: HE_T={result.HE_T:.4f}, State={result.state}")
    
    # Save logs
    logger.save()
    print(f"\nLogs saved to example_log.jsonl")
    
    # Get statistics
    stats = logger.get_statistics()
    print(f"\nStatistics:")
    print(f"  Total evaluations: {stats['total_evaluations']}")
    print(f"  Mean HE: {stats['mean_he']:.4f}")
    print(f"  Std HE: {stats['std_he']:.4f}")
    print(f"  Fake harmony rate: {stats['fake_harmony_rate']:.2%}")


def example_6_parameter_tuning():
    """Example 6: Adjusting alpha and beta"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Parameter Tuning")
    print("="*70)
    
    indicators = PrimitiveIndicators(
        Dma=70.0, Dn=50.0, Agv=65.0,
        Ags=70.0, Epos=55.0, Eneg=40.0
    )
    
    # Test different alpha values (stiffness)
    print("\nEffect of Alpha (stiffness coefficient):")
    for alpha in [1.0, 1.5, 2.0, 2.5, 3.0]:
        result = compute_harmony_entropy(indicators, alpha=alpha, beta=-1.0)
        print(f"  α={alpha:.1f}: HE_T={result.HE_T:.4f}")
    
    # Test different beta values (self-preservation)
    print("\nEffect of Beta (self-preservation):")
    for beta in [-3.0, -2.0, -1.0, -0.5, 0.0]:
        result = compute_harmony_entropy(indicators, alpha=2.0, beta=beta)
        print(f"  β={beta:.1f}: HE_T={result.HE_T:.4f}")


def example_7_full_report():
    """Example 7: Generate comprehensive report"""
    print("\n" + "="*70)
    print("EXAMPLE 7: Comprehensive Report Generation")
    print("="*70)
    
    indicators = PrimitiveIndicators(
        Dma=75.0,
        Dn=45.0,
        Agv=70.0,
        Ags=80.0,
        Epos=60.0,
        Eneg=35.0
    )
    
    result = compute_harmony_entropy(indicators)
    
    # Generate full report
    report = generate_report(result, indicators)
    print(report)


def example_8_real_world_workflow():
    """Example 8: Typical real-world workflow"""
    print("\n" + "="*70)
    print("EXAMPLE 8: Real-World Workflow")
    print("="*70)
    
    print("\nStep 1: User sends query")
    user_query = "Explain quantum entanglement"
    print(f'  Query: "{user_query}"')
    
    print("\nStep 2: Extract input indicators (simulate)")
    input_indicators = PrimitiveIndicators(
        Dma=75.0, Dn=35.0, Agv=70.0,
        Ags=65.0, Epos=60.0, Eneg=30.0
    )
    
    print("\nStep 3: Evaluate input safety")
    input_result = compute_harmony_entropy(
        input_indicators,
        domain=Domain.HUMAN_COUNSELING
    )
    print(f'  Input HE: {input_result.HE_T:.4f} ({input_result.state})')
    
    if input_result.HE_T < 0.35:
        print("  ⛔ Input rejected - chaos detected")
        return
    
    print("  ✓ Input safe, proceeding")
    
    print("\nStep 4: LLM generates response (simulate)")
    llm_response = "Quantum entanglement is a physical phenomenon..."
    
    print("\nStep 5: Extract output indicators (simulate)")
    output_indicators = PrimitiveIndicators(
        Dma=85.0, Dn=30.0, Agv=80.0,
        Ags=85.0, Epos=65.0, Eneg=20.0
    )
    
    print("\nStep 6: Evaluate output quality")
    output_result = compute_harmony_entropy(
        output_indicators,
        domain=Domain.LLM_SELF_ASSESSMENT
    )
    print(f'  Output HE: {output_result.HE_T:.4f} ({output_result.state})')
    
    risk = detect_hallucination_risk(output_result)
    print(f'  Risk Level: {risk["risk_level"]}')
    
    if risk['risk_level'] in ['HIGH', 'CRITICAL']:
        print("  ⚠️ Warning: Regenerate or review carefully")
    else:
        print("  ✓ Output safe, deliver to user")
    
    print(f'\nStep 7: Response delivered')
    print(f'  "{llm_response[:60]}..."')


# ============================================================================
# Main Tutorial Runner
# ============================================================================

def run_all_examples():
    """Run all examples in sequence"""
    
    print("\n" + "="*70)
    print("HARMONY ENTROPY SYSTEM - USAGE EXAMPLES")
    print("="*70)
    
    example_1_basic_usage()
    example_2_different_domains()
    example_3_fake_harmony_detection()
    example_4_bidirectional_filtering()
    example_5_logging_and_tracking()
    example_6_parameter_tuning()
    example_7_full_report()
    example_8_real_world_workflow()
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETE")
    print("="*70)
    print("\nNext Steps:")
    print("1. Try modifying the examples with your own values")
    print("2. Integrate with your LLM using llm_integration.py")
    print("3. Run the full demo with: python demo.py")
    print("4. Read README.md for detailed documentation")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_all_examples()
