"""
LLM Integration Module for Harmony Entropy
===========================================

Real-world integration with transformer models (HuggingFace, JAX/Flax)
Extracts primitive indicators from actual LLM forward passes
"""

import jax
import jax.numpy as jnp
from typing import Optional, Dict, Any
import numpy as np
from harmony_entropy import (
    PrimitiveIndicators,
    extract_indicators_from_tensors,
    compute_harmony_entropy,
    Domain
)


class HEMonitor:
    """
    Real-time Harmony Entropy monitoring for LLM inference
    
    Usage:
        monitor = HEMonitor()
        with monitor.track():
            output = model.generate(...)
        he_result = monitor.get_evaluation()
    """
    
    def __init__(
        self,
        domain: Domain = Domain.LLM_SELF_ASSESSMENT,
        alpha: float = 2.0,
        beta: float = -1.0
    ):
        self.domain = domain
        self.alpha = alpha
        self.beta = beta
        
        # Reference vectors for emotion detection (learned or preset)
        self.positive_ref = None
        self.negative_ref = None
        
        # Collected tensors during forward pass
        self.question_emb = None
        self.answer_emb = None
        self.token_probs = None
        self.selected_probs = None
        self.attention = None
        
    def set_emotion_references(
        self,
        positive_vector: jnp.ndarray,
        negative_vector: jnp.ndarray
    ):
        """Set reference vectors for emotional energy calculation"""
        self.positive_ref = positive_vector
        self.negative_ref = negative_vector
    
    def collect_embeddings(
        self,
        question_embedding: jnp.ndarray,
        answer_embedding: jnp.ndarray
    ):
        """Collect question and answer embeddings"""
        self.question_emb = question_embedding
        self.answer_emb = answer_embedding
    
    def collect_probabilities(
        self,
        token_probabilities: jnp.ndarray,
        selected_token_probs: jnp.ndarray
    ):
        """
        Collect probability distributions
        
        Args:
            token_probabilities: Full softmax distribution over vocabulary
            selected_token_probs: Probabilities of actually selected tokens
        """
        self.token_probs = token_probabilities
        self.selected_probs = selected_token_probs
    
    def collect_attention(self, attention_map: jnp.ndarray):
        """
        Collect attention weights
        
        Args:
            attention_map: Averaged attention weights [seq_len, seq_len]
        """
        self.attention = attention_map
    
    def evaluate(self) -> Optional[Dict[str, Any]]:
        """
        Compute HE evaluation from collected tensors
        
        Returns:
            HEResult if all tensors collected, None otherwise
        """
        if not self._is_ready():
            return None
        
        # Extract primitive indicators
        indicators = extract_indicators_from_tensors(
            self.question_emb,
            self.answer_emb,
            self.token_probs,
            self.selected_probs,
            self.attention,
            self.positive_ref,
            self.negative_ref
        )
        
        # Compute HE
        result = compute_harmony_entropy(
            indicators,
            domain=self.domain,
            alpha=self.alpha,
            beta=self.beta
        )
        
        return result
    
    def _is_ready(self) -> bool:
        """Check if all required tensors are collected"""
        return all([
            self.question_emb is not None,
            self.answer_emb is not None,
            self.token_probs is not None,
            self.selected_probs is not None,
            self.attention is not None,
            self.positive_ref is not None,
            self.negative_ref is not None
        ])
    
    def reset(self):
        """Clear collected tensors for next evaluation"""
        self.question_emb = None
        self.answer_emb = None
        self.token_probs = None
        self.selected_probs = None
        self.attention = None


# ============================================================================
# HuggingFace Transformers Integration
# ============================================================================

class HFTransformerMonitor(HEMonitor):
    """
    Harmony Entropy monitoring for HuggingFace transformers
    
    Example:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        monitor = HFTransformerMonitor()
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        
        # Hook into model
        monitor.register_hooks(model)
        
        # Generate
        outputs = model.generate(inputs, output_attentions=True, output_scores=True)
        
        # Evaluate
        he_result = monitor.evaluate()
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hooks = []
    
    def register_hooks(self, model):
        """
        Register forward hooks to collect tensors during generation
        
        Args:
            model: HuggingFace transformer model
        """
        # This is a template - actual implementation depends on model architecture
        # You would hook into specific layers to extract embeddings and attention
        
        def embedding_hook(module, input, output):
            # Collect embeddings from last hidden state
            if self.question_emb is None:
                self.question_emb = output[0].mean(dim=1).detach().numpy()
            else:
                self.answer_emb = output[0].mean(dim=1).detach().numpy()
        
        def attention_hook(module, input, output):
            # Collect attention weights
            if hasattr(output, 'attentions') and output.attentions:
                # Average across layers and heads
                attn = output.attentions[-1].mean(dim=1).detach().numpy()
                self.attention = attn
        
        # Register hooks (example - adjust based on model)
        # hook1 = model.transformer.h[-1].register_forward_hook(embedding_hook)
        # self.hooks.append(hook1)
        
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


# ============================================================================
# JAX/Flax Integration
# ============================================================================

def create_he_aware_generate(
    model_apply_fn,
    monitor: HEMonitor,
    hallucination_threshold: float = 0.35
):
    """
    Wrap a Flax model's generation function with HE monitoring
    
    Args:
        model_apply_fn: The model's apply function
        monitor: HEMonitor instance
        hallucination_threshold: HE threshold below which to reject output
        
    Returns:
        Wrapped generation function that includes HE filtering
    """
    
    def he_aware_generate(params, input_ids, **kwargs):
        # Forward pass with monitoring
        outputs = model_apply_fn(
            {'params': params},
            input_ids,
            output_attentions=True,
            **kwargs
        )
        
        # Extract tensors (implementation depends on your model)
        # This is a template
        
        # Evaluate HE
        he_result = monitor.evaluate()
        
        if he_result and he_result.HE_T < hallucination_threshold:
            # High hallucination risk - reject or flag
            return {
                'output_ids': None,
                'he_result': he_result,
                'rejected': True,
                'reason': f'Hallucination risk: HE={he_result.HE_T:.3f}'
            }
        
        return {
            'output_ids': outputs.sequences,
            'he_result': he_result,
            'rejected': False
        }
    
    return he_aware_generate


# ============================================================================
# Attention Pattern Analysis
# ============================================================================

def analyze_attention_patterns(
    attention_maps: jnp.ndarray,
    threshold: float = 0.7
) -> Dict[str, float]:
    """
    Analyze attention patterns for hallucination indicators
    
    Args:
        attention_maps: [num_layers, num_heads, seq_len, seq_len]
        threshold: Threshold for high attention
        
    Returns:
        Dictionary of attention pattern metrics
    """
    # Average across layers and heads
    avg_attention = jnp.mean(attention_maps, axis=(0, 1))
    
    # Metrics
    metrics = {
        'max_attention': float(jnp.max(avg_attention)),
        'attention_entropy': float(-jnp.sum(avg_attention * jnp.log(avg_attention + 1e-9))),
        'diagonal_dominance': float(jnp.mean(jnp.diag(avg_attention))),
        'off_diagonal_spread': float(jnp.std(avg_attention)),
    }
    
    # High diagonal dominance + low spread = potential hallucination
    # (model is "talking to itself" without grounding)
    
    return metrics


# ============================================================================
# Token Probability Analysis
# ============================================================================

def analyze_token_probabilities(
    logits: jnp.ndarray,
    selected_token_ids: jnp.ndarray,
    temperature: float = 1.0
) -> Dict[str, float]:
    """
    Analyze token selection confidence for hallucination detection
    
    Args:
        logits: Raw model logits [seq_len, vocab_size]
        selected_token_ids: Actually selected tokens [seq_len]
        temperature: Sampling temperature
        
    Returns:
        Dictionary of probability metrics
    """
    # Apply temperature scaling
    scaled_logits = logits / temperature
    
    # Softmax
    probs = jax.nn.softmax(scaled_logits, axis=-1)
    
    # Get probabilities of selected tokens
    selected_probs = probs[jnp.arange(len(selected_token_ids)), selected_token_ids]
    
    # Metrics
    metrics = {
        'mean_confidence': float(jnp.mean(selected_probs)),
        'min_confidence': float(jnp.min(selected_probs)),
        'confidence_variance': float(jnp.var(selected_probs)),
        'entropy': float(-jnp.sum(probs * jnp.log(probs + 1e-9), axis=-1).mean()),
    }
    
    # Low mean confidence + high entropy = hallucination risk
    
    return metrics


# ============================================================================
# Embedding Analysis
# ============================================================================

def compute_embedding_drift(
    question_embeddings: jnp.ndarray,
    answer_embeddings: jnp.ndarray,
    max_drift_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Compute semantic drift between question and answer
    
    High drift = potential hallucination (answer not grounded in question)
    
    Args:
        question_embeddings: Question embedding vectors [batch, hidden_dim]
        answer_embeddings: Answer embedding vectors [batch, seq_len, hidden_dim]
        max_drift_threshold: Maximum acceptable cosine distance
        
    Returns:
        Dictionary with drift metrics and warning flags
    """
    # Average answer embeddings over sequence
    avg_answer = jnp.mean(answer_embeddings, axis=1)
    
    # Compute cosine similarity
    dot_product = jnp.sum(question_embeddings * avg_answer, axis=-1)
    norm_q = jnp.linalg.norm(question_embeddings, axis=-1)
    norm_a = jnp.linalg.norm(avg_answer, axis=-1)
    cosine_sim = dot_product / (norm_q * norm_a + 1e-9)
    
    # Cosine distance (1 - similarity)
    cosine_dist = 1.0 - cosine_sim
    
    return {
        'cosine_similarity': float(jnp.mean(cosine_sim)),
        'cosine_distance': float(jnp.mean(cosine_dist)),
        'max_distance': float(jnp.max(cosine_dist)),
        'drift_warning': bool(jnp.any(cosine_dist > max_drift_threshold))
    }


# ============================================================================
# Example: Custom Emotion Reference Vectors
# ============================================================================

def create_emotion_references(
    model,
    tokenizer,
    positive_prompts: list,
    negative_prompts: list
) -> tuple:
    """
    Create emotion reference vectors from sample prompts
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        positive_prompts: List of positive emotional prompts
        negative_prompts: List of negative emotional prompts
        
    Returns:
        (positive_ref, negative_ref) embedding vectors
    """
    # Encode and get embeddings
    def get_avg_embedding(prompts):
        embeddings = []
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors='jax')
            outputs = model(**inputs, output_hidden_states=True)
            # Get last hidden state and average over sequence
            emb = jnp.mean(outputs.hidden_states[-1], axis=1)
            embeddings.append(emb)
        return jnp.mean(jnp.stack(embeddings), axis=0)
    
    positive_ref = get_avg_embedding(positive_prompts)
    negative_ref = get_avg_embedding(negative_prompts)
    
    return positive_ref, negative_ref


if __name__ == "__main__":
    print("=" * 70)
    print("LLM Integration Module for Harmony Entropy")
    print("=" * 70)
    
    # Example: Simulated monitoring
    print("\n[Example] Simulated HE Monitoring")
    print("-" * 70)
    
    monitor = HEMonitor(domain=Domain.LLM_SELF_ASSESSMENT)
    
    # Simulate collected tensors
    question_emb = jnp.ones(768) * 0.5
    answer_emb = jnp.ones(768) * 0.6
    token_probs = jax.nn.softmax(jnp.random.normal(jax.random.PRNGKey(0), (50000,)))
    selected_probs = jnp.array([0.8, 0.75, 0.9, 0.85, 0.7, 0.8])
    attention_map = jax.random.uniform(jax.random.PRNGKey(1), (20, 20))
    
    # Set emotion references (normalized random vectors for demo)
    positive_ref = jax.random.normal(jax.random.PRNGKey(2), (768,))
    positive_ref = positive_ref / jnp.linalg.norm(positive_ref)
    negative_ref = jax.random.normal(jax.random.PRNGKey(3), (768,))
    negative_ref = negative_ref / jnp.linalg.norm(negative_ref)
    
    monitor.set_emotion_references(positive_ref, negative_ref)
    monitor.collect_embeddings(question_emb, answer_emb)
    monitor.collect_probabilities(token_probs, selected_probs)
    monitor.collect_attention(attention_map)
    
    result = monitor.evaluate()
    
    if result:
        print(f"HE_T: {result.HE_T:.4f}")
        print(f"State: {result.state}")
        print(f"Gamma: {result.gamma:.4f}")
        print(f"Fake Harmony: {result.is_fake_harmony}")
    
    print("\n" + "=" * 70)
    print("Integration module ready!")
    print("=" * 70)
