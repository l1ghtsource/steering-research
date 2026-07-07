# Latent Behavioral Features for Controlling LLM Reasoning
 
Large language models often exhibit persistent behavioral patterns that influence their reasoning beyond any single task. Remarkably, recent work has shown that fine-tuning on insecure code can create a latent "bad persona", leading to deceptive or unsafe behavior even in unrelated domains. This suggests that high-level behavioral traits are encoded as reusable internal features that can be identified and manipulated.

This project investigates whether latent behavioral features can be identified and steered to control reasoning in large language models. We will combine activation steering, sparse feature discovery, and behavioral evaluations to develop interpretable methods for controlling reasoning while preserving task performance.

# Plan

1. Build a benchmark of positive and negative behavioral phenomena in LLM reasoning and agentic workflows (i.e. hallucination, deception, sycophancy, overconfidence, premature refusal, unsafe planning).
2. Identify candidate latent behavioral features using representation analysis methods such as Contrastive Activation Addition (CAA), Sparse Autoencoders (SAEs), and related techniques.
3. Evaluate and compare behavioral steering methods to determine which approaches most effectively control reasoning while preserving task performance.
4. Investigate whether behavioral features discovered in one setting transfer to unseen tasks and different LLM architectures.
    
# Reading:

https://arxiv.org/pdf/2502.17424
https://arxiv.org/pdf/2506.19823
https://arxiv.org/pdf/2507.21509
https://arxiv.org/pdf/2604.07729