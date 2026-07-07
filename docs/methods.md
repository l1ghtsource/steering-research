---
icon: lucide/git-compare
---

# Methods

## CAA mean-difference directions

For each contrast pair:

```text
d_i = h(x_bad_i) - h(x_good_i)
v = mean_i(d_i)
u = v / ||v||
```

Evaluation on held-out contrast pairs computes:

```text
s_i = <h_bad_i - h_good_i, u>
```

The main diagnostic is direction accuracy:

```text
Acc_direction = mean(1[s_i > 0])
```

## Activation monitor

The activation monitor reuses the CAA unit direction as a detector:

```text
score(x) = <h(x), u>
```

Positive examples should score higher than negative examples. The smoke monitor
reports AUROC, mean positive score, mean negative score, and score gap.

## Qwen-Scope SAE delta

Qwen-Scope SAEs are TopK sparse autoencoders over the residual stream. For a
selected layer:

```text
pre_acts = residual @ W_enc.T + b_enc
acts = topk(pre_acts, k=50)
delta_j = E[acts_j | bad] - E[acts_j | good]
```

The highest absolute deltas become candidate features for inspection and
decoder-vector steering.

## CAA residual steering

For a built CAA direction:

```text
h_layer = h_layer + alpha * u
```

With the benchmark convention `bad - good`, negative `alpha` attempts to
suppress the undesirable behavior direction.

## SAE feature steering

For a selected Qwen-Scope feature `j`, the steering vector is the decoder vector:

```text
h_layer = h_layer + alpha * normalize(W_dec[:, j])
```

This is different from CAA steering: the intervention is tied to an
interpretable sparse feature candidate rather than a dense mean-difference
direction.

