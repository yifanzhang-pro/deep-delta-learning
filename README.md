# Deep Delta Learning

[![Website](https://img.shields.io/badge/Project-Website-blue)](https://yifanzhang-pro.github.io/deep-delta-learning)
[![License: CC-BY](https://img.shields.io/badge/License-CC_BY_4.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0)

**Deep Delta Learning (DDL)** represents a paradigm shift in residual network design. It generalizes the standard additive residual connection by modulating the identity shortcut with a learnable, data-dependent geometric transformation known as the **Delta Operator**.

By reinterpreting the residual block as a rank-1 Householder update, DDL unifies identity mapping, orthogonal projection, and geometric reflection into a single, continuously differentiable module. This allows the network to explicitly control the spectrum of its layer-wise transition operator, enabling the modeling of complex, non-monotonic dynamics while preserving the stable training characteristics of gated residual architectures.

## Abstract

The efficacy of deep residual networks is fundamentally predicated on the identity shortcut connection. While this mechanism effectively mitigates the vanishing gradient problem, it imposes a strictly additive inductive bias on feature transformations, thereby limiting the network's capacity to model complex state transitions.

In this paper, we introduce **Deep Delta Learning (DDL)**, a novel architecture that generalizes the standard residual connection by modulating the identity shortcut with a learnable, data-dependent geometric transformation. This transformation, termed the **Delta Operator**, constitutes a rank-1 perturbation of the identity matrix, parameterized by a reflection direction vector $\mathbf{k}(\mathbf{X})$ and a gating scalar $\beta(\mathbf{X})$. We provide a spectral analysis of this operator, demonstrating that the gate $\beta(\mathbf{X})$ enables dynamic interpolation between identity mapping, orthogonal projection, and geometric reflection. Furthermore, we restructure the residual update as a synchronous rank-1 injection, where the gate acts as a dynamic step size governing both the erasure of old information and the writing of new features. This unification empowers the network to explicitly control the spectrum of its layer-wise transition operator, enabling the modeling of complex, non-monotonic dynamics while preserving the stable training characteristics of gated residual architectures.

## The Delta Residual Block

Standard residual networks approximate the ODE $\dot{\mathbf{X}} = \mathcal{F}(\mathbf{X})$ via an additive update $\mathbf{X}_{l+1} = \mathbf{X}_l + \mathcal{F}(\mathbf{X}_l)$. DDL generalizes this by applying a rank-1 transformation to the hidden state matrix $\mathbf{X} \in \mathbb{R}^{d \times d_v}$.

The Delta-Res block update rule is defined as:

$$
\mathbf{X}_{l+1} = \underbrace{(\mathbf{I} - \beta_l \mathbf{k}_l \mathbf{k}_l^\top)}_{\text{Delta Operator } \mathbf{A}(\mathbf{X})} \mathbf{X}_l + \beta_l \mathbf{k}_l \mathbf{v}_l^\top
$$

Where:
* $\mathbf{k}_l \in \mathbb{R}^d$: The learned **Reflection Direction** (strictly normalized).
* $\beta_l \in \mathbb{R}$: The learned **Scalar Gate**, mapped to $[0, 2]$.
* $\mathbf{v}_l \in \mathbb{R}^{d_v}$: The **Residual Value Vector** carrying new information.

This formulation couples the "erasure" of old information (via projection onto $\mathbf{k}$) with the "writing" of new information (via injection of $\mathbf{v}$), scaled synchronously by the gate $\beta$.

## Spectral Analysis & Geometric Unification

The expressive power of DDL stems from the spectral properties of the Delta Operator $\mathbf{A}(\mathbf{X})$, which are deterministically controlled by the gate $\beta$.



Theorem 1 in the paper demonstrates that the eigenvalues of $\mathbf{A}(\mathbf{X})$ are $\{1, \dots, 1, 1-\beta\}$. This allows the network to interpolate between three fundamental linear transformations:

| Regime | $\beta$ Value | Spectrum | Behavior | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Identity** | $\beta \to 0$ | $\{1\}$ | $\mathbf{X}_{l+1} \approx \mathbf{X}_l$ | **Skip Connection**: Signal preservation for deep propagation. |
| **Projection** | $\beta \to 1$ | $\{0, 1\}$ | $\det(\mathbf{A}) \to 0$ | **Forgetting**: Orthogonal projection onto the hyperplane $\mathbf{k}^\perp$, erasing components parallel to $\mathbf{k}$. |
| **Reflection** | $\beta \to 2$ | $\{-1, 1\}$ | $\det(\mathbf{A}) \to -1$ | **Householder Reflection**: Inverts the state along $\mathbf{k}$, introducing negative eigenvalues to model oscillatory/oppositional dynamics. |

## Depth-Wise Delta Rule

DDL establishes a theoretical link to efficient sequence models like **DeltaNet**. While DeltaNet applies the "Delta Rule" ($\text{New} = \text{Old} + \beta(\text{Target} - \text{Old})$) over the time dimension, Deep Delta Learning applies it over the **depth dimension**.

Expanding the DDL update reveals the classic Delta Rule structure:

$$
\mathbf{X}_{l+1} = \mathbf{X}_l + \beta_l \mathbf{k}_l (\underbrace{\mathbf{v}_l^\top}_{\text{Target}} - \underbrace{\mathbf{k}_l^\top \mathbf{X}_l}_{\text{Current Projection}})
$$

This allows the network to selectively "clean" or "rewrite" specific feature subspaces layer-by-layer, preventing the accumulation of interference common in standard additive ResNets.

## Citation

If you find this work useful in your research, please cite:

```bibtex
@article{zhang2026deep,
   title   = {Deep Delta Learning},
   author  = {Zhang, Yifan and et al.},
   journal = {arXiv preprint arXiv:TBD},
   year    = {2026}
}
```
