# RKHS-optimal sampling

## Setting

Consider a RKHS $\mathcal{V}$ with kernel $k : \mathcal{X}\times\mathcal{X}\to\mathbb{R}$ and let the $d$-dimensional subspace $\mathcal{V}_d\subseteq\mathcal{V}$ be spanned by the $\mathcal{V}$-ONB $b_1,\ldots,b_d$.

Let $u\in L^2$ and a given sample $x_1, \ldots, x_n\in\mathcal{X}$ let $K_{ij} := k(x_i, x_j)$ then define
$$
    u_{d,n} := \argmin_{v\in\mathcal{V}_d} \|u - v\|_{n}
    \qquad\text{with}\qquad
    \|v\|_n := \boldsymbol{v}^\intercal K^+ \boldsymbol{v}
    \qquad\text{and}\qquad
    \boldsymbol{v}_i := v(x_i) .
$$
Defining further $u_{\star}\in\mathcal{V}$ as the $L^\infty$-best approximation of $u$ in $\mathcal{V}$ and $u_d\in\mathcal{V}_d$ as the $\mathcal{V}$-best approximation of $u_\star$ in $\mathcal{V}_d$, it has been shown that
$$
    \|u_\star - u_{d,n}\|_{\mathcal{V}}
    \le (1 + 2\mu) \|u_\star - u_d\|_{\mathcal{V}_d}  + 2\mu_\infty \|u - u_\star\|_{L^\infty} .
$$

Defining $B_{li} := b_l(x_i)$, the constants $\mu$ and $\mu_\infty$ are given by
$$
    \mu
    := \max_{v\in\mathcal{V}_d} \frac{\|v\|_{\mathcal{V}}}{\|v\|_{n}}
    = \lambda_{\mathrm{min}}(BK^+B^\intercal)^{-1/2}
    \qquad\text{and}\qquad
    \mu_\infty
    % = \mu\beta
    := \max_{v\in\mathcal{V}_d} \frac{\|v\|_{\mathcal{V}}}{\|\boldsymbol{v}\|_{\infty}} .
$$

> **Note:** We can not optimise $\mu$ and $\mu_\infty$ simultaneously!
> But observe that
> $$
> \begin{aligned}
>     \mu_\infty
>     % &= \max_{v\in\mathcal{V}_d} \frac{\|v\|_{\mathcal{V}}}{\|\boldsymbol{v}\|_{\infty}}
>     = \max_{c\in\mathbb{R}^d} \frac{\|c\|_{2}}{\|c^\intercal B\|_{\infty}}
>     \le \sqrt{n} \max_{c\in\mathbb{R}^d} \frac{\|c\|_{2}}{\|c^\intercal B\|_{2}}
>     = \left(\min_{\substack{c\in\mathbb{R}^d\\\|c\|_{2}=1}}\ \frac{1}{n}c^\intercal BB^\intercal c\right)^{-1/2}
>     = \lambda_{\mathrm{min}}(G)^{-1/2} ,
> \end{aligned}
> $$
> where $G$ is the empirical $L^2$-Gramian matrix of the $\mathcal{V}$-ONB $b_1,\ldots, b_d$.

Assuming $\|u-u_\star\|_{L^\infty}$ to be negligible, we can optimise the points as to minimise the consant $\mu$.
A common surrogate for the maximisation of the smallest eigenvalue is the determinant.

> **Note:** To see why the determinant is a good surrogate in this case, observe that
> $$
>     \lambda_{\mathrm{max}}(BK^+B^\intercal) = \max_{v\in\mathcal{V}_d} \frac{\|P_Wv\|_{\mathcal{V}}}{\|v\|_{\mathcal{V}}} \le 1 .
> $$
> This implies
> $$
>     \lambda_{\mathrm{min}}(BK^+B^\intercal)^d \le \det(BK^+B^\intercal) \le \lambda_{\mathrm{min}}(BK^+B^\intercal).
> $$
> Hence, a maximiser of the determinant will necessarily maximise the smallest eigenvalue.
> It also implies that
> $$
>     \det(BK^+B^\intercal) \le \lambda_{\mathrm{min}}(BK^+B^\intercal) \le \lambda_{\mathrm{max}}(BK^+B^\intercal) \le 1,
> $$
> which implies that $\int\det(BK^+B^\intercal)\,\mathrm{d}\rho^{\otimes n} \le 1$ is finite.

Instead of maximising the determinant function directly, we propose to draw samples from it and condition on the event $\mu \le \mu_0$.

## Well-posedness
For it to makes sense to draw a sample from the determinant
$$
    \det(BK^+ B^\intercal) ,
$$
we need to show that it is an (unnormalised) probability density function,
i.e. that it is
1. non-negative and
2. integrable.

To show non-negativity, recall that $K$ is a kernel matrix, and thus positive definite by definition.
This implies that $K^+$ is positive definite and that for any $c\in\mathbb{R}^d$
$$
    c^\intercal B K^+ B^\intercal c
    = (B^\intercal c)^\intercal K^+ (B^\intercal c)
    \ge 0 .
$$
This proves, that $\det(BK^+ B^\intercal)$ must always be non-negative.
To show integrability, observe that by non-negativity, the Leibniz formula and the triangle inequality
$$
\begin{aligned}
    \det(BK^+B^\intercal)
    &= |\det(BK^+B^\intercal)| \\
    &= \left|\sum_{\sigma \in S_d} \operatorname{sgn}(\sigma) \prod_{l = 1}^d \boldsymbol{b}_l^\intercal K^+ \boldsymbol{b}_{\sigma(l)}\right| \\
    &\le \sum_{\sigma \in S_d} \prod_{l = 1}^d |\boldsymbol{b}_l^\intercal K^+ \boldsymbol{b}_{\sigma(l)}| ,
\end{aligned}
$$
where $\boldsymbol{b}_l := (b_l(x_1), \ldots, b_l(x_n))^\intercal$.
Now let $W := \operatorname{span}\{k(x_1,\bullet), \ldots, k(x_n, \bullet)\}$ and observe that
$$
\begin{aligned}
    |\boldsymbol{b}_l^\intercal K^+ \boldsymbol{b}_{m}|
    &= |\boldsymbol{b}_l^\intercal K^+ K K^+ \boldsymbol{b}_{m}| \\
    &= |(P_W b_l, P_W b_m)_{\mathcal{V}}| \\
    &\le \|P_W b_l\|_{\mathcal{V}} \|P_W b_m\|_{\mathcal{V}} \\
    &\le \|b_l\|_{\mathcal{V}} \|b_m\|_{\mathcal{V}} \\
    &= 1 .
\end{aligned}
$$
Combining both equations, we obtain
$$
    \det(BK^+B^\intercal)
    \le \sum_{\sigma\in S_d} \prod_{l=1}^d 1
    = d!
$$
and, consequently,
$$
    \int_{\mathcal{X}^n} \det(BK^+B^\intercal) \,\mathrm{d}\rho^{\otimes n} \le d! < \infty.
$$


## Sampling
To draw a sample from the distribution described in the preceding section we let $n=d$ and conditioning on the event $\operatorname{ker}(K)=0$ (i.e. we assume that all sample points are distinct).
Under these assumptions, we can rewrite the determinant as
$$
    \det(BK^+ B^\intercal) = \frac{\det(B^\intercal B)}{\det(K)} .
$$
Now, assume that a sufficiently good rank-$R$ approximation of the Mercer decomposition
$$
    k(x, y) = \sum_{l=1}^\infty \sigma_l \phi_l(x) \phi_l(y) \approx \sum_{l=1}^R \psi_l(x)\psi_l(y)
$$
is given.
Then we can define $\Psi_{li} := \psi_l(x_i)$ and write
$$
    K_{ij} = k(x_i, x_j) = \sum_{l=1}^R \psi_l(x_i)\psi_l(x_j) = (\Psi^\intercal \Psi)_{ij} .
$$
Now recall that for any matrix $A = [a_1\ \tilde{A}_1]$, the Woodbury matrix identity implies
$$
    \det(A^\intercal A) = a_1^\intercal(I - \tilde{A}_1\tilde{A}_1^+)a_1 \cdot \det(\tilde{A}_1^\intercal \tilde{A}_1) .
$$
Recursive application of this relation yields
$$
    \det(BK^+ B^\intercal)
    = \frac{\det(B^\intercal B)}{\det(\Psi^\intercal\Psi)}
    = \prod_{i=1}^n \frac{\boldsymbol{b}_i^\intercal(I - \tilde{B}_i\tilde{B}_i^+) \boldsymbol{b}_i}{\boldsymbol{\psi}_i^\intercal (I - \tilde{\Psi}_i\tilde{\Psi}_i^+) \boldsymbol{\psi}_i}
    .
$$
Here $\tilde{B}_i \in \mathbb{R}^{d\times (n-i)}$ and $\tilde{\Psi}_i \in \mathbb{R}^{R\times (n-i)}$ are given by
$$
    (\tilde{B}_i)_{lj} := b_l(x_j)
    \qquad\text{and}\qquad
    (\tilde{\Psi}_i)_{lj} := \psi_l(x_j),
$$
respectively,
and $\boldsymbol{b}_i\in\mathbb{R}^d$ and $\boldsymbol{\psi}_i\in\mathbb{R}^R$ are defined by
$$
    (\boldsymbol{b}_i)_l := b_l(x_i)
    \qquad\text{and}\qquad
    (\boldsymbol{\psi}_i)_l := \psi_l(x_i),
$$
respectively.
Since $\tilde{B}_i$ and $\tilde\Psi_i$ only depend on the points $x_{j}$ for $j>i$, this demonstrates how we can recursively sample from the (unnormalised) density $\det(BK^+ B^\intercal)$ by sequentially sampling from the ratios
$$
    \frac{\boldsymbol{b}_i^\intercal(I - \tilde{B}_i\tilde{B}_i^+) \boldsymbol{b}_i}{\boldsymbol{\psi}_i^\intercal (I - \tilde{\Psi}_i\tilde{\Psi}_i^+) \boldsymbol{\psi}_i} .
$$

## Example

Consider the Hilbert space $\mathcal{V} = H^1([-1, 1])$ with reproducing kernel
$$
    k(x, y) \propto \cosh(1 - \max\{x, y\}) \cosh(1 + \min\{x, y\})
$$
and the $d$-dimensional polynomial subspace $\mathcal{V}_d = \operatorname{span}\{1, x, \ldots, x^{d-1}\}$.
Performing the sequential sampling procedure described above, we can see that the repulsive properties of the projection DPP $\det(B^\intercal B)$ being preserved in the ratio $\frac{\det(B^\intercal B)}{\det(\Psi^\intercal \Psi)}$.
![](plot/R-1000/sampling_density_step-1.png)
![](plot/R-1000/sampling_density_step-3.png)
![](plot/R-1000/sampling_density_step-7.png)
![](plot/R-1000/sampling_density_step-10.png)

However, the sampling becomes numerically unstable, due to the divsion by a function with zeros.
We therefore have to draw samples from a regularised density.
In particular, we also observe that the quality of the Mercer approximation $K \approx \Psi^\intercal \Psi$ influences the the quality of the resulting samples.
This is illstrated in the following three plots.
All experiments start the sampling with the same random seed and differ only in the rank ($R \in \{10, 100, 1000\}$).

![](plot/R-10/sample_statistics.png)
![](plot/R-100/sample_statistics.png)
![](plot/R-1000/sample_statistics.png)

## Greedy update

A greedy update of a given sample $x_1, \ldots, x_d$ could be done as follows

0. Initialize the joint sample $\underline{x} := (x_1, \ldots, x_d)$.
1. Draw a new sample $\tilde{x}_1, \ldots, \tilde{x}_d$.
2. Update the joint sample $\underline{x} := (\underline{x}, \tilde{x}_i)$ such that $\mu(\underline{x})$ is minimised.
3. If $\mu(\underline{x})$ is still too large, go to step 1.

This procedure is possible because $\mu(\underline{x})$ decreases monotonically with the sample size $n$.
Numerical evidence for this is provided in the subsequent plot.

![](plot/quasi-optimality_factor.png)

Indeed, this is easy to prove by observing that
$$
    \mu = \max_{v\in \mathcal{V}_d} \frac{\|v\|_{\mathcal{V}}}{\|P_Wv\|_{\mathcal{V}}},
$$
where $P_W$ is the $\mathcal{V}$-orthogonal projection onto the space $W := \operatorname{span}\{k(x_1,\bullet), \ldots, k(x_n, \bullet)\}$.
Adding a sample $x_{n+1}$ can only increase the space $W \subseteq \tilde{W} := W \oplus \operatorname{span}\{k(x_{n+1}, \bullet)\}$ and thereby increase the norm $\|P_{\tilde{W}}v\|_{\mathcal{V}} \ge \|P_{W}v\|_{\mathcal{V}}$ for every $v\in\mathcal{V}_d$.
This implies
$$
    \mu_W
    = \max_{v\in \mathcal{V}_d} \frac{\|v\|_{\mathcal{V}}}{\|P_Wv\|_{\mathcal{V}}}
    \ge \max_{v\in \mathcal{V}_d} \frac{\|v\|_{\mathcal{V}}}{\|P_{\tilde{W}}v\|_{\mathcal{V}}}
    = \mu_{\tilde{W}} .
$$

## Optimisation

Note, that by the preceding argument we can devise an evolutionary optimisation algorithm for the generation of an optimal sample.

0. Draw an initial sample $\underline{x}$ of size $M$.
1. Draw a new independent sample $\tilde{x}_1,\ldots,\tilde{x}_d$ of minimal size $d$.
2. For every $j=1,\ldots,d$ define $\underline{x}^{+j} := (\underline{x}, \tilde{x}_j)$.
3. Compute $j^\star = \argmin_{j=1,\ldots,d} \mu(\underline{x}^{+j})$ and redefine $\underline{x} := \underline{x}^{+j^\star}$.
4. For every $j=1,\ldots,M+1$ define $\underline{x}^{-j}$ as $\underline{x}$ without the $j$-th point.
5. Compute $j^\star = \argmin_{j=1,\ldots,d} \mu(\underline{x}^{-j})$ and redefine $\underline{x} := \underline{x}^{-j^\star}$.
6. Go to 1. or terminate.

Note that, we only add or remove a single sample point, the monotonicity of $\mu$ guarantees that the value of $\mu$ decreases in every step of this procedure.
However, the resulting sample remains random and may get stuck in a local minimum.
This is illustrated in the subsequent plot.

![](plot/optimisation_statistics.png)

> **Note:** From a theoretical perspective, it would be interesting to know the minimal value of $\mu$.
> To answer this question (at least partially), observe that the maximal value that we can achieve for $\lambda_{\mathrm{min}}(BK^+B^\intercal)$ is given by
> $$
> \begin{aligned}
>     \max_{\substack{W\subseteq\mathcal{V}\\\dim(W) = n}} \min_{\substack{v\in\mathcal{V}_d\\\|v\|_{\mathcal{V}}=1}} \|P_Wv\|_{\mathcal{V}}^2
>     &= \max_{\substack{W\subseteq\mathcal{V}\\\dim(W) = n}} \min_{\substack{v\in\mathcal{V}_d\\\|v\|_{\mathcal{V}}=1}} (1 - \|(I - P_W)v\|_{\mathcal{V}}^2) \\
>     &= \max_{\substack{W\subseteq\mathcal{V}\\\dim(W) = n}} \left(1 - \max_{\substack{v\in\mathcal{V}_d\\\|v\|_{\mathcal{V}}=1}} \|(I - P_W)v\|_{\mathcal{V}}^2 \right) \\
>     &= \max_{\substack{W\subseteq\mathcal{V}\\\dim(W) = n}} \left(1 - \|(I - P_W)P_{\mathcal{V}_d}\|^2 \right) \\
>     &\overset?= \max_{\substack{W\subseteq\mathcal{V}\\\dim(W) = n}} \left(\|P_{\mathcal{V}_d}\|^2 - \|(I - P_W)P_{\mathcal{V}_d}\|^2 \right) \\
>     &\overset?= \max_{\substack{W\subseteq\mathcal{V}\\\dim(W) = n}} \|P_WP_{\mathcal{V}_d}\|^2 .
> \end{aligned}
> $$
> This is an approximation theoretic constant that measures how well the functions in $\mathcal{V}_d$ can be approximated by a linear space of the form $W$.

> **Note:** Note, however, that knowledge of this constant is not very relevant in practice.
In practice, we want to obtain a sample for which $\mu$ satisfies some pre-specified bound $\mu\le\mu_0$ for some $\mu_0 > 0$.

To achieve this goal, we propose the following algorithm.

1. Step

   0. Initialise the accumulate sample $\bar{\underline{x}}$.
   1. Draw $\underline{x}$ according to the algorithm above uniformly at random.
   2. Append $\underline{x}$ to the accumulate sample $\bar{\underline{x}}$.
   3. While $\mu > \mu_0$, go to 1.

2. Step

    0. Choose an oversampling factor $\gamma\in\mathbb{N}$.
    1. Repeat the above procedure $\gamma$ times.
    2. Combine the samples into a single accumulate sample $\bar{\underline{x}}$.

3. Step

    1. While $\mu \le \mu_0$, remove the sample which least increases $\mu$.

> **Note:** It would be interesting if we could sample directly from the distribution with a larger sample size. But I think this is not an easy task.

The subsequent figure demonstrates the sample distribution as well as the obtained quasi-optimality factors and sample sizes for this algorithm with $\mu_0 = 2$ and $\beta=1$.

![](plot/optimisation_statistics_sample_size.png)

To demonstrate how this algorithm behaves in practice, we consider again the example from above and approximate the function
$$
    u(x) := \sin(2 \pi x) + \cos(2 \pi d x)
$$
in a space of polynomials of degree $g \le 9$.
We then compute the quasi-optimality factor
$$
    \hat\mu \approx \mu \ge \frac{1}{2} \left(\frac{\|u_\star - u_{d,n}\|_{\mathcal{V}}}{\|u_\star - u_d\|_{\mathcal{V}}} - 1 \right)
$$
empirically.
The subsequent plot illustrates that this factor is indeed well below the chosen threshold $\mu_0 = 2$.

![](plot/optimal_sampled_least_squares_wave.png)

The same values for the function
$$
    u(x) := \exp(x)
$$
are plotted in the subsequent graph.
![](plot/optimal_sampled_least_squares_exp.png)

Finally, we plot the same values for the function
$$
    u(x) := \sum_{k=0}^{30} b_k .
$$
![](plot/optimal_sampled_least_squares_ones.png)