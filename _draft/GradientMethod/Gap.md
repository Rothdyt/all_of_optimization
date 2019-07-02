## CD, R-CD and C-CD

Consider following optimization problem
$$
\min _{x \in \mathbb{R}^{n}} f(x) \triangleq x^{T} Q x,
$$
where $Q$ is positive defined. Denote $\lambda_{\text{max}}$,  $\lambda_{\text{min}}$, and  $\lambda_{\text{avg}}$ as its largest, minimal and average eigenvalues.

### Complexity for CD

Clearly, our objective function $f(x)$'s gradient is $L$-Lipschitz continuous. So consider the update scheme
$$
x^{t+1} \leftarrow x^t - \frac{1}{L}\nabla f(x^t),
$$
where $L$ is set to  $\lambda_{\text{max}}$, the largest eigenvalue of $Q$. 

The Lipschitz continuity implies
$$
\begin{align*}
f(y) \leq f(x)+\langle\nabla f(x), y-x\rangle+\frac{L}{2}\|y-x\|^{2}\quad \forall x,y. \tag{1}
\end{align*}
$$

[Polyak-Lojasiewicz condition] A function satisfies the PL inequality if the following holds for some $\mu>0$,
$$
\frac{1}{2}\|\nabla g(x)\|^{2} \geq \mu\left(g(x)-g^{*}\right) \quad \forall x,
$$
where $g^*$ is the global minimal. If set $\mu = \lambda_{\text{min}}$, where $\lambda_{\text{min}}$ is minimal eigen-value of $Q$,  then $f(x)$ satisfy the PL inequality.

Plug the update scheme into (1), we obtain
$$
\begin{align*}
f\left(x^{t+1}\right)-f\left(x^{t}\right) \leq-\frac{1}{2 L}\left\|\nabla f\left(x^{t}\right)\right\|^{2}. \tag{2}
\end{align*}
$$
Combie (2) with PL inequality, we derive 
$$
f\left(x^{t+1}\right)-f^{*} \leq\left(1-\frac{\lambda_{\text{min}}}{\lambda_{\text{max}}}\right)\left(f\left(x^{t}\right)-f^{*}\right),
$$
which implies 
$$
\begin{align*}
f\left(x^{t}\right)-f^{*} \leq\left(1-\frac{1}{\kappa}\right)^t\left(f\left(x^{0}\right)-f^{*}\right),\tag{3}
\end{align*}
$$
where $\kappa = \frac{\lambda_{\text{max}}}{\lambda_{\text{min}}}$.  (3) implies linear convergence rate.

To obtain $\epsilon$-accuracy, i.e., $\frac{f\left(x^{t}\right)-f^{*}}{f\left(x^{0}\right)-f^{*}}\leq \epsilon$, we need at least $\frac{\log\frac{1}{\epsilon}}{\log\frac{1}{1-1/\kappa}}\approx \kappa\log\frac{1}{\epsilon}$ iterations. And in each iteration, we need calculate $\nabla f(x)=Qx$, which takes $\mathcal{O}(n^2)$ operations. So the **time complexity** is $\mathcal{O}(n^2\kappa\log\frac{1}{\epsilon})$.

### Complexity for R-CD

For general convex function,  we assume the coordinate-wise gradient Lipschitiz continuity, i.e.,
$$
\begin{align*}
\|\nabla_i g(x+\mathbf{e_{i}}t_i) - \nabla_i g(x)\|\leq L_i\|t\|, \tag{4}
\end{align*}
$$
where $\mathbf{e}_{i}$ is unit vector with $i$'s coordiante  set to $1$ and $\nabla_{i} g(x)$ is the $i$-th component of $\nabla g(x)$, or equivalently $\frac{\partial g(x)}{\partial x_{i}}$.

Then R-CD update scheme [Nesterov, 2012] is defined as,

---

a. Initialize $x_0\in R^n$.

b. At Itearation $t$,

​	1. Choose the index $i_t\in \{1,2,\cdots, n\}$ with probability 
$$
P(i_t=j)= \frac{L_j}{\sum_{j=1}^n L_j}\overset{\Delta}{=}p_j.
$$
​	2. Update $i_t$'s coordinate with 
$$
x^{t+1} \leftarrow x^t - \frac{1}{L_{i_t}}\mathbf{e}_{i_t}\nabla_{i_t} g(x).
$$
c. Terminate until convergence.

---



Define $\phi_t:= E_{\{i_0,\cdots,  i_t-1\}} [g(x^t)]$. Then
$$
\begin{align*}
g(x^t) - E_{i_t} [g(x^{t+1})] 
& = \sum_{i=1}^n p_i [g(x^t) - g(x^t-\frac{1}{L_i}\nabla_i g(x^t)\mathbf{e_i})] \\
& \overset{(4)}{\geq}\sum_{i=1}^n \frac{p_i}{2L_i}\|\nabla_i  g(x^t)\|^2\\
&= \frac{1}{2\sum_{i=1}^n L_i}\|\nabla g(x^t)\|^2. \tag{5}
\end{align*}
$$
For our problem, we have following observations,

1. $L_i$ can be set to $Q_{ii}$, the $i$-th diagonal entry of $Q$. Then $L_{\text{avg}}=\frac{1}{n}\sum_{i=1}^n L_i = \lambda_{\text{avg}}.$

2. $f(x)$ is strongly convex with parameter $\sigma=\lambda_{\min}$, i.e.,
   $$
   f(y)\geq f(x) + (y-x)^T\nabla(x) + \frac{1}{2}\sigma \|y-x\|^2 \quad \forall x, y\in R^n.
   $$
   

If we minimize $h(y)=(y-x)^T\nabla(x) + \frac{1}{2}\sigma \|y-x\|^2$ with respect to $y$, then we obtain $h_{\text{min}}=h(x-\frac{1}{\sigma}\nabla f(x))=-\frac{1}{2\sigma}\|\nabla f(x)\|^2$.  Suppose $y^*$ is the minimizer of $f(y)$, then by the strong convexity,  we have 
$$
\begin{align*}
f(x) - f(y^*) \leq -h(y^*) \leq - h_{\text{min}} = \frac{1}{2\sigma}\|\nabla f(x)\|^2 \tag{6}
\end{align*}
$$

So substitute $g$ with $f$ and plug (6) into (5), we further obtain
$$
\begin{align*}
f(x^t) - E_{i_t} [f(x^{t+1})] 
&\geq \frac{1}{2\sum_{i=1}^n L_i}\|\nabla f(x^t)\|^2\\
&\geq  \frac{\sigma}{\sum_{i=1}^n L_i} (f(x^t) - f^*).
\end{align*}
$$
Take expectation on both side with resepect to $\{i_0,\cdots,  i_t-1\}$, and rearrange terms, then we obtain
$$
\phi^{t+1} - f^* \leq (1 - \frac{\sigma}{\sum_{i=1}^n L_i}) (\phi^t - f^*) = (1 - \frac{1}{n\lambda_{\text{avg}}/ \lambda_{\text{min}}}) (\phi^t - f^*),
$$
which implies
$$
\begin{align*}
\phi^t-f^{*} \leq(1 - \frac{1}{n\lambda_{\text{avg}}/ \lambda_{\text{min}}}) ^t\left(f\left(x^{0}\right)-f^{*}\right).
\end{align*}
$$
To make the comparison with CD fair, we need consider the error sequence $\{\phi^0-f*, \cdots, \phi^{(n-1)t}-f*, \phi^{nt}-f*, \cdots\}$. Then $\epsilon$-accuracy is defined as $\frac{\phi\left(x^{nt}\right)-f^{*}}{f\left(x^{0}\right)-f^{*}}\leq \epsilon$, we need at least $\frac{\log\frac{1}{\epsilon}}{\log\frac{1}{1-1/n\kappa_{CD}}}\approx \kappa_{CD}\log\frac{1}{\epsilon}$ n-randmdom-iterations. 

In each n-random-iteration, we need calculate $\nabla_i f(x)=\sum_{j=1}^nx_j Q_{ij}$ n-times, which takes $\mathcal{O}(n^2)$ operations.  So the **time complexity** is $\mathcal{O}(n^2\kappa_{CD}\log\frac{1}{\epsilon})$, where $\kappa_{CD}={\lambda_{\text{avg}}/ \lambda_{\text{min}}}$.

