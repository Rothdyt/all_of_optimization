<div align=center>
 <span style="color:Orange;font-size:2em;font-weight: bold;font-family:Courier New;">
  From Gauss-Elimination to LU Factorization
 </span>
</div>
# Example 

Given the following matrix $A$, how to perform the Gauss Elimination on it to get an upper triangular matrix $U$?

$$
A=\begin{bmatrix}
	{2} & {1} & {1} \\ {4} & {-6} & {0} \\ {-2} & {7} & {2}
\end{bmatrix}
$$

```txt
* Add -2 times the first row to the second row;
* Add 1 times the first row to the third row;
* Add 1 times the second row to the third row.
```
The above procedure can be summarized below.
$$
\underbrace{\begin{bmatrix}
	{1} & {} & {} \\ {} & {1} & {}\\ {} & {\bf{1}} & {1}
\end{bmatrix}}_{E3}
\underbrace{\begin{bmatrix}
	{1} & {} & {} \\ {} & {1} & {}\\ {1} & {} & {\bf{1}}
\end{bmatrix}}_{E2}
\underbrace{\begin{bmatrix}
	{1} & {} & {} \\ {\bf{-2}} & {1} & {}\\ {} & {} & {1}
\end{bmatrix}}_{E1}
\underbrace{\begin{bmatrix}
	{2} & {1} & {1} \\ {4} & {-6} & {0} \\ {-2} & {7} & {2}
\end{bmatrix}}_{A} = 
\underbrace{\begin{bmatrix}
	{2} & {1} & {1} \\ {0} & {-8} & {-2} \\ {0} & {0} & {1}
\end{bmatrix}}_{U}
$$
So $L$ can be computed by $(E_3E_2E_1)^{-1}$.  A much more clever way to derive $L$ is based on the fact that we can use $L$ to transform $U$ back to $A$, which can be done by inverting the above procedure. So 
$$
L=\begin{bmatrix}
	{1} & {0} & {0} \\ {2} & {1} & {0} \\ {-1} & {-1} & {1}
\end{bmatrix}.
$$
The Gaussian Elimination takes $\mathcal{O}(n^3)$ operations[^1].



## LU Factorization

In general, for $A\in R^{n\times n}, \exists$ a permutation matrix[^ 2] $P$ such that $PA=LU$, where $L$ and $U$ are a lower triangular matrix with all diagonal elements equal to $1$ and an upper triangular matrix respectively.



[^1]: Eliminating the first row will require 𝑛 additions and 𝑛 multiplications for $𝑛−1$ rows. Therefore, the number of operations for the first column is $2𝑛(𝑛−1)$. For the second row, similarly, we have $2(𝑛−1)(𝑛−2$) operations. Therefore, the total number of operations required    is $\sum_{i=1}^n2(𝑛−𝑖)(𝑛−𝑖+1)\xlongequal{j=n-i+1} = 2\sum_{j=0}^{n-1}(j^2 + j)=2(1/3n^3 - 1/3n^2)$
[^2]:  In each column and row, only has one position set to $1$ while others are set to $0$.

