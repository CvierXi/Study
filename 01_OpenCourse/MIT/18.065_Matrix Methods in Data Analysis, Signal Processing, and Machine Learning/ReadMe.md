<!-- omit in toc -->
# Matrix Methods in Data Analysis, Signal Processing, and Machine Learning

![图片](./images/Relationship.jpg)

<!-- omit in toc -->
## 课程信息

- 资源: [MIT 18.065](https://ocw.mit.edu/courses/mathematics/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/index.htm
)
- 教师: [Prof. Gilbert Strang](http://math.mit.edu/~gs/)
- 视频: [YouTube](https://www.youtube.com/watch?v=t36jZG07MYc)
- 教材: [教材](http://math.mit.edu/~gs/learningfromdata/)
- 习题: [习题](Assignments%20problem%20sets/MIT18_065S18PSets.pdf)
- 答案: [答案](Assignments%20problem%20sets/Solutions%20to%20Exercises.pdf)
- 介绍:
  - Linear algebra concepts are key for understanding and creating machine learning algorithms, especially as applied to deep learning and neural networks. This course reviews linear algebra with applications to probability and statistics and optimization–and above all a full explanation of deep learning.
- 参考书:
  - [Trefethen, Bau - Numerical Linear Algebra](https://github.com/CvierXi/Study/blob/master/01_OpenCourse/MIT/18.065_Matrix%20Methods%20in%20Data%20Analysis%2C%20Signal%20Processing%2C%20and%20Machine%20Learning/Reference%20Books/Trefethen%2C%20Bau%20-%20Numerical%20Linear%20Algebra.pdf)
  - [Golub, VanLoan - Matrix Computations](https://github.com/CvierXi/Study/blob/master/01_OpenCourse/MIT/18.065_Matrix%20Methods%20in%20Data%20Analysis%2C%20Signal%20Processing%2C%20and%20Machine%20Learning/Reference%20Books/Golub%2C%20VanLoan%20-%20Matrix%20Computations.pdf)
- 其他:
  - Chrome浏览器需要安装[MathJax Plugin for Github](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima/related)插件以正确显示公式.  

<!-- omit in toc -->
## 视频课程

- [00.1_课程简介](#001_课程简介)
- [00.2_采访](#002_采访)
- [01_矩阵列空间](#01_矩阵列空间)
- [02_矩阵相乘和分解](#02_矩阵相乘和分解)
- [03_正交矩阵](#03_正交矩阵)
- [04_特征值和特征向量](#04_特征值和特征向量)
- [05_正定和半正定矩阵](#05_正定和半正定矩阵)
- [06_奇异值分解SVD](#06_奇异值分解svd)
- [07_A的最邻近秩k矩阵](#07_a的最邻近秩k矩阵)
- [08_向量和矩阵的范数](#08_向量和矩阵的范数)
- [09_最小二乘法的四种解法](#09_最小二乘法的四种解法)
- [10_Ax=b](#10_axb)
- [11_最小化||x||，使得Ax=b](#11_最小化x使得axb)
- [12_计算特征值和奇异值](#12_计算特征值和奇异值)
- [13_矩阵相乘的随机抽样](#13_矩阵相乘的随机抽样)
- [14_发生低秩改变后的矩阵A及其逆](#14_发生低秩改变后的矩阵a及其逆)
- [15_A(t)的特征值关于t的导数](#15_at的特征值关于t的导数)
- [16_A(t)的逆和奇异值关于t的导数](#16_at的逆和奇异值关于t的导数)
- [17_Prof Alex: 数值计算中的低秩矩阵](#17_prof-alex-数值计算中的低秩矩阵)
- [18_SVD, LU, QR分解的自由度计算，鞍点](#18_svd-lu-qr分解的自由度计算鞍点)
- [19_鞍点](#19_鞍点)
- [20_均值，方差，协方差](#20_均值方差协方差)
- [21_凸优化](#21_凸优化)
- [22_梯度下降法](#22_梯度下降法)
- [23_加速梯度下降法](#23_加速梯度下降法)
- [24_线性规划](#24_线性规划)
- [25_随机梯度下降法](#25_随机梯度下降法)
- [26_神经网络结构](#26_神经网络结构)
- [27_反向传播](#27_反向传播)
- [30_循环矩阵](#30_循环矩阵)
- [31_循环矩阵的特征向量，Fourier矩阵](#31_循环矩阵的特征向量fourier矩阵)

### 00.1_课程简介

- <https://www.bilibili.com/video/BV1dg4y187Y8?p=1>

### 00.2_采访

- <https://www.bilibili.com/video/BV1dg4y187Y8?p=2>

### 01_矩阵列空间

- <https://www.bilibili.com/video/BV1dg4y187Y8?p=3>
- Description
  - In this first lecture, Professor Strang introduces the linear algebra principles critical for understanding the content of the course.  In particular, matrix-vector multiplication $Ax$ and the column space of a matrix and the rank.
- Summary
  - Independent columns = basis for the column space
  - Rank = number of independent columns
  - $A=CR$ leads to: Row rank equals column rank

### 02_矩阵相乘和分解

- <https://www.bilibili.com/video/BV1dg4y187Y8?p=4>
- Description
  - Multiplying and factoring matrices are the topics of this lecture. Professor Strang reviews multiplying columns by rows:  $AB=$ sum of rank one matrices. He also introduces the five most important factorizations.
- Summary
  - Multiplying columns by rows:  $AB=$ sum of rank one matrices
  - Five great factorizations:
    - $A=LU$ from elimination
    - $A=QR$ from orthogonalization (Gram-Schmidt)
    - $S=Q\Lambda Q^T$ from eigenvectors of a symmetric matrix $S$
    - $A=X\Lambda X^{-1}$ diagonalizes $A$ by the eigenvector matrix $X$
    - $A=U\Sigma V^T=$ (orthogonal)(diagonal)(orthogonal) = Singular Value Decomposition
- Related section in textbook: I.2
  - $A=CR$ leads to: Row rank equals column rank

### 03_正交矩阵

- <https://www.bilibili.com/video/BV1dg4y187Y8?p=5>
- Description
  - This lecture focuses on orthogonal matrices and subspaces. Professor Strang reviews the four fundamental subspaces: column space $C(A)$, row space $C(A^T)$, nullspace $N(A)$, left nullspace $N(A^T)$.
- Summary
  - Rotations
  - Reflections
  - Hadamard matrices
  - Haar wavelets
  - Discrete Fourier Transform (DFT)
  - Complex inner product
- Related section in textbook: I.5

### 04_特征值和特征向量

- <https://www.bilibili.com/video/BV1dg4y187Y8?p=6>
- Description
  - Professor Strang begins this lecture talking about eigenvectors and eigenvalues and why they are useful. Then he moves to a discussion of symmetric matrices, in particular, positive definite matrices.
- Summary
  - $Ax=\lambda x$
  - $A^2x=\lambda^2 x$
  - Write other vectors as combinations of eigenvectors
  - Similar matrix $B=M^{-1}AM$ has the same eigenvalues as $A$
- Related section in textbook: I.6

### 05_正定和半正定矩阵

- <https://www.bilibili.com/video/BV1dg4y187Y8?p=7>
- Description
  - In this lecture, Professor Strang continues reviewing key matrices, such as positive definite and semidefinite matrices. This lecture concludes his review of the highlights of linear algebra.
- Summary
  - All $\lambda_i>0$
  - Energy $x^TSx>0$
  - $S=A^TA$ (independent cols in $A$)
  - All leading determinants $>0$
  - All pivots in elimination $>0$
- Related section in textbook: I.7

### 06_奇异值分解SVD

- <https://www.bilibili.com/video/BV1dg4y187Y8?p=8>
- Description
  - Singular Value Decomposition (SVD) is the primary topic of this lecture. Professor Strang explains and illustrates how the SVD separates a matrix into rank one pieces, and that those pieces come in order of importance.
- Summary
  - $A=U\Sigma V^T$
  - Columns of $V$ are orthonormal eigenvectors of $A^TA$.
  - $Av=\sigma u$ gives orthonormal eigenvectors $u$ of $AA^T$.
  - $\sigma^2=$ eigenvalue of $A^TA$ = eigenvalue of $AA^T\neq0$
  - $A$ = (rotation)(stretching)(rotation) $U\Sigma V^T$ for every $A$
- Related section in textbook: I.8

### 07_A的最邻近秩k矩阵

- <https://www.bilibili.com/video/BV1dg4y187Y8?p=9>
- Description
  - A norm is a way to measure the size of a vector, a matrix, a tensor, or a function. Professor Strang reviews a variety of norms that are important to understand including S-norms, the nuclear norm, and the Frobenius norm.
- Summary
  - $A_k=\sigma_1u_1v^T_1+\cdots+\sigma_ku_kv^T_k$
  - $\lVert A-B_k \rVert \geq \lVert A-A_k \rVert$
  - Norms
    - ${\lVert v \rVert}_1 = \lvert v_1 \rvert + \cdots + \lvert v_n \rvert$
    - ${\lVert v \rVert}_2 = \sqrt{{\lvert v_1 \rvert}^2 + \cdots + {\lvert v_1 \rvert}^2}$
    - ${\lVert v \rVert}_{\infty} = \max {\lvert v_i \rvert}$
    - ${\lVert A \rVert}_{Nuclear} = \sigma_1 + \cdots + \sigma_r$
    - ${\lVert A \rVert}_{Frobenius} = \sqrt{{\lvert a_{11} \rvert}^2 + {\lvert a_{12} \rvert}^2 + \cdots + {\lvert a_{mn} \rvert}^2}$
    - ${\lVert A \rVert}_2 = \sigma_1$
  - The idea of Principal Component Analysis (PCA)
- Related section in textbook: I.9

### 08_向量和矩阵的范数

- <https://www.bilibili.com/video/BV1dg4y187Y8?p=10>
- Description
  - In this lecture, Professor Strang reviews Principal Component Analysis (PCA), which is a major tool in understanding a matrix of data. In particular, he focuses on the Eckart-Young low rank approximation theorem.
- Summary
  - ${\lVert v \rVert}_1 = \lvert v_1 \rvert + \cdots + \lvert v_n \rvert$
  - ${\lVert v \rVert}_2 = \sqrt{{\lvert v_1 \rvert}^2 + \cdots + {\lvert v_1 \rvert}^2}$
  - ${\lVert v \rVert}_{\infty} = \max {\lvert v_i \rvert}$
  - ${\lVert A \rVert}_{Nuclear} = \sigma_1 + \cdots + \sigma_r$
  - ${\lVert A \rVert}_{Frobenius} = \sqrt{{\lvert a_{11} \rvert}^2 + {\lvert a_{12} \rvert}^2 + \cdots + {\lvert a_{mn} \rvert}^2}$
  - ${\lVert A \rVert}_2 = \sigma_1$
- Related section in textbook: I.11

### 09_最小二乘法的四种解法

- <https://www.bilibili.com/video/BV1dg4y187Y8?p=11>
- Description
  - In this lecture, Professor Strang details the four ways to solve least-squares problems. Solving least-squares problems comes in to play in the many applications that rely on data fitting.
- Summary
  - Solve $A^TA\hat{x}=A^Tb$ to minimize $\lVert Ax-b \rVert ^2$
  - Gram-Schmidt $A=QR$ leads to $x=R^{-1}Q^Tb$
  - The pseudoinverse directly multiplies $b$ to give $x$
  - The best $x$ is the limit of $(A^TA+\delta I)^{-1}A^Tb$ as $\delta \rightarrow 0$.
- Related section in textbook: II.2

### 10_Ax=b

- <https://www.bilibili.com/video/BV1dg4y187Y8?p=12>
- Description
  - The subject of this lecture is the matrix equation $Ax=b$. Solving for $x$ presents a number of challenges that must be addressed when doing computations with large matrices.
- Summary
  - Large condition number $\lVert A \rVert$ $\lVert A^{-1} \rVert$
  - $A$ is ill-conditioned and small errors are amplified.
  - Penalty method regularizes a singular problem.
- Related chapter in textbook: Introduction to Chapter II

### 11_最小化||x||，使得Ax=b

- <https://www.bilibili.com/video/BV1dg4y187Y8?p=13>
- Description
  - In this lecture, Professor Strang revisits the ways to solve least squares problems. In particular, he focuses on the Gram-Schmidt process that finds orthogonal vectors.
- Summary
  - Picture the shortest $x$ in $\ell^1$ and $\ell^2$ and $\ell^{\infty}$ norms
  - The $\ell^1$ norm gives a sparse solution $x$.
  - Details of Gram-Schmidt orthogonalization and $A=QR$
  - Orthogonal vectors in $Q$ from independent vectors in $A$
- Related chapter in textbook: Chapter I.11

### 12_计算特征值和奇异值

- <https://www.bilibili.com/video/BV1dg4y187Y8?p=14>
- Description
  - Numerical linear algebra is the subject of this lecture and, in particular, how to compute eigenvalues and singular values. This includes discussion of the Hessenberg matrix, a square matrix that is almost (except for one extra diagonal) triangular.
- Summary
  - $QR$ method for eigenvalues: Reverse $A=QR$ to $A_1=RQ$
  - Then reverse $A_1=Q_1R_1$ to $A_2=R_1Q_1$: Include shifts
  - $A$'s become triangular with eigenvalues on the diagonal.
  - Krylov spaces and Krylov iterations
- Related chapter in textbook: Chapter II.1

### 13_矩阵相乘的随机抽样

- <https://www.bilibili.com/video/BV1dg4y187Y8?p=15>
- Description
  - This lecture focuses on randomized linear algebra, specifically on randomized matrix multiplication. This process is useful when working with very large matrices. Professor Strang introduces and describes the basic steps of randomized computations.
- Summary
  - Sample a few columns of $A$ and rows of $B$
  - Use probabilities proportional to lengths $\lVert A_i \rVert$ $\lVert B_i \rVert$
  - See the key ideas of probability: Mean and Variance
  - Mean $=AB$ (correct) and variance to be minimized
- Related chapter in textbook: Chapter II.4

### 14_发生低秩改变后的矩阵A及其逆

- <https://www.bilibili.com/video/BV1dg4y187Y8?p=16>
- Description
  - In this lecture, Professor Strang introduces the concept of low rank matrices. He demonstrates how using the Sherman-Morrison-Woodbury formula is useful to efficiently compute how small changes in a matrix affect its inverse.
- Summary
  - If $A$ is changed by a rank-one matrix, so is its inverse.
  - Woodbury-Morrison formula for those changes
  - New data in least squares will produce these changes.
  - Avoid recomputing over again with all data
  - Note: Formula in class is correct in the textbook.
- Related chapter in textbook: Chapter III.1

### 15_A(t)的特征值关于t的导数

- <https://www.bilibili.com/video/BV1dg4y187Y8?p=17>
- Description
  - This lecture is about changes in eigenvalues and changes in singular values. When matrices move, their inverses, their eigenvalues, and their singular values change. Professor Strang explores the resulting formulas.
- Summary
  - Matrices $A$ depending on $t$ /Derivative $=dA/dt$
  - $\cfrac{d\lambda}{dt}=y^T\cfrac{dA}{dt}x$, $x$ = eigenvector, $y$ = eigenvector of transpose of $A$
  - Eigenvalues from adding rank-one matrix are interlaced.
- Related chapter in textbook: Chapter III.1-2

### 16_A(t)的逆和奇异值关于t的导数

- <https://www.bilibili.com/video/BV1dg4y187Y8?p=18>
- Description
  - In this lecture, Professor Strang reviews how to find the derivatives of inverse and singular values. Later in the lecture, he discusses LASSO optimization, the nuclear norm, matrix completion, and compressed sensing.
- Summary
  - $\cfrac{dA^2}{dt}=A\cfrac{dA}{dt} + \cfrac{dA}{dt}A$, NOT $2A\cfrac{dA}{dt}$
  - $\cfrac{dA^{-1}}{dt}=-A^{-1}\cfrac{dA}{dt}A^{-1}$
  - $\cfrac{d\sigma}{dt}=u^T\cfrac{dA}{dt}v$
  - Interlacing of eigenvalues / Weyl inequalities
- Related chapter in textbook: Chapter III.1-2

### 17_Prof Alex: 数值计算中的低秩矩阵

- <https://www.bilibili.com/video/BV1dg4y187Y8?p=19>
- Description
  - Professor Alex Townsend gives this guest lecture answering the question “Why are there so many low rank matrices that appear in computational math?” Working effectively with low rank matrices is critical in image compression applications.
- Summary
  - Professor Alex Townsend's lecture
  - Why do so many matrices have low effective rank?
  - Sylvester test for rapid decay of singular values
  - Image compression: Rank $k$ needs only $2kn$ numbers.
  - Flags give many examples / diagonal lines give high rank.
- Related chapter in textbook: Chapter III.3

### 18_SVD, LU, QR分解的自由度计算，鞍点

- <https://www.bilibili.com/video/BV1dg4y187Y8?p=20>
- Description
  - In this lecture, Professor Strang reviews counting the free parameters in a variety of key matrices. He then moves on to finding saddle points from constraints and Lagrange multipliers.
- Summary
  - Find $n^2$ parameters in $L$ and $U$, $Q$ and $R$, ...
  - Find $(m+n-r)r$ parameters in a matrix of rank $r$
  - Find saddle points from constraints and Lagrange multipliers
- Related chapter in textbook: Chapter III.2

### 19_鞍点

- <https://www.bilibili.com/video/BV1dg4y187Y8?p=21>
- Description
  - Professor Strang continues his discussion of saddle points, which are critical for deep learning applications. Later in the lecture, he reviews the Maxmin Principle, a decision rule used in probability and statistics to optimize outcomes.
- Summary
  - $\cfrac{x^TSx}{x^Tx}$ has a saddle at eigenvalues between lowest / highest.
  - (Max over all $k$-dim spaces) of (Min of $\cfrac{x^TSx}{x^Tx}$) = evalue
  - Sample mean and expected mean
  - Sample variance and $k^{th}$ eigenvalue variance
- Related chapter in textbook: Chapter III.2 and V.1

### 20_均值，方差，协方差

- <https://www.bilibili.com/video/BV1dg4y187Y8?p=22>
- Description
  - This lecture continues the focus on probability, which is critical for working with large sets of data. Topics include sample mean, expected mean, sample variance, covariance matrices, Chebyshev's inequality, and Markov's inequality.
- Summary
  - $E[x]=m=$ average outcome weighted by probabilities
  - $E$ uses expected outcomes not actual sample outcomes.
  - $E[(x-m)^2]=E[x^2]-m^2$ is the variance of $x$.
  - Markov's inequality $Prob[x\geq a] \leq \cfrac{\bar{X}}{a}$ (when all $x$'s $\geq$ 0)
  - Chebyshev's inequality $Prob[|x-m|\geq a] \leq \cfrac{\sigma^2}{a^2}$
- Related chapter in textbook: Chapter V.1, V.3

### 21_凸优化

- <https://www.bilibili.com/video/BV1dg4y187Y8?p=23>
- Description
  - In this lecture, Professor Strang discusses optimization, the fundamental algorithm that goes into deep learning. Later in the lecture he reviews the structure of convolutional neural networks (CNN) used in analyzing visual imagery.
- Summary
  - Three terms of a Taylor series of $F(x)$: many variables $x$
  - Downhill direction decided by first partial derivatives of $F$ at $x$
  - Newton's method uses higher derivatives (Hessian at higher cost).
- Related chapter in textbook: Chapter VI.1, VI.4

### 22_梯度下降法

- <https://www.bilibili.com/video/BV1dg4y187Y8?p=24>
- Description
  - Gradient descent is the most common optimization algorithm in deep learning and machine learning. It only takes into account the first derivative when performing updates on parameters—the stepwise process that moves downhill to reach a local minimum.
- Summary
  - Gradient descent: Downhill from \(x\) to new \(X = x - s (\partial F / \partial x)\)
  - Excellent example: \(F(x,y) = \frac{1}{2} (x^2 + by^2)\)
  - If \(b\) is small we take a zig-zag path toward (0, 0).
  - Each step multiplies by \((b - 1)/(b + 1)\)
  - Remarkable function: logarithm of determinant of \(X\)
- Related chapter in textbook: Chapter VI.4
  
### 23_加速梯度下降法

- <https://www.bilibili.com/video/BV1dg4y187Y8?p=25>
- Description
  - In this lecture, Professor Strang explains both momentum-based gradient descent and Nesterov's accelerated gradient descent.
- Summary
  - Study the zig-zag example: Minimize $F=\cfrac{1}{2}(x^2 + by^2)$
  - Add a momentum term / heavy ball remembers its directions.
  - New point $k+1$ comes from TWO old points $k$ and $k-1$.
  - "1st order" becomes "2nd order" or "1st order system" as in ODEs.
  - Convergence rate improves: $1-b$ to $1-\sqrt{b}$!
- Related chapter in textbook: Chapter VI.4

### 24_线性规划

- <https://www.bilibili.com/video/BV1dg4y187Y8?p=26>
- Description
  - This lecture focuses on several topics that are specific parts of optimization. These include linear programming (LP), the max-flow min-cut theorem, two-person zero-sum games, and duality.
- Summary
  - **Linear program**: Minimize cost subject to $Ax=b$ and $x>0$
  - Inequalities make the problem piecewise linear.
  - Simplex method reduces cost from corner point to corner point.
  - Dual linear program is a maximization: Max = Min!
  - **Game**: $X$ chooses rows of payoff matrix, $Y$ chooses columns.
- Related chapter in textbook: Chapter VI.2–VI.3
  
### 25_随机梯度下降法

- <https://www.bilibili.com/video/BV1dg4y187Y8?p=27>
- Description
  - Professor Suvrit Sra gives this guest lecture on stochastic gradient descent (SGD), which randomly selects a minibatch of data at each step. The SGD is still the primary method for training large-scale machine learning systems.
- Summary
  - Full gradient descent uses all data in each step.
  - Stochastic method uses a minibatch of data (often 1 sample!).
  - Each step is much faster and the descent starts well.
  - Later the points bounce around / time to stop!
  - This method is the favorite for weights in deep learning.
- Related chapter in textbook: Chapter VI.6

### 26_神经网络结构

- <https://www.bilibili.com/video/BV1dg4y187Y8?p=28>
- Description
  - This lecture is about the central structure of deep neural networks, which are a major force in machine learning. The aim is to find the function that’s constructed to learn the training data and then apply it to the test data.
- Summary
  - The net has layers of nodes. Layer zero is the data.
  - We choose matrix of "weights" from layer to layer.
  - Nonlinear step at each layer! Negative values become zero!
  - We know correct class for the training data.
  - Weights optimized to (usually) output that correct class.
- Related chapter in textbook: Chapter VII.1

### 27_反向传播

- <https://www.bilibili.com/video/BV1dg4y187Y8?p=29>
- Description
  - In this lecture, Professor Strang presents Professor Sra’s theorem which proves the convergence of stochastic gradient descent (SGD). He then reviews backpropagation, a method to compute derivatives quickly, using the chain rule.
- Summary
  - Computational graph: Each step in computing $F(x)$ from the weights
  - Derivative of each step + chain rule gives gradient of $F$.
  - Reverse mode: Backwards from output to input
  - The key step to optimizing weights is backprop + stoch grad descent.
- Related chapter in textbook: Chapter VII.3

### 30_循环矩阵

- <https://www.bilibili.com/video/BV1dg4y187Y8?p=30>
- Description
  - Professor Strang starts this lecture asking the question “Which matrices can be completed to have a rank of 1?” He then provides several examples. In the second part, he introduces convolution and cyclic convolution.
- Summary
  - Which matrices can be completed to have rank = 1?
  - Perfect answer: No cycles in a certain graph
  - Cyclic permutation $P$ and circulant matrices
  - $c_0I + c_1P + c_2P^2 + \cdots$
  - Start of Fourier analysis for vectors
- Related section in textbook: IV.8 and IV.2

### 31_循环矩阵的特征向量，Fourier矩阵

- <https://www.bilibili.com/video/BV1dg4y187Y8?p=31>
- Description
  - This lecture continues with constant-diagonal circulant matrices. Each lower diagonal continues on an upper diagonal to produce $n$ equal entries. The eigenvectors are always the columns of the Fourier matrix and computing is fast.
- Summary
  - Circulants $C$ have $n$ constant diagonals (completed cyclically).
  - Cyclic convolution with $c_0, \cdots, c_{n-1}$ multiplication by $C$
  - Linear shift invariant: LSI for periodic problems
  - Eigenvectors of every $C$ columns of the Fourier matrix
  - Eigenvalues of $C$ (Fourier matrix)(column zero of $C$)
- Related section in textbook: IV.2
