# Matrix Methods in Data Analysis, Signal Processing, and Machine Learning
![图片](./images/Relationship.jpg)
## 课程信息
- 资源：[MIT 18.065](https://ocw.mit.edu/courses/mathematics/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/index.htm
)
- 教师：[Prof. Gilbert Strang](http://math.mit.edu/~gs/)
- 视频：[YouTube](https://www.youtube.com/watch?v=t36jZG07MYc)
- 教材：[教材](http://math.mit.edu/~gs/learningfromdata/)
- 习题：[习题](Assignments%20problem%20sets/MIT18_065S18PSets.pdf)
- 答案：[答案](Assignments%20problem%20sets/Solutions%20to%20Exercises.pdf)
- 介绍：
  - Linear algebra concepts are key for understanding and creating machine learning algorithms, especially as applied to deep learning and neural networks. This course reviews linear algebra with applications to probability and statistics and optimization–and above all a full explanation of deep learning.

## 视频课程
### 00.1_课程简介
- https://www.bilibili.com/video/BV1dg4y187Y8?p=1

### 00.2_采访
- https://www.bilibili.com/video/BV1dg4y187Y8?p=2

### 01_矩阵列空间
- https://www.bilibili.com/video/BV1dg4y187Y8?p=3
- Description
  - In this first lecture, Professor Strang introduces the linear algebra principles critical for understanding the content of the course.  In particular, matrix-vector multiplication $Ax$ and the column space of a matrix and the rank.
- Summary
  - Independent columns = basis for the column space
  - Rank = number of independent columns
  - $A=CR$ leads to: Row rank equals column rank

### 02_矩阵相乘和分解
- https://www.bilibili.com/video/BV1dg4y187Y8?p=4
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
- https://www.bilibili.com/video/BV1dg4y187Y8?p=5
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
- https://www.bilibili.com/video/BV1dg4y187Y8?p=6
- Description
  - Professor Strang begins this lecture talking about eigenvectors and eigenvalues and why they are useful. Then he moves to a discussion of symmetric matrices, in particular, positive definite matrices.
- Summary
  - $Ax=\lambda x$
  - $A^2x=\lambda^2 x$
  - Write other vectors as combinations of eigenvectors
  - Similar matrix $B=M^{-1}AM$ has the same eigenvalues as $A$
- Related section in textbook: I.6
