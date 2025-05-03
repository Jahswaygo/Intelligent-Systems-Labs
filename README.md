# Intelligent Systems Labs

This repository contains a series of labs completed as part of an Intelligent Systems course. Each lab explores different machine learning and data analysis techniques, ranging from regression to clustering and neural networks. Below is a detailed explanation of each lab, including the methods used and their purpose.

---

## Lab 1: Linear Regression and Standardization

### Overview
Lab 1 focuses on implementing linear regression from scratch and comparing it with scikit-learn's implementation. The lab also explores the impact of standardization on regression performance.

### Methods Used
1. **Linear Regression**:
   - Linear regression is a supervised learning algorithm used to model the relationship between a dependent variable (`y`) and one or more independent variables (`X`) by fitting a linear equation to the data.
   - The equation is of the form:  
     `y = m * X + b`  
     where `m` is the slope (coefficient) and `b` is the intercept.

2. **Gradient Descent**:
   - Gradient descent is an optimization algorithm used to minimize the cost function by iteratively updating the model parameters (`m` and `b`).
   - The cost function used is the Mean Squared Error (MSE):  
     `Cost = (1/N) * Σ(y - (m * X + b))²`
   - The parameters are updated using the gradients:  
     `m = m - α * ∂Cost/∂m`  
     `b = b - α * ∂Cost/∂b`  
     where `α` is the learning rate.

3. **Standardization**:
   - Standardization transforms the features to have a mean of 0 and a standard deviation of 1.
   - This ensures numerical stability and faster convergence during gradient descent.

4. **Comparison with scikit-learn**:
   - The lab compares the custom implementation of linear regression with scikit-learn's `LinearRegression` model to validate the results.

### Key Insights
- Standardization improves numerical stability and ensures faster convergence.
- Gradient descent is sensitive to the choice of the learning rate (`α`).
- The scikit-learn implementation provides similar results, validating the custom implementation.

---

## Lab 2: Exploratory Data Analysis (EDA) and Logistic Regression

### Overview
Lab 2 focuses on exploratory data analysis (EDA) and the implementation of logistic regression for binary classification.

### Methods Used
1. **Exploratory Data Analysis (EDA)**:
   - **Class Distribution**: Visualized using a pie chart to understand the balance between classes.
   - **Correlation Matrix**: Displayed to identify relationships between features.
   - **One-Hot Encoding**: Categorical variables are converted into numerical representations using one-hot encoding.

2. **Logistic Regression**:
   - Logistic regression is a supervised learning algorithm used for binary classification.
   - The hypothesis function is:  
     `h(z) = 1 / (1 + e^(-z))`  
     where `z = X * θ`.
   - The cost function is the binary cross-entropy loss:  
     `Cost = -(1/m) * Σ(y * log(h(z)) + (1 - y) * log(1 - h(z)))`.

3. **Gradient Descent**:
   - Parameters (`θ`) are updated iteratively to minimize the cost function.

4. **Learning Curves**:
   - The lab explores the impact of different learning rates and epochs on the convergence of the cost function.

5. **Comparison with scikit-learn**:
   - The custom implementation is compared with scikit-learn's `LogisticRegression` to validate the results.

### Key Insights
- Logistic regression is effective for binary classification tasks.
- Learning rate and the number of epochs significantly impact the convergence of the cost function.
- The scikit-learn implementation provides similar results, validating the custom implementation.

---

## Lab 3: Clustering Algorithms

### Overview
Lab 3 explores clustering techniques, including K-Means, Nearest Neighbor Clustering, and DBSCAN. The lab also compares custom implementations with scikit-learn's implementations.

### Methods Used
1. **K-Means Clustering**:
   - K-Means is an unsupervised learning algorithm that partitions data into `k` clusters.
   - Steps:
     1. Initialize `k` cluster centers randomly.
     2. Assign each data point to the nearest cluster center.
     3. Recalculate cluster centers as the mean of assigned points.
     4. Repeat until convergence (no change in cluster centers).

2. **Nearest Neighbor Clustering**:
   - A custom clustering algorithm that groups points based on a distance threshold.
   - Points within the threshold distance are assigned to the same cluster.

3. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**:
   - DBSCAN identifies clusters based on density.
   - Parameters:
     - `ε` (epsilon): Maximum distance between two points to be considered neighbors.
     - `min_samples`: Minimum number of points required to form a dense region.
   - Points are classified as:
     - **Core Points**: Points with at least `min_samples` neighbors within `ε`.
     - **Border Points**: Points within `ε` of a core point but with fewer than `min_samples` neighbors.
     - **Noise Points**: Points that are neither core nor border points.

4. **Performance Metrics**:
   - **Intra-Cluster Distance**: Measures compactness within clusters.
   - **Inter-Cluster Distance**: Measures separation between clusters.

5. **Comparison with scikit-learn**:
   - The custom implementations are compared with scikit-learn's `KMeans`, `AgglomerativeClustering`, and `DBSCAN`.

### Key Insights
- K-Means is sensitive to the initialization of cluster centers and assumes spherical clusters.
- Nearest Neighbor Clustering is simple but computationally expensive for large datasets.
- DBSCAN is effective for identifying clusters of varying densities and shapes but requires careful tuning of `ε` and `min_samples`.

---

## Lab 4: Neural Networks and Regression

### Overview
Lab 4 focuses on building and training neural networks for regression tasks using TensorFlow/Keras.

### Methods Used
1. **Dataset Generation**:
   - A synthetic dataset is generated using a polynomial function:  
     `y = 0.2x⁴ + 2x³ + 0.1x² + 10`.

2. **Data Preprocessing**:
   - **Shuffling**: Randomly shuffles the dataset to ensure the model generalizes well.
   - **Splitting**: Splits the data into training, validation, and test sets (30%, 20%, 50%).
   - **Scaling**: Scales the data to the range [0, 1] for faster convergence.

3. **Neural Network Architecture**:
   - A feedforward neural network is built using TensorFlow/Keras.
   - Example architecture:
     - Input layer
     - Hidden layers with ReLU or Tanh activation
     - Output layer with a linear activation function

4. **Training**:
   - The model is trained using the Adam optimizer and Mean Squared Error (MSE) loss function.
   - Training is performed for 20 epochs with a batch size of 12.

5. **Evaluation**:
   - The model's performance is evaluated using metrics such as:
     - Mean Absolute Error (MAE)
     - Mean Squared Error (MSE)
     - Root Mean Squared Error (RMSE)
     - R² Score

6. **Experimentation**:
   - The lab experiments with different neural network architectures, activation functions, and scaling techniques to observe their impact on performance.

7. **XOR Problem**:
   - A neural network is trained to solve the XOR problem, demonstrating the ability of neural networks to model non-linear decision boundaries.

### Key Insights
- Neural networks are powerful for modeling complex, non-linear relationships.
- Data preprocessing (e.g., scaling) significantly impacts model performance.
- The choice of architecture and activation functions affects the model's ability to learn.
