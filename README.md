## **The Iris Dataset**

**What it is**

- A multiclass classification dataset collected by Ronald Fisher (1936).
- It’s small and simple: 150 samples of flowers.
- Each sample is a type of iris flower with 4 measured features.

**Goal**: Predict the species of iris from those measurements.

**Features (Inputs, X): Each flower has 4 numeric features (continuous variables):**

- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)
- Sepal = the outer leaf-like part of the flower; Petal = the colored part.

**Target (Output, y)**

- Three possible classes (species of iris):
- Iris setosa (50 samples)
- Iris versicolor (50 samples)
- Iris virginica (50 samples)

**Why it’s useful**

- Small size: Easy to load and visualize (only 150 rows).
- Balanced: Each class has exactly 50 samples → no imbalance problems.
- Clear structure: Some species are easy to separate (Setosa), others overlap (Versicolor vs Virginica).
- Multiclass: Students learn beyond binary classification.

Classic: Almost every ML library has it built-in (scikit-learn, R, MATLAB).

**Example Row**
**Sepal length	Sepal width	Petal length	Petal width	Species**
**5.1 cm	3.5 cm	1.4 cm	0.2 cm	Iris setosa**



## **The Model**

Standard logistic regression using PCA (2 components)



**Required Packages**

- numpy
- pandas
- matplotlib
- scikit-learn