# Machine Learning Implementations: A Learning Journey

This repository documents my learning process and implementations of various machine learning algorithms and concepts. It covers fundamental techniques in Classification, Regression, Deep Learning, Natural Language Processing, and Model Selection using Python.

## Repository Structure

The code is organized into the following main categories:

* **`/Classification`**: Contains implementations of various supervised classification algorithms.
* **`/Deep Learning`**: Explores Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN).
* **`/Model_Selection`**: Demonstrates techniques for evaluating and comparing different models for both classification and regression tasks.
* **`/Natural Language Processing`**: Includes basic NLP techniques, likely focusing on text classification.
* **`/Regression`**: Contains implementations of various supervised regression algorithms.

Each algorithm typically has its own subdirectory containing:
* A Python script (`.py`) with the core implementation.
* A Jupyter Notebook (`.ipynb`) for interactive exploration and explanation.
* The dataset (`.csv` or `.tsv`) used for training and testing.
* (Often) A `Color Blind Friendly Images` folder with visualizations (e.g., training/test set results).

## Topics Covered

### 1. Classification

Algorithms implemented using `Social_Network_Ads.csv` dataset:
* Decision Tree Classification
* K-Nearest Neighbors (K-NN)
* Kernel SVM (Support Vector Machine)
* Logistic Regression
* Naive Bayes
* Random Forest Classification
* Support Vector Machine (SVM)

### 2. Regression

Algorithms implemented include:
* **Simple Linear Regression:** Using `Salary_Data.csv`
* **Multiple Linear Regression:** Using `50_Startups.csv`
* **Polynomial Regression:** Using `Position_Salaries.csv`
* **Support Vector Regression (SVR):** Using `Position_Salaries.csv`
* **Decision Tree Regression:** Using `Position_Salaries.csv`
* **Random Forest Regression:** Using `Position_Salaries.csv`

### 3. Deep Learning

* **Artificial Neural Networks (ANN):** Implemented for a classification task (likely churn prediction) using `Churn_Modelling.csv`. Includes a visualization for Stochastic Gradient Descent.
* **Convolutional Neural Networks (CNN):** Implementation of a CNN (details on the specific task, e.g., image classification, would be inside the notebook/script).

### 4. Natural Language Processing (NLP)

* **Basic NLP Techniques:** Implementation likely focused on text processing and classification using `Restaurant_Reviews.tsv`.

### 5. Model Selection & Evaluation

* Contains code demonstrating how to evaluate and compare the performance of different models presented in the Classification and Regression sections, using a `Data.csv` dataset (contents may vary based on regression/classification task).

## Datasets Used

* `Social_Network_Ads.csv`
* `Churn_Modelling.csv`
* `Position_Salaries.csv`
* `50_Startups.csv`
* `Salary_Data.csv`
* `Restaurant_Reviews.tsv`
* `Data.csv` (Used in Model Selection, potentially different versions for Classification/Regression)

## Technology Stack

* **Language:** Python 3.x
* **Core Libraries:**
    * Scikit-learn (for ML algorithms, preprocessing, metrics)
    * Pandas (for data manipulation and loading CSVs)
    * NumPy (for numerical operations)
    * Matplotlib / Seaborn (for plotting and visualization)
    * (Potentially) TensorFlow / Keras (for Deep Learning models - ANN/CNN)
* **Environment:** Jupyter Notebooks (`.ipynb`) for exploration, Python scripts (`.py`) for standalone execution.

## How to Use

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```
2.  **Navigate** to the specific algorithm or topic folder you are interested in (e.g., `cd Classification/"Decision Tree Classification"/Python`).
3.  **Install Dependencies:** It's recommended to create a virtual environment. You might need to install libraries mentioned in the Technology Stack. Consider adding a `requirements.txt` file to your repository for easier setup.
    ```bash
    pip install numpy pandas matplotlib scikit-learn jupyter notebook # Add others like tensorflow if needed
    ```
4.  **Run the Code:**
    * Execute the Python scripts: `python <script_name>.py`
    * Explore the Jupyter Notebooks: `jupyter notebook <notebook_name>.ipynb`

## Visualizations

Many sections include plots to visualize model performance, decision boundaries, or data distributions. Where applicable, efforts have been made to provide color-blind friendly versions of these plots.
