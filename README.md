#  Titanic Survival Prediction Using Machine Learning Classification Models

This project predicts the survival of passengers aboard the RMS Titanic using multiple machine learning algorithms.  
By analyzing passenger data such as age, gender, ticket class, and fare, the model aims to determine the likelihood of survival.  
The goal is to compare the performance of various classifiers and choose the best-performing model.

---

### Dataset Columns:
| Column Name | Description |
|--------------|-------------|
| PassengerId | Unique ID of the passenger |
| Survived | Survival (0 = No, 1 = Yes) |
| Pclass | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) |
| Name | Passenger’s name |
| Sex | Gender |
| Age | Age in years |
| SibSp | Number of siblings/spouses aboard |
| Parch | Number of parents/children aboard |
| Ticket | Ticket number |
| Fare | Passenger fare |
| Cabin | Cabin number |
| Embarked | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |

---

##  Project Workflow

1. **Import Necessary Libraries**  
   Load essential Python libraries such as pandas, numpy, matplotlib, seaborn, and scikit-learn.

2. **Read In and Explore the Data**  
   Import the Titanic dataset and perform exploratory data analysis to understand distributions and correlations.

3. **Data Analysis**  
   Investigate relationships between features (e.g., gender, class, age) and survival outcomes.

4. **Data Visualization**  
   Use graphs (bar plots, histograms, heatmaps) to identify key trends and correlations visually.

5. **Cleaning Data**  
   Handle missing values, encode categorical variables, and normalize numerical data.

6. **Choosing the Best Model**  
   Train and evaluate multiple machine learning models using cross-validation and accuracy metrics.

7. **Creating Submission File**  
   Generate predictions on the test dataset and prepare the final submission file.

---

## Machine Learning Models Used

| Model | Description |
|--------|-------------|
| Gaussian Naive Bayes | Probabilistic classifier based on Bayes' theorem |
| Logistic Regression | Binary classification model for survival prediction |
| Support Vector Machines (SVM) | Finds the best separating hyperplane |
| Perceptron | Simple neural network model for binary classification |
| Decision Tree Classifier | Tree-based model for interpretability |
| Random Forest Classifier | Ensemble of decision trees for robust predictions |
| k-Nearest Neighbors (KNN) | Instance-based learning algorithm |
| Stochastic Gradient Descent (SGD) | Optimization-based linear classifier |
| Gradient Boosting Classifier | Ensemble boosting method for improved accuracy |


---

