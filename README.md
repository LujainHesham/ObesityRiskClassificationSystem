# Obesity Risk Classification System
### ITIDA/NTI Machine Learning Summer Training Capstone Project

**👥 Team**: Lujain Hesham & Jayda Mohamed  
**🎓 Program**: ITIDA/NTI Machine Learning Summer Training 2024  
**📊 Results**: 90.1% accuracy with Random Forest  
**🏆 Achievement**: 99% training score, Top 80 graduate  
**🔗 Dataset**: UCI Machine Learning Repository  

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset Information](#-dataset-information)
- [Results](#-results)
- [Methodology](#-methodology)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Team & Acknowledgments](#-team--acknowledgments)
- [License](#-license)

## 🎓 Project Overview

This machine learning classification system was developed as our **capstone project for the ITIDA/NTI Machine Learning Summer Training 2024**, where we achieved a **99% final score** and ranked among the **top 80 graduates nationwide**. The project implements a complete end-to-end pipeline to classify individuals into 7 obesity risk levels based on lifestyle, dietary habits, and physical condition data from three Latin American countries.

### 🎯 Project Goals
1. **Practical Application**: Demonstrate comprehensive ML skills learned during ITIDA/NTI training
2. **Real-world Problem Solving**: Address a meaningful healthcare classification challenge
3. **Collaborative Development**: Showcase team-based agile development practices
4. **Full Pipeline Implementation**: From data acquisition to model deployment readiness

### ✨ Key Features
- **Complete ML Pipeline**: EDA → Preprocessing → Feature Engineering → Modeling → Evaluation
- **Multiple Model Comparison**: 5 classifiers with detailed performance analysis
- **Comprehensive Visualization**: Interactive EDA and result visualization
- **Production-ready Code**: Clean, modular, and well-documented implementation
- **Team Collaboration**: Git-based development with regular code reviews

## 📁 Dataset Information

### 📍 Source
- **Name**: Estimation of obesity levels based on eating habits and physical condition
- **Origin**: UCI Machine Learning Repository
- **URL**: https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition
- **Access Date**: August 2024

### 📊 Dataset Characteristics
| Attribute | Details |
|-----------|---------|
| **Total Records** | 2,111 individuals |
| **Features** | 17 attributes |
| **Geographic Scope** | Mexico, Peru, Colombia |
| **Data Composition** | 77% synthetic (Weka + SMOTE), 23% real user data |
| **Collection Method** | Web platform responses + synthetic augmentation |
| **Target Variable** | 7-class obesity classification |

### 🎯 Target Variable: Obesity Levels
The dataset classifies individuals into 7 distinct obesity categories:
1. **Insufficient Weight**
2. **Normal Weight**
3. **Overweight Level I**
4. **Overweight Level II**
5. **Obesity Type I**
6. **Obesity Type II**
7. **Obesity Type III**

### 🔧 Feature Categories
| Category | Features | Description |
|----------|----------|-------------|
| **Demographic** | Gender, Age, Height, Weight | Basic individual characteristics |
| **Genetic** | Family history with overweight | Hereditary risk factors |
| **Dietary Habits** | FAVC, FCVC, NCP, CAEC, CH2O, CALC | Food consumption patterns and frequencies |
| **Physical Activity** | FAF | Frequency of physical activity |
| **Lifestyle** | SMOKE, SCC, TUE, MTRANS | Daily habits and transportation |
| **Engineered** | BMI | Calculated Body Mass Index |

## 📊 Results & Performance

### 🏆 Model Performance Summary
| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| **Random Forest** | **90.1%** | **0.902** | **0.901** | **0.901** | 4.2s |
| Multi-Layer Perceptron | 89.6% | 0.897 | 0.896 | 0.896 | 12.8s |
| Logistic Regression | 88.4% | 0.885 | 0.884 | 0.884 | 1.5s |
| Support Vector Machine | 87.8% | 0.879 | 0.878 | 0.878 | 8.7s |
| K-Nearest Neighbors | 87.2% | 0.873 | 0.872 | 0.872 | 0.8s |

### 📈 Key Performance Metrics
- **Best Overall Model**: Random Forest Classifier
- **Cross-Validation Score**: 89.8% ± 0.8% (5-fold CV)
- **Class-wise Performance**: Balanced across all 7 obesity categories
- **Confusion Matrix**: Clear diagonal dominance with minimal misclassification

### 🔍 Important Findings
1. **Feature Impact**: BMI (engineered feature) showed highest importance
2. **Data Quality**: Synthetic augmentation improved model generalization
3. **Class Balance**: Effective handling of multi-class imbalance
4. **Model Stability**: Consistent performance across validation splits

## 🧪 Methodology & Technical Approach

### 1. Exploratory Data Analysis (EDA)
- **Statistical Summary**: Distribution analysis of all features
- **Correlation Study**: Heatmap visualization of feature relationships
- **Missing Values**: Comprehensive null value analysis
- **Outlier Detection**: IQR and visualization methods
- **Class Distribution**: Analysis of target variable balance

### 2. Data Preprocessing Pipeline
```python
# Key preprocessing steps implemented:
1. Data type validation and conversion
2. Handling of categorical variables:
   - Label Encoding for binary features
   - One-Hot Encoding for multi-class features
3. Feature scaling: StandardScaler for continuous variables
4. Train-test split: 80-20 stratified split
```

### 3. Feature Engineering
- **BMI Calculation**: `Weight / (Height²)` - most impactful engineered feature
- **Outlier Management**: IQR-based capping for continuous variables
- **Feature Selection**:
  - ANOVA F-test for initial filtering
  - Random Forest importance for final selection
  - Correlation analysis to remove multicollinearity

### 4. Model Development Strategy
```python
Models Implemented:
1. Logistic Regression - Baseline model
2. K-Nearest Neighbors - Distance-based classification
3. Random Forest - Ensemble method (best performer)
4. Support Vector Machine - Margin optimization
5. Multi-Layer Perceptron - Neural network approach

Hyperparameter Tuning:
- GridSearchCV with 5-fold cross-validation
- Randomized search for complex parameter spaces
- Early stopping for neural network training
```

### 5. Evaluation Framework
- **Primary Metric**: Accuracy for overall performance
- **Secondary Metrics**: Precision, Recall, F1-Score per class
- **Visualization**: Confusion matrices, ROC curves, learning curves
- **Statistical Tests**: Paired t-tests for model comparison

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for version control)

### Step-by-Step Installation
```bash
# 1. Clone the repository
git clone https://github.com/LujainHesham/obesity-risk-classification.git
cd obesity-risk-classification

# 2. Create and activate virtual environment
python -m venv obesity_env
source obesity_env/bin/activate  # On Windows: obesity_env\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset (automated in notebook)
# The notebook will automatically fetch the dataset from UCI repository
```

### 📦 Dependencies (`requirements.txt`)
```
# Core Data Science Stack
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.3.0

# Visualization
matplotlib==3.7.1
seaborn==0.12.2
ydata-profiling==4.5.1

# Jupyter Environment
jupyter==1.0.0
ipython==8.14.0

# Utilities
tqdm==4.65.0
joblib==1.3.1
```

## 🚀 Usage Guide

### Running the Complete Pipeline
```bash
# Method 1: Run via Jupyter Notebook (Recommended)
jupyter notebook
# Open and execute notebooks in this order:
# 1. 01_eda_analysis.ipynb
# 2. 02_preprocessing.ipynb
# 3. 03_model_training.ipynb
# 4. 04_evaluation.ipynb

# Method 2: Run as Python scripts
python src/preprocessing.py
python src/training.py
python src/evaluation.py
```

### Notebook Execution Order
1. **01_eda_analysis.ipynb** - Data exploration and visualization
2. **02_preprocessing.ipynb** - Data cleaning and feature engineering
3. **03_model_training.ipynb** - Model training and hyperparameter tuning
4. **04_evaluation.ipynb** - Performance evaluation and results analysis

### Generating Reports
```python
# Generate EDA report
python -c "from src.eda import generate_report; generate_report()"

# Train and save best model
python src/train_model.py --model random_forest --save_model

# Make predictions on new data
python src/predict.py --input data/new_samples.csv --output predictions.csv
```

## 📁 Project Structure
```
obesity-risk-classification/
│
├── data/
│   ├── raw/                          # Original dataset files
│   │   ├── ObesityDataSet.csv        # Primary dataset
│   │   └── data_description.txt      # Dataset documentation
│   │
│   ├── processed/                    # Cleaned and processed data
│   │   ├── train_data.csv           # Training split
│   │   ├── test_data.csv            # Testing split
│   │   └── features_selected.csv    # Selected features
│   │
│   └── external/                     # External resources
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_eda_analysis.ipynb        # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb       # Data preprocessing
│   ├── 03_model_training.ipynb      # Model training
│   ├── 04_evaluation.ipynb          # Model evaluation
│   └── 05_final_report.ipynb        # Comprehensive report
│
├── src/                              # Source code
│   ├── __init__.py
│   ├── data_loader.py               # Data loading utilities
│   ├── preprocessing.py             # Preprocessing functions
│   ├── feature_engineering.py       # Feature engineering
│   ├── models.py                    # Model definitions
│   ├── training.py                  # Training routines
│   ├── evaluation.py                # Evaluation metrics
│   └── visualization.py             # Plotting functions
│
├── models/                           # Trained models
│   ├── random_forest.pkl           # Best model
│   ├── svm_model.pkl               # SVM model
│   └── mlp_model.pkl               # Neural network model
│
├── reports/                          # Generated reports
│   ├── eda_report.html             # ydata-profiling report
│   ├── feature_importance.png      # Feature importance plot
│   ├── confusion_matrix.png        # Confusion matrix
│   └── performance_summary.csv     # Model performance summary
│
├── tests/                           # Unit tests
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_evaluation.py
│
├── requirements.txt                 # Dependencies
├── environment.yml                  # Conda environment
├── LICENSE                          # MIT License
└── README.md                        # This file
```

### 🧑‍💻 Development Team
**Lujain Hesham** & **Jayda Mohamed**

### 🤝 Collaborative Approach
- **Agile Methodology**: One-week sprint with daily standups
- **Code Reviews**: Mutual code review and quality assurance
- **Pair Programming**: Collaborative problem-solving sessions
- **Version Control**: Git workflow with feature branching
- **Documentation**: Shared responsibility for comprehensive docs

### 🙏 Acknowledgments
- **ITIDA/NTI** for providing exceptional Machine Learning training and this project opportunity
- **UCI Machine Learning Repository** for maintaining and providing high-quality datasets
- **Dataset Creators** (P. M. S. et al.) for collecting and sharing this valuable healthcare data
- **Training Mentors** for their guidance, feedback, and support throughout the program
- **Open Source Community** for the incredible tools and libraries that made this project possible

### 📚 References
1. UCI Machine Learning Repository: Estimation of obesity levels dataset
2. Scikit-learn Documentation
3. ITIDA/NTI Machine Learning Training Materials
4. Research papers on obesity prediction and classification

## 📄 License & Citation

### 📜 License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### 📚 Academic Citation
If you use this project in academic work, please cite:

```bibtex
@software{obesity_classification_2024,
  author = {Lujain Hesham and Jayda Mohamed},
  title = {Obesity Risk Classification System},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/LujainHesham/obesity-risk-classification}},
  note = {ITIDA/NTI Machine Learning Training Capstone Project}
}
```

### 📊 Dataset Citation
```bibtex
@misc{uci_obesity_2019,
  author = {P. M. S. et al.},
  title = {Estimation of obesity levels based on eating habits and physical condition},
  year = {2019},
  publisher = {UCI Machine Learning Repository},
  url = {https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition}
}
```

### ⚠️ Disclaimer
This project was developed for **educational purposes** as part of the ITIDA/NTI Machine Learning Training Program. The models are not intended for medical diagnosis or clinical decision-making. Always consult healthcare professionals for medical advice.

---

## 🌟 Future Enhancements

### Planned Improvements
1. **Web Application**: Deploy as interactive prediction tool
2. **API Development**: REST API for model integration
3. **Real-time Prediction**: Mobile app integration
4. **Additional Models**: Experiment with XGBoost, LightGBM
5. **Advanced Features**: Incorporate time-series data if available

### Research Directions
1. **Explainable AI**: SHAP/LIME for model interpretability
2. **Transfer Learning**: Apply to similar healthcare datasets
3. **Ensemble Methods**: Stacking and voting classifiers
4. **Deep Learning**: CNN/LSTM for sequential pattern recognition

---

**🎓 Training Context**: This project represents the culmination of 120+ hours of intensive Machine Learning training through the ITIDA/NTI program, where both team members achieved a **99% final score** and ranked among the **top 80 graduates nationwide**.

**📧 Contact**: For questions or collaborations, please contact [lujainnheshamm@gmail.com](mailto:lujainnheshamm@gmail.com)

**⭐ Support**: If you find this project useful, please consider giving it a star on GitHub!

---

*Last Updated: March 2025*  
*Project Duration: August 2024*  
*Training Program: ITIDA/NTI Machine Learning Summer Training 2024*
