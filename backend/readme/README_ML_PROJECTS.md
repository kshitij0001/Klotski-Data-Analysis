# 🧩 Klotski Puzzle Machine Learning Projects

A comprehensive machine learning analysis of Klotski sliding block puzzles, featuring predictive modeling, clustering analysis, and puzzle complexity insights. This project demonstrates end-to-end ML pipeline development from data generation to business intelligence.

## 📊 Dataset Overview

**Generated Dataset**: 1,000-2,000 unique Klotski puzzle boards with 22 engineered features

### Core Features
- **Puzzle Metadata**: `puzzle_id`, `timestamp`, `is_solvable`
- **Block Composition**: `total_blocks`, `block_count_1x1`, `block_count_1x2`, `block_count_2x1`, `block_count_2x2`
- **Goal Analysis**: `goal_initial_row`, `goal_initial_col`, `goal_distance_to_target`, `goal_manhattan_distance`
- **Spatial Features**: `blocks_between_goal_target`, `wall_adjacent_sides`
- **Adjacent Block Analysis**: `adjacent_1x1_count`, `adjacent_1x2_count`, `adjacent_2x1_count`
- **Solution Metrics**: `solution_length`, `solve_time_seconds`
- **Visual Representation**: `board_visual` (Unicode rendering)
- **Initial Pieces Co-ordinate** : `initial_block_states`

## 🎯 Machine Learning Projects

### 1. Binary Classification: Solvability Prediction
**Objective**: Predict whether a puzzle can be solved
```python
Target Variable: is_solvable (True/False)
Expected Accuracy: 85-95%
Business Value: Quality control for puzzle generation
Models: Random Forest, XGBoost, Logistic Regression
```

**Key Questions**:
- Which board configurations are most likely unsolvable?
- What spatial arrangements predict impossibility?
- Can we avoid generating unsolvable puzzles?

### 2. Regression: Solution Length Prediction
**Objective**: Predict how many moves a solvable puzzle requires
```python
Target Variable: solution_length (for solvable puzzles)
Expected R²: 0.6-0.8
Business Value: Difficulty estimation and player experience
Models: Random Forest Regressor, Gradient Boosting
```

**Key Questions**:
- Which features best predict puzzle difficulty?
- How does block arrangement affect solution complexity?
- Can we generate puzzles with target difficulty?

### 3. Multi-class Classification: Difficulty Tiers
**Objective**: Classify puzzles into difficulty categories
```python
Categories:
- Easy: 1-20 moves
- Medium: 21-50 moves  
- Hard: 51+ moves
- Impossible: unsolvable

Expected Accuracy: 70-85%
Business Value: Automatic difficulty rating system
```

### 4. Unsupervised Learning: Puzzle Clustering
**Objective**: Discover natural puzzle archetypes
```python
Methods: K-means, Hierarchical Clustering, t-SNE visualization
Expected Outcomes: 4-6 distinct puzzle types
Business Value: Understanding puzzle diversity and generation patterns
```

**Cluster Analysis**:
- Identify common puzzle patterns
- Understand feature relationships
- Optimize puzzle generation parameters

### 5. Anomaly Detection: Unusual Puzzles
**Objective**: Find puzzles with unexpected characteristics
```python
Methods: Isolation Forest, Local Outlier Factor
Expected Outcomes: 3-5% anomalous puzzles
Business Value: Quality assurance and interesting edge cases
```

### 6. Feature Importance Analysis
**Objective**: Understand what makes puzzles difficult
```python
Methods: SHAP values, Permutation importance, Feature correlation
Expected Insights: Top 5-7 predictive features
Business Value: Inform puzzle design and generation strategy
```

## 📈 Expected Results & Portfolio Value

### Model Performance Targets
- **Solvability Prediction**: 90%+ accuracy
- **Solution Length**: R² > 0.65  
- **Difficulty Classification**: 75%+ accuracy
- **Feature Importance**: Clear ranking of predictive features

### Business Intelligence Insights
- **Optimal puzzle generation parameters**
- **Key difficulty drivers identified**
- **Puzzle archetype taxonomy created**
- **Quality control metrics established**

### Portfolio Highlights
- **"Achieved 92% accuracy predicting puzzle solvability using ensemble methods"**
- **"Identified 5 distinct puzzle archetypes through unsupervised clustering"**
- **"Built regression model predicting solution complexity with 0.74 R²"**
- **"Discovered spatial density as strongest difficulty predictor"**

## 🚀 Quick Start Guide

### Prerequisites
```bash
# Required Python packages
pip install pandas numpy scikit-learn matplotlib seaborn
pip install xgboost tqdm rich
```

### 1. Generate Dataset
```bash
# Run from project root
python backend/scripts/enhanced_data_generator.py
# Select: 1000 puzzles, CSV format
# Output will be saved to backend/data/enhanced_dataset_YYYYMMDD_HHMMSS.csv
```

### 2. Load and Explore Data
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# Load the most recent dataset (adjust path as needed)
data_files = glob.glob('backend/data/enhanced_dataset_*.csv')
if data_files:
    latest_file = max(data_files)  # Get most recent file
    df = pd.read_csv(latest_file)
    print(f"Loaded: {latest_file}")
else:
    df = pd.read_csv('enhanced_dataset.csv')  # Fallback

# Quick overview
print(f"Dataset shape: {df.shape}")
print(f"Solvable rate: {df['is_solvable'].mean():.2%}")
print(f"Average solution length: {df[df['is_solvable']]['solution_length'].mean():.1f}")
```

### 3. Solvability Prediction Model
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Prepare features
feature_cols = [col for col in df.columns if col not in 
               ['puzzle_id', 'timestamp', 'board_visual', 'is_solvable', 
                'solution_length', 'solve_time_seconds']]
X = df[feature_cols]
y = df['is_solvable']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate
y_pred = rf_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 10 Most Important Features:")
print(feature_importance.head(10))
```

### 4. Solution Length Regression
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Filter to solvable puzzles only
solvable_df = df[df['is_solvable'] == True]
X_reg = solvable_df[feature_cols]
y_reg = solvable_df['solution_length']

# Train regression model
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2)

rf_regressor.fit(X_train_reg, y_train_reg)
y_pred_reg = rf_regressor.predict(X_test_reg)

print(f"MAE: {mean_absolute_error(y_test_reg, y_pred_reg):.2f}")
print(f"R² Score: {r2_score(y_test_reg, y_pred_reg):.3f}")
```

### 5. Clustering Analysis
```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Prepare data for clustering (use only numeric features)
X_numeric = df[feature_cols].select_dtypes(include=[np.number])
X_scaled = StandardScaler().fit_transform(X_numeric)

# K-means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Visualize with PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
plt.colorbar(scatter)
plt.title('Puzzle Clusters (PCA Visualization)')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()
```

## 📊 2-Day Sprint Timeline

### Day 1: Core ML Models (8 hours)
```
├── Data generation & cleaning: 1 hour
├── Exploratory data analysis: 2 hours  
├── Solvability prediction: 2 hours
├── Solution length regression: 2 hours
└── Feature importance analysis: 1 hour
```

### Day 2: Advanced Analysis (8 hours)
```
├── Difficulty classification: 2 hours
├── Clustering analysis: 2 hours
├── Anomaly detection: 1 hour
├── Visualization & insights: 2 hours
└── Documentation & presentation: 1 hour
```

## 🎯 Success Metrics

### Minimum Viable Analysis
- ✅ 85%+ solvability prediction accuracy
- ✅ R² > 0.6 for solution length regression
- ✅ Clear feature importance ranking
- ✅ 3-5 puzzle clusters identified
- ✅ Professional visualizations

### Portfolio Enhancement Goals
- ✅ Business recommendations based on insights
- ✅ Model interpretability with SHAP values
- ✅ Comprehensive evaluation metrics
- ✅ Clear problem-solving narrative
- ✅ Technical depth demonstration

## 🔧 Technical Stack

- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, XGBoost
- **Visualization**: matplotlib, seaborn, plotly
- **Model Interpretation**: SHAP, permutation importance
- **Clustering**: K-means, hierarchical clustering
- **Anomaly Detection**: Isolation Forest

## 📁 Project Structure

```
klotski-ml-analysis/
├── data/
│   ├── enhanced_dataset.csv          # Generated puzzle dataset
│   └── processed_data.csv            # Cleaned and engineered features
├── notebooks/
│   ├── 01_data_exploration.ipynb     # EDA and data quality
│   ├── 02_classification_models.ipynb # Solvability and difficulty prediction
│   ├── 03_regression_analysis.ipynb   # Solution length prediction
│   └── 04_clustering_analysis.ipynb   # Unsupervised learning
├── src/
│   ├── data_processing.py            # Data cleaning utilities
│   ├── model_training.py             # ML model implementations
│   └── visualization.py              # Plotting functions
├── results/
│   ├── model_performance.json        # Evaluation metrics
│   ├── feature_importance.csv        # Feature rankings
│   └── cluster_analysis.json         # Clustering results
└── README_ML_PROJECTS.md            # This file
```

## 🏆 Expected Portfolio Impact

This project demonstrates:
- **End-to-end ML pipeline development**
- **Multiple ML paradigms**: supervised and unsupervised learning
- **Real-world problem solving**: game difficulty and user experience
- **Feature engineering expertise**: from raw puzzles to predictive features
- **Business intelligence**: actionable insights for game design
- **Technical depth**: model interpretation and evaluation

Perfect for showcasing ML skills to potential employers in gaming, entertainment, or data science roles!

---

**Ready to build impressive ML models from puzzle data!** 🧩🚀