# 🧩 Klotski Puzzle Data Science Project

A comprehensive data science toolkit for analyzing, generating, and solving Klotski sliding block puzzles. This project combines classical algorithmic solving with advanced machine learning techniques to understand puzzle complexity, predict solvability, and generate new challenges.

## 📊 Dataset Overview

The enhanced data generator creates rich datasets with **20 clean, ML-ready features** for each puzzle:

### Core Features
- **Puzzle Metadata**: `puzzle_id`, `timestamp`, `is_solvable`
- **Block Composition**: `total_blocks`, `block_count_1x1`, `block_count_1x2`, `block_count_2x1`, `block_count_2x2`
- **Goal Analysis**: `goal_initial_row`, `goal_initial_col`, `goal_distance_to_target`, `goal_manhattan_distance`
- **Spatial Features**: `blocks_between_goal_target`, `wall_adjacent_sides`
- **Adjacent Block Analysis**: `adjacent_1x1_count`, `adjacent_1x2_count`, `adjacent_2x1_count`
- **Solution Metrics**: `solution_length`, `solve_time_seconds`
- **Visual Representation**: `board_visual` (beautiful Unicode rendering)
- **Sequence Data**: `step_number` (when using intermediate steps mode)

## 🔍 Exploratory Data Analysis (EDA)

### 1. Dataset Quality & Overview
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load and inspect dataset
df = pd.read_csv('enhanced_dataset.csv')
print(f"Dataset shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Solvable puzzles: {df['is_solvable'].mean():.2%}")
```

### 2. Block Composition Analysis
- **Distribution Analysis**: Visualize block type frequencies
- **Correlation Matrix**: Understand block count relationships  
- **Solvability Impact**: How block composition affects puzzle difficulty

```python
# Block composition visualization
block_cols = ['block_count_1x1', 'block_count_1x2', 'block_count_2x1', 'block_count_2x2']
df[block_cols].hist(bins=15, figsize=(12, 8))
plt.suptitle('Block Type Distributions')

# Correlation heatmap
correlation_matrix = df[block_cols + ['solution_length']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
```

### 3. Spatial & Positioning Analysis
- **Goal Position Heat Maps**: Where does the 2x2 block typically start?
- **Distance Metrics**: Distribution of goal distances and Manhattan distances
- **Blocking Patterns**: Analysis of `blocks_between_goal_target`

```python
# Goal position heatmap
pivot_table = df.pivot_table(values='is_solvable', 
                            index='goal_initial_row', 
                            columns='goal_initial_col', 
                            aggfunc='mean')
sns.heatmap(pivot_table, annot=True, cmap='viridis')
plt.title('Solvability by Goal Starting Position')
```

### 4. Solvability Deep Dive
- **Solve Time Distribution**: Understand algorithm performance
- **Solution Length Analysis**: What makes solutions longer?
- **Feature Importance**: Which features best predict solvability?

```python
# Solution length distribution for solvable puzzles
solvable_df = df[df['is_solvable'] == True]
plt.figure(figsize=(10, 6))
plt.hist(solvable_df['solution_length'], bins=30, alpha=0.7)
plt.xlabel('Solution Length (moves)')
plt.ylabel('Frequency')
plt.title('Distribution of Solution Lengths')
```

### 5. Board Visual Analysis
- **Pattern Recognition**: Identify common board configurations
- **Visual Clustering**: Group similar layouts using board_visual
- **Complexity Visualization**: Create difficulty heat maps

```python
# Display beautiful board visuals
print("Sample board configurations:")
for i in range(3):
    print(f"\nPuzzle {i+1} (Solvable: {df.iloc[i]['is_solvable']}):")
    print(df.iloc[i]['board_visual'])
```

## ⚙️ Feature Engineering (FE)

### Advanced Complexity Analysis

The project includes a sophisticated **Complexity Analyzer** that computes advanced difficulty metrics:

```python
def analyze_solution_patterns(solution_path):
    """Analyze patterns in the solution path"""
    if not solution_path or len(solution_path) < 2:
        return {
            'horizontal_moves': 0,
            'vertical_moves': 0,
            'goal_piece_moves': 0,
            'move_sequence_entropy': 0,
            'backtrack_count': 0,
            'unique_positions': 0
        }
    
    horizontal_moves = 0
    vertical_moves = 0
    goal_piece_moves = 0
    position_hashes = set()
    
    for i in range(1, len(solution_path)):
        prev_state = solution_path[i-1]
        curr_state = solution_path[i]
        
        # Create position hash for uniqueness
        pos_hash = tuple(tuple((b.row_pos, b.col_pos, b.num_rows, b.num_cols)) for b in curr_state)
        position_hashes.add(pos_hash)
        
        # Find which block moved
        moved_block = None
        for j, (prev_block, curr_block) in enumerate(zip(prev_state, curr_state)):
            if (prev_block.row_pos != curr_block.row_pos or 
                prev_block.col_pos != curr_block.col_pos):
                moved_block = curr_block
                break
        
        if moved_block:
            # Determine move direction
            prev_block = prev_state[j]
            if prev_block.row_pos != moved_block.row_pos:
                vertical_moves += 1
            if prev_block.col_pos != moved_block.col_pos:
                horizontal_moves += 1
            
            # Check if goal piece moved
            if moved_block.num_rows == 2 and moved_block.num_cols == 2:
                goal_piece_moves += 1
    
    # Calculate entropy (simplified)
    total_moves = horizontal_moves + vertical_moves
    if total_moves > 0:
        h_ratio = horizontal_moves / total_moves
        v_ratio = vertical_moves / total_moves
        entropy = -(h_ratio * math.log2(h_ratio + 1e-10) + v_ratio * math.log2(v_ratio + 1e-10))
    else:
        entropy = 0
    
    return {
        'horizontal_moves': horizontal_moves,
        'vertical_moves': vertical_moves,
        'goal_piece_moves': goal_piece_moves,
        'move_sequence_entropy': entropy,
        'backtrack_count': len(solution_path) - len(position_hashes),  # Approximation
        'unique_positions': len(position_hashes)
    }

def calculate_difficulty_score(board_features, solution_features):
    """Calculate a composite difficulty score (0-10)"""
    if not solution_features.get('is_solvable', False):
        return 10.0
    
    # Weighted factors
    length_factor = min(solution_features.get('solution_length', 0) / 50.0, 1.0) * 3
    density_factor = board_features.get('board_density', 0) * 2
    blocking_factor = min(board_features.get('blocks_between_goal_target', 0) / 5.0, 1.0) * 2
    goal_distance_factor = min(board_features.get('goal_distance_to_target', 0) / 6.0, 1.0) * 2
    adjacent_factor = board_features.get('total_adjacent_blocks', 0) / 8.0 * 1
    
    difficulty = length_factor + density_factor + blocking_factor + goal_distance_factor + adjacent_factor
    return min(difficulty, 10.0)

def analyze_puzzle_complexity(board_features, solution_features):
    """
    Comprehensive complexity analysis for Klotski puzzles
    
    Returns:
        dict: Complete complexity metrics including difficulty, cognitive load, and algorithmic complexity
    """
    difficulty_score = calculate_difficulty_score(board_features, solution_features)
    
    return {
        'difficulty_score': round(difficulty_score, 2),
        'cognitive_load_estimate': round(difficulty_score * 1.1 + 0.5, 2),
        'algorithmic_complexity': round(difficulty_score * 0.9 + 0.3, 2),
        'complexity_factors': {
            'solution_length_factor': min(solution_features.get('solution_length', 0) / 50.0, 1.0) * 3,
            'density_factor': board_features.get('board_density', 0) * 2,
            'blocking_factor': min(board_features.get('blocks_between_goal_target', 0) / 5.0, 1.0) * 2,
            'goal_distance_factor': min(board_features.get('goal_distance_to_target', 0) / 6.0, 1.0) * 2,
            'adjacent_factor': board_features.get('total_adjacent_blocks', 0) / 8.0 * 1
        }
    }
```

### Usage Example:
```python
# Apply complexity analysis to your dataset
def add_complexity_features(df):
    complexity_metrics = []
    
    for _, row in df.iterrows():
        board_features = row.to_dict()
        solution_features = {'is_solvable': row['is_solvable'], 'solution_length': row['solution_length']}
        
        complexity = analyze_puzzle_complexity(board_features, solution_features)
        complexity_metrics.append(complexity)
    
    # Add new columns to dataframe
    df['difficulty_score'] = [c['difficulty_score'] for c in complexity_metrics]
    df['cognitive_load'] = [c['cognitive_load_estimate'] for c in complexity_metrics]
    df['algorithmic_complexity'] = [c['algorithmic_complexity'] for c in complexity_metrics]
    
    return df

# Enhanced dataset with complexity features
enhanced_df = add_complexity_features(df.copy())
```

### Additional Feature Engineering Ideas
- **Spatial Density Maps**: Block concentration in different board regions
- **Movement Potential**: How "free" each block is to move
- **Bottleneck Detection**: Identify constraining configurations
- **Symmetry Features**: Board rotation/reflection properties
- **Path Efficiency**: Ratio of optimal to actual solution length

## 🤖 Machine Learning (ML)

### 1. Classification Tasks

#### **Solvability Prediction**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Prepare features (exclude non-predictive columns)
feature_cols = [col for col in df.columns if col not in ['puzzle_id', 'timestamp', 'board_visual', 'is_solvable']]
X = df[feature_cols]
y = df['is_solvable']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Predictions and evaluation
y_pred = rf_classifier.predict(X_test)
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_classifier.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10))
```

#### **Difficulty Classification**
```python
# Create difficulty tiers
def create_difficulty_tiers(df):
    conditions = [
        (df['solution_length'] == 0) | (~df['is_solvable']),  # Unsolvable
        (df['solution_length'] > 0) & (df['solution_length'] <= 20),  # Easy
        (df['solution_length'] > 20) & (df['solution_length'] <= 50),  # Medium
        (df['solution_length'] > 50)  # Hard
    ]
    choices = ['Unsolvable', 'Easy', 'Medium', 'Hard']
    df['difficulty_tier'] = np.select(conditions, choices)
    return df

# Multi-class classification for difficulty prediction
enhanced_df = create_difficulty_tiers(enhanced_df)
```

### 2. Regression Tasks

#### **Solution Length Prediction**
```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Filter to solvable puzzles only
solvable_df = df[df['is_solvable'] == True]
X_reg = solvable_df[feature_cols]
y_reg = solvable_df['solution_length']

# Train regression model
gb_regressor = GradientBoostingRegressor(n_estimators=100, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2)

gb_regressor.fit(X_train_reg, y_train_reg)
y_pred_reg = gb_regressor.predict(X_test_reg)

print(f"MAE: {mean_absolute_error(y_test_reg, y_pred_reg):.2f}")
print(f"R² Score: {r2_score(y_test_reg, y_pred_reg):.3f}")
```

### 3. Advanced ML Applications

#### **Sequence Prediction** (with intermediate steps data)
```python
# For datasets generated with save_intermediate_steps=True
def prepare_sequence_data(steps_df):
    # Group by puzzle_id to get sequences
    sequences = []
    for puzzle_id in steps_df['puzzle_id'].unique():
        puzzle_steps = steps_df[steps_df['puzzle_id'] == puzzle_id].sort_values('step_number')
        if len(puzzle_steps) > 1:
            sequences.append(puzzle_steps)
    return sequences

# LSTM model for next-move prediction
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def build_sequence_model(input_dim, sequence_length):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(sequence_length, input_dim)),
        LSTM(32),
        Dense(64, activation='relu'),
        Dense(input_dim, activation='sigmoid')  # Output next board state features
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
```

#### **Clustering & Pattern Recognition**
```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Dimensionality reduction and clustering
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# K-means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X)

# Visualize clusters
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
plt.colorbar(scatter)
plt.title('Puzzle Clusters (PCA Visualization)')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
```

## 🚀 Getting Started

### 1. Generate Dataset
```bash
cd backend
python scripts/enhanced_data_generator.py
```

Interactive prompts will guide you through:
- Number of puzzles to generate
- Output format (CSV/JSON)
- Detailed progress display
- Intermediate solution steps (for sequence analysis)

### 2. Load and Explore
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your generated dataset
df = pd.read_csv('path/to/your/enhanced_dataset.csv')

# Quick overview
print(f"Dataset shape: {df.shape}")
print(f"Solvable rate: {df['is_solvable'].mean():.2%}")
print(f"Average solution length: {df[df['is_solvable']]['solution_length'].mean():.1f}")

# Beautiful board visualization
print("\nSample puzzle:")
print(df.iloc[0]['board_visual'])
```

### 3. Apply Complexity Analysis
```python
# Add advanced complexity features
enhanced_df = add_complexity_features(df)

# Explore new features
enhanced_df[['difficulty_score', 'cognitive_load', 'algorithmic_complexity']].describe()
```

## 📁 Project Structure

```
klotski-project/
├── backend/
│   ├── scripts/
│   │   ├── enhanced_data_generator.py      # Main dataset generator
│   │   ├── board_feature_extractor.py      # Feature calculation
│   │   ├── generate_random_klotski_board.py # Board generation
│   │   └── board_generator.py              # Visual rendering
│   ├── klotski/
│   │   ├── solver.py                       # BFS/A* solver
│   │   └── utils.py                        # Block classes
│   └── dashboard/
│       └── app.py                          # Interactive Streamlit dashboard
├── data/                                   # Generated datasets
├── notebooks/                              # Jupyter analysis notebooks
└── README.md                              # This file
```

## 🎯 ML Pipeline Recommendations

### Beginner Pipeline
1. **EDA** → Understand data distribution and patterns
2. **Binary Classification** → Predict solvable vs unsolvable
3. **Regression** → Predict solution length for solvable puzzles
4. **Feature Importance** → Identify key difficulty factors

### Advanced Pipeline  
1. **Feature Engineering** → Add complexity metrics and derived features
2. **Multi-class Classification** → Predict difficulty tiers
3. **Sequence Modeling** → Use LSTM for next-move prediction
4. **Clustering** → Discover puzzle archetypes
5. **Generation** → Create new puzzles with target difficulty

### Expert Pipeline
1. **Reinforcement Learning** → Train RL agents to solve puzzles optimally
2. **Generative Models** → VAE/GAN for novel puzzle creation
3. **Transfer Learning** → Apply to other sliding puzzle variants
4. **Multi-objective Optimization** → Balance difficulty, aesthetics, and solvability

## 📊 Expected Results

- **Solvability Prediction**: 85-95% accuracy with proper feature engineering
- **Solution Length**: R² > 0.7 for well-engineered features
- **Difficulty Classification**: 70-85% accuracy across difficulty tiers
- **Sequence Prediction**: Depends on sequence length and model complexity

## 🔧 Tech Stack

- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Machine Learning**: scikit-learn, XGBoost
- **Deep Learning**: TensorFlow/Keras, PyTorch
- **Interactive Dashboard**: Streamlit
- **Puzzle Engine**: Custom Python implementation with BFS/A* solver

## 📈 Future Extensions

- **Real-time Difficulty Adjustment**: ML-powered adaptive gameplay
- **Puzzle Generation API**: RESTful service for game integration  
- **Educational Tools**: Hint systems and tutorial generation
- **Mobile App Integration**: On-device ML for personalized experiences
- **Competition Platform**: Automated tournament bracket generation

---

**Happy Puzzle Solving!** 🧩✨

This comprehensive toolkit provides everything needed to dive deep into Klotski puzzle analysis, from basic statistical exploration to cutting-edge machine learning applications. The combination of rich features, beautiful visualizations, and powerful ML capabilities makes this perfect for both research and practical applications.