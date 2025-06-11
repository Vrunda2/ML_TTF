import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score
import warnings

warnings.filterwarnings('ignore')

print("🚀 Gradient Boosting TTTF Prediction Pipeline (200K Stratified Sample)")
print("=" * 70)

# Step 1: Load and Sample Data Strategically
print("\n📊 Step 1: Loading and Stratified Sampling (200K)")
print("-" * 50)

df = pd.read_csv('Data/TTF_Dataset_Weeks1.csv')
print(f"✅ Full dataset loaded: {df.shape}")


# Create stratification bins for balanced sampling
def create_strata(df):
    """Create stratification categories for balanced sampling"""
    strata = pd.DataFrame()

    # Machine model stratification
    if 'model' in df.columns:
        strata['model'] = df['model']
    else:
        strata['model'] = 'default'

    # Failure risk stratification
    if 'failure_within_48h' in df.columns:
        strata['risk'] = df['failure_within_48h']
    else:
        strata['risk'] = 0

    # TTTF range stratification (short/medium/long term)
    avg_tttf = df[['ttf_comp1_weeks', 'ttf_comp2_weeks', 'ttf_comp3_weeks', 'ttf_comp4_weeks']].mean(axis=1)
    strata['tttf_range'] = pd.cut(avg_tttf, bins=3, labels=['short', 'medium', 'long'])

    # Age stratification
    if 'age' in df.columns:
        strata['age_group'] = pd.cut(df['age'], bins=3, labels=['new', 'mid', 'old'])
    else:
        strata['age_group'] = 'default'

    # Combine all strata
    strata['combined'] = (strata['model'].astype(str) + '_' +
                          strata['risk'].astype(str) + '_' +
                          strata['tttf_range'].astype(str) + '_' +
                          strata['age_group'].astype(str))

    return strata['combined']


# Create stratification labels
strata_labels = create_strata(df)
print(f"📊 Created {strata_labels.nunique()} stratification groups")

# Stratified sampling to get 200k representative samples
sample_size = min(200000, len(df))  # Use 200k or full dataset if smaller
sss = StratifiedShuffleSplit(n_splits=1, train_size=sample_size, random_state=42)
sample_idx, _ = next(sss.split(df, strata_labels))

df_sample = df.iloc[sample_idx].copy()
print(f"✅ Stratified sample created: {df_sample.shape}")
print(f"📈 Sample represents {df_sample.shape[0] / df.shape[0] * 100:.1f}% of original data")

# Step 2: Convert TTF to Floor Values
print("\n🔢 Step 2: Converting TTF to Floor Values")
print("-" * 40)

target_features = ['ttf_comp1_weeks', 'ttf_comp2_weeks', 'ttf_comp3_weeks', 'ttf_comp4_weeks']

print("Original TTF statistics:")
for col in target_features:
    if col in df_sample.columns:
        original_stats = df_sample[col].describe()
        print(f"{col}: Mean={original_stats['mean']:.2f}, Std={original_stats['std']:.2f}")

# Convert to floor values
print("\nConverting to floor values...")
for col in target_features:
    if col in df_sample.columns:
        df_sample[col + '_floor'] = np.floor(df_sample[col])

# Update target features to use floor versions
target_features_floor = [col + '_floor' for col in target_features if col in df_sample.columns]
print(f"✅ Created floor versions: {target_features_floor}")

print("\nFloor TTF statistics:")
for col in target_features_floor:
    floor_stats = df_sample[col].describe()
    print(f"{col}: Mean={floor_stats['mean']:.2f}, Std={floor_stats['std']:.2f}")

# Step 3: Define Features
print("\n🎯 Step 3: Feature Engineering")
print("-" * 35)

# Feature groups
sensor_features = ['volt', 'rotate', 'pressure', 'vibration']
operational_features = ['machineID', 'model', 'age', 'error_count', 'failure_within_48h']
maintenance_features = ['days_since_comp1_maint', 'days_since_comp2_maint',
                        'days_since_comp3_maint', 'days_since_comp4_maint']

# Get available features
available_features = []
for feature_group in [sensor_features, operational_features, maintenance_features]:
    available_features.extend([f for f in feature_group if f in df_sample.columns])

print(f"🔧 Sensor features: {len([f for f in sensor_features if f in df_sample.columns])}")
print(f"⚙  Operational features: {len([f for f in operational_features if f in df_sample.columns])}")
print(f"🔨 Maintenance features: {len([f for f in maintenance_features if f in df_sample.columns])}")
print(f"🎯 Target features (floor): {len(target_features_floor)}")
print(f"📋 Total available features: {len(available_features)}")

# Step 4: Data Preprocessing
print("\n🔄 Step 4: Data Preprocessing")
print("-" * 32)

# Separate numerical and categorical features
categorical_features = ['model']
numerical_features = [f for f in available_features if f not in categorical_features]

print(f"📊 Categorical features: {len(categorical_features)}")
print(f"🔢 Numerical features: {len(numerical_features)}")

# Handle categorical variables first
print("Encoding categorical variables...")
for cat_feat in categorical_features:
    if cat_feat in available_features:
        # Fill missing categorical values with mode
        if df_sample[cat_feat].isnull().any():
            mode_value = df_sample[cat_feat].mode()[0] if not df_sample[cat_feat].mode().empty else 'unknown'
            df_sample[cat_feat] = df_sample[cat_feat].fillna(mode_value)

        # Label encode
        le = LabelEncoder()
        df_sample[cat_feat] = le.fit_transform(df_sample[cat_feat].astype(str))
        print(f"✅ Encoded {cat_feat}")

# Handle missing values for numerical features
if numerical_features:
    print("Handling missing values for numerical features...")
    imputer = SimpleImputer(strategy='median')
    df_sample[numerical_features] = imputer.fit_transform(df_sample[numerical_features])
    print("✅ Missing values handled")

# Prepare features and targets
X = df_sample[available_features]
y = df_sample[target_features_floor]

print(f"✅ Features shape: {X.shape}")
print(f"✅ Targets shape: {y.shape}")

# Step 5: Train-Test Split
print("\n📊 Step 5: Train-Test Split")
print("-" * 30)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=df_sample['failure_within_48h']
)

print(f"📈 Train set: {X_train.shape}")
print(f"📉 Test set: {X_test.shape}")

# Step 6: Feature Scaling
print("\n📏 Step 6: Feature Scaling")
print("-" * 28)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Features scaled using StandardScaler")

# Step 7: Gradient Boosting Model Training
print("\n🚀 Step 7: Gradient Boosting Model Training")
print("-" * 45)

# Configure Gradient Boosting parameters
gb_params = {
    'n_estimators': 200,
    'learning_rate': 0.1,
    'max_depth': 6,
    'min_samples_split': 5,
    'min_samples_leaf': 3,
    'subsample': 0.8,
    'random_state': 42,
    'verbose': 1
}

print("Gradient Boosting Parameters:")
for param, value in gb_params.items():
    print(f"  {param}: {value}")

# Create MultiOutput Gradient Boosting model
print("\nTraining MultiOutput Gradient Boosting Regressor...")
gb_model = MultiOutputRegressor(GradientBoostingRegressor(**gb_params))

# Train the model
gb_model.fit(X_train_scaled, y_train)
print("✅ Model training completed!")

# Step 8: Model Evaluation
print("\n📊 Step 8: Model Evaluation")
print("-" * 32)

# Make predictions
y_train_pred = gb_model.predict(X_train_scaled)
y_test_pred = gb_model.predict(X_test_scaled)

# Calculate metrics for each target
results = []
for i, target in enumerate(target_features_floor):
    # Training metrics
    train_mse = mean_squared_error(y_train.iloc[:, i], y_train_pred[:, i])
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train.iloc[:, i], y_train_pred[:, i])
    train_r2 = r2_score(y_train.iloc[:, i], y_train_pred[:, i])

    # Test metrics
    test_mse = mean_squared_error(y_test.iloc[:, i], y_test_pred[:, i])
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test.iloc[:, i], y_test_pred[:, i])
    test_r2 = r2_score(y_test.iloc[:, i], y_test_pred[:, i])

    results.append({
        'Target': target,
        'Train_RMSE': train_rmse,
        'Test_RMSE': test_rmse,
        'Train_MAE': train_mae,
        'Test_MAE': test_mae,
        'Train_R2': train_r2,
        'Test_R2': test_r2
    })

# Display results
results_df = pd.DataFrame(results)
print("\n🎯 Model Performance Summary:")
print(results_df.round(4))

# Overall performance
print(f"\n📊 Overall Performance:")
print(f"Average Test RMSE: {results_df['Test_RMSE'].mean():.4f}")
print(f"Average Test MAE: {results_df['Test_MAE'].mean():.4f}")
print(f"Average Test R²: {results_df['Test_R2'].mean():.4f}")

# Step 9: Feature Importance Analysis
print("\n🎯 Step 9: Feature Importance Analysis")
print("-" * 40)

# Get feature importance for each component
feature_importance_data = []
for i, estimator in enumerate(gb_model.estimators_):
    importance = estimator.feature_importances_
    for j, feature in enumerate(available_features):
        feature_importance_data.append({
            'Component': target_features_floor[i],
            'Feature': feature,
            'Importance': importance[j]
        })

importance_df = pd.DataFrame(feature_importance_data)

# Top features overall
top_features = importance_df.groupby('Feature')['Importance'].mean().sort_values(ascending=False).head(10)
print("\n🏆 Top 10 Most Important Features (Average):")
for feature, importance in top_features.items():
    print(f"  {feature}: {importance:.4f}")

# Step 10: Visualizations
print("\n📈 Step 10: Creating Visualizations")
print("-" * 38)

# Set up the plotting style
plt.style.use('default')
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Gradient Boosting TTTF Prediction Results (Floor Values)', fontsize=16, fontweight='bold')

# Plot 1: Actual vs Predicted for each component
for i, target in enumerate(target_features_floor):
    row, col = divmod(i, 3)
    if row < 2 and col < 3:
        ax = axes[row, col]
        ax.scatter(y_test.iloc[:, i], y_test_pred[:, i], alpha=0.6, s=1)
        ax.plot([y_test.iloc[:, i].min(), y_test.iloc[:, i].max()],
                [y_test.iloc[:, i].min(), y_test.iloc[:, i].max()], 'r--', lw=2)
        ax.set_xlabel('Actual TTF (weeks)')
        ax.set_ylabel('Predicted TTF (weeks)')
        ax.set_title(f'{target}\nR² = {results_df.iloc[i]["Test_R2"]:.3f}')
        ax.grid(True, alpha=0.3)

# Plot 5: Model Performance Comparison
ax = axes[1, 2]
metrics = ['Test_RMSE', 'Test_MAE']
x = np.arange(len(target_features_floor))
width = 0.35

for i, metric in enumerate(metrics):
    values = results_df[metric].values
    ax.bar(x + i * width, values, width, label=metric.replace('Test_', ''))

ax.set_xlabel('Components')
ax.set_ylabel('Error Value')
ax.set_title('Model Performance by Component')
ax.set_xticks(x + width / 2)
ax.set_xticklabels([f'Comp{i + 1}' for i in range(len(target_features_floor))])
ax.legend()
ax.grid(True, alpha=0.3)

# Remove empty subplot
if len(target_features_floor) < 6:
    for i in range(len(target_features_floor), 6):
        row, col = divmod(i, 3)
        if row < 2:
            fig.delaxes(axes[row, col])

plt.tight_layout()
plt.show()

# Feature importance plot
plt.figure(figsize=(12, 8))
top_features_plot = top_features.head(15)
plt.barh(range(len(top_features_plot)), top_features_plot.values)
plt.yticks(range(len(top_features_plot)), top_features_plot.index)
plt.xlabel('Average Feature Importance')
plt.title('Top 15 Most Important Features (Gradient Boosting)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Step 11: Cross-Validation
print("\n🔄 Step 11: Cross-Validation")
print("-" * 32)

print("Performing 5-fold cross-validation...")
cv_scores = []
for target in target_features_floor:
    gb_single = GradientBoostingRegressor(**gb_params)
    scores = cross_val_score(gb_single, X_train_scaled, y_train[target],
                             cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-scores)
    cv_scores.append(cv_rmse.mean())
    print(f"{target}: CV RMSE = {cv_rmse.mean():.4f} (±{cv_rmse.std():.4f})")

print(f"\n📊 Average CV RMSE: {np.mean(cv_scores):.4f}")

# Step 12: Prediction Examples
print("\n🔮 Step 12: Prediction Examples")
print("-" * 35)

# Show some prediction examples
sample_indices = np.random.choice(len(X_test), 5, replace=False)
print("\nSample Predictions (Floor Values):")
print("Index | Actual TTF (weeks) | Predicted TTF (weeks)")
print("-" * 60)

for idx in sample_indices:
    actual = y_test.iloc[idx].values
    predicted = y_test_pred[idx]
    print(f"{idx:5d} | {actual} | {predicted.round()}")

print("\n✅ Gradient Boosting Pipeline Completed!")
print(f"📊 Model trained on {X_train.shape[0]:,} samples")
print(f"🎯 Predicting floor values for {len(target_features_floor)} components")
print(f"📈 Average Test R²: {results_df['Test_R2'].mean():.4f}")
print("=" * 70)