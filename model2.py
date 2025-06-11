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
import pickle
import joblib
import warnings

warnings.filterwarnings('ignore')

print("🚀 Improved Gradient Boosting TTTF Prediction Pipeline")
print("=" * 70)

# Step 1: Load and Sample Data Strategically
print("\n📊 Step 1: Loading and Stratified Sampling (200K)")
print("-" * 50)

df = pd.read_csv('Data/TTF_Dataset_Weeks.csv')
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
    target_cols = ['ttf_comp1_weeks', 'ttf_comp2_weeks', 'ttf_comp3_weeks', 'ttf_comp4_weeks']
    available_target_cols = [col for col in target_cols if col in df.columns]

    if available_target_cols:
        avg_tttf = df[available_target_cols].mean(axis=1)
        strata['tttf_range'] = pd.cut(avg_tttf, bins=3, labels=['short', 'medium', 'long'])
    else:
        strata['tttf_range'] = 'default'

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
sample_size = min(200000, len(df))
sss = StratifiedShuffleSplit(n_splits=1, train_size=sample_size, random_state=42)
sample_idx, _ = next(sss.split(df, strata_labels))

df_sample = df.iloc[sample_idx].copy()
print(f"✅ Stratified sample created: {df_sample.shape}")
print(f"📈 Sample represents {df_sample.shape[0] / df.shape[0] * 100:.1f}% of original data")

# Step 2: Define Features and Targets
print("\n🎯 Step 2: Feature and Target Definition")
print("-" * 40)

# Define target features
target_features = ['ttf_comp1_weeks', 'ttf_comp2_weeks', 'ttf_comp3_weeks', 'ttf_comp4_weeks']
available_targets = [col for col in target_features if col in df_sample.columns]

# Feature groups
sensor_features = ['volt', 'rotate', 'pressure', 'vibration']
operational_features = ['machineID', 'model', 'age', 'error_count', 'failure_within_48h']
maintenance_features = ['days_since_comp1_maint', 'days_since_comp2_maint',
                        'days_since_comp3_maint', 'days_since_comp4_maint']

# Get available features
available_features = []
for feature_group in [sensor_features, operational_features, maintenance_features]:
    available_features.extend([f for f in feature_group if f in df_sample.columns])

print(f"🎯 Available target features: {available_targets}")
print(f"📋 Available input features: {len(available_features)}")

# Step 3: Train-Test Split BEFORE preprocessing
print("\n📊 Step 3: Train-Test Split (80-20)")
print("-" * 35)

# Prepare initial features and targets
X_initial = df_sample[available_features].copy()
y_initial = df_sample[available_targets].copy()

# Stratify based on failure risk if available
stratify_col = df_sample['failure_within_48h'] if 'failure_within_48h' in df_sample.columns else None

X_train, X_test, y_train, y_test = train_test_split(
    X_initial, y_initial,
    test_size=0.2,
    random_state=42,
    stratify=stratify_col
)

print(f"📈 Train set: {X_train.shape[0]} samples ({X_train.shape[0] / (X_train.shape[0] + X_test.shape[0]) * 100:.1f}%)")
print(f"📉 Test set: {X_test.shape[0]} samples ({X_test.shape[0] / (X_train.shape[0] + X_test.shape[0]) * 100:.1f}%)")

# Step 4: Data Preprocessing (Applied separately to train and test)
print("\n🔄 Step 4: Data Preprocessing")
print("-" * 32)

# Separate numerical and categorical features
categorical_features = ['model']
numerical_features = [f for f in available_features if f not in categorical_features]

print(f"📊 Categorical features: {len(categorical_features)}")
print(f"🔢 Numerical features: {len(numerical_features)}")

# Initialize preprocessors
label_encoders = {}
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

# Process training data
X_train_processed = X_train.copy()
X_test_processed = X_test.copy()

# Handle categorical variables
print("Encoding categorical variables...")
for cat_feat in categorical_features:
    if cat_feat in available_features:
        # Fit label encoder on training data
        le = LabelEncoder()

        # Handle missing values in training data
        train_values = X_train_processed[cat_feat].fillna('unknown').astype(str)
        le.fit(train_values)
        X_train_processed[cat_feat] = le.transform(train_values)

        # Transform test data (handle unseen categories)
        test_values = X_test_processed[cat_feat].fillna('unknown').astype(str)
        # Handle unseen categories by mapping them to 'unknown'
        test_values = test_values.apply(lambda x: x if x in le.classes_ else 'unknown')
        X_test_processed[cat_feat] = le.transform(test_values)

        # Store encoder for later use
        label_encoders[cat_feat] = le
        print(f"✅ Encoded {cat_feat}")

# Handle missing values for numerical features
if numerical_features:
    print("Handling missing values for numerical features...")
    # Fit imputer on training data
    X_train_processed[numerical_features] = imputer.fit_transform(X_train_processed[numerical_features])
    # Transform test data
    X_test_processed[numerical_features] = imputer.transform(X_test_processed[numerical_features])
    print("✅ Missing values handled")

# Convert TTF to floor values AFTER split
print("\nConverting TTF to floor values...")
y_train_floor = np.floor(y_train)
y_test_floor = np.floor(y_test)

print("Original vs Floor TTF statistics:")
for i, col in enumerate(available_targets):
    orig_mean = y_train.iloc[:, i].mean()
    floor_mean = y_train_floor.iloc[:, i].mean()
    print(f"{col}: Original={orig_mean:.2f}, Floor={floor_mean:.2f}")

# Feature scaling
print("\nScaling features...")
X_train_scaled = scaler.fit_transform(X_train_processed)
X_test_scaled = scaler.transform(X_test_processed)
print("✅ Features scaled using StandardScaler")

# Step 5: Model Training
print("\n🚀 Step 5: Gradient Boosting Model Training")
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

# Create and train model
print("\nTraining MultiOutput Gradient Boosting Regressor...")
gb_model = MultiOutputRegressor(GradientBoostingRegressor(**gb_params))
gb_model.fit(X_train_scaled, y_train_floor)
print("✅ Model training completed!")

# Step 6: Model Evaluation
print("\n📊 Step 6: Model Evaluation on Test Set")
print("-" * 40)

# Make predictions on test set
y_train_pred = gb_model.predict(X_train_scaled)
y_test_pred = gb_model.predict(X_test_scaled)

print("🔮 Predictions completed on test set!")
print(f"📊 Test predictions shape: {y_test_pred.shape}")

# Calculate metrics for each target
results = []
for i, target in enumerate(available_targets):
    # Training metrics
    train_mse = mean_squared_error(y_train_floor.iloc[:, i], y_train_pred[:, i])
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train_floor.iloc[:, i], y_train_pred[:, i])
    train_r2 = r2_score(y_train_floor.iloc[:, i], y_train_pred[:, i])

    # Test metrics
    test_mse = mean_squared_error(y_test_floor.iloc[:, i], y_test_pred[:, i])
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test_floor.iloc[:, i], y_test_pred[:, i])
    test_r2 = r2_score(y_test_floor.iloc[:, i], y_test_pred[:, i])

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

# Step 7: Save Model and Preprocessors
print("\n💾 Step 7: Saving Model and Preprocessors")
print("-" * 42)

# Create model package
model_package = {
    'model': gb_model,
    'scaler': scaler,
    'imputer': imputer,
    'label_encoders': label_encoders,
    'feature_names': available_features,
    'target_names': available_targets,
    'model_params': gb_params,
    'performance_metrics': results_df.to_dict('records')
}

# Save using joblib (recommended for sklearn models)
joblib.dump(model_package, 'tttf_gb_model.pkl')
print("✅ Model package saved as 'tttf_gb_model.pkl'")

# Also save as pickle for compatibility
with open('tttf_gb_model_backup.pkl', 'wb') as f:
    pickle.dump(model_package, f)
print("✅ Backup model saved as 'tttf_gb_model_backup.pkl'")

# Save test data for validation
test_data_package = {
    'X_test_original': X_test,
    'X_test_processed': X_test_processed,
    'X_test_scaled': X_test_scaled,
    'y_test_original': y_test,
    'y_test_floor': y_test_floor,
    'y_test_predictions': y_test_pred
}

joblib.dump(test_data_package, 'test_data_sample.pkl')
print("✅ Test data sample saved as 'test_data_sample.pkl'")

# Step 8: Sample Predictions Display
print("\n🔮 Step 8: Sample Test Predictions")
print("-" * 38)

# Show some prediction examples
sample_indices = np.random.choice(len(X_test), min(10, len(X_test)), replace=False)
print("\nSample Predictions (Floor Values):")
print("Index | Component | Actual | Predicted | Error")
print("-" * 50)

for idx in sample_indices[:5]:  # Show first 5 samples
    for i, target in enumerate(available_targets):
        actual = y_test_floor.iloc[idx, i]
        predicted = y_test_pred[idx, i]
        error = abs(actual - predicted)
        print(f"{idx:5d} | {target:15s} | {actual:6.0f} | {predicted:9.1f} | {error:5.1f}")
    print("-" * 50)

print(f"\n✅ Training Pipeline Completed Successfully!")
print(f"📊 Model trained on {X_train.shape[0]:,} samples (80%)")
print(f"🧪 Tested on {X_test.shape[0]:,} samples (20%)")
print(f"🎯 Predicting floor values for {len(available_targets)} components")
print(f"📈 Average Test R²: {results_df['Test_R2'].mean():.4f}")
print(f"💾 Model and preprocessors saved successfully")
print("=" * 70)