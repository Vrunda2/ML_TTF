import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

print("🔮 TTTF Prediction Demo")
print("=" * 50)


def load_and_predict():
    """Load model and make predictions on test data"""

    # Step 1: Load the trained model
    print("📂 Step 1: Loading Trained Model")
    print("-" * 35)

    try:
        model_package = joblib.load('tttf_gb_model.pkl')
        print("✅ Model loaded successfully!")

        # Display model info
        print(f"🎯 Target features: {model_package['target_names']}")
        print(f"📊 Input features: {len(model_package['feature_names'])}")

    except FileNotFoundError:
        print("❌ Model file not found! Please run the training pipeline first.")
        return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

    # Step 2: Load test data (automatically created during training)
    print("\n🧪 Step 2: Loading Test Data")
    print("-" * 32)

    try:
        test_data = joblib.load('test_data_sample.pkl')
        print("✅ Test data loaded successfully!")

        # Get the data components
        X_test_original = test_data['X_test_original']  # Original features
        y_test_floor = test_data['y_test_floor']  # True TTF values (floor)

        print(f"📊 Test samples: {len(X_test_original)}")
        print(f"🎯 Components to predict: {y_test_floor.shape[1]}")

    except FileNotFoundError:
        print("❌ Test data file not found! Please run the training pipeline first.")
        return None
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return None

    # Step 3: Make Predictions
    print("\n🔮 Step 3: Making Predictions")
    print("-" * 33)

    # Preprocess the test data using saved preprocessors
    model = model_package['model']
    scaler = model_package['scaler']
    imputer = model_package['imputer']
    label_encoders = model_package['label_encoders']
    feature_names = model_package['feature_names']
    target_names = model_package['target_names']

    # Prepare the data (same preprocessing as training)
    X_test_processed = X_test_original[feature_names].copy()

    # Handle categorical features
    for cat_feat in ['model']:
        if cat_feat in feature_names and cat_feat in label_encoders:
            le = label_encoders[cat_feat]
            values = X_test_processed[cat_feat].fillna('unknown').astype(str)
            values = values.apply(lambda x: x if x in le.classes_ else 'unknown')
            X_test_processed[cat_feat] = le.transform(values)

    # Handle numerical features
    numerical_features = [f for f in feature_names if f != 'model']
    if numerical_features:
        X_test_processed[numerical_features] = imputer.transform(X_test_processed[numerical_features])

    # Scale features
    X_test_scaled = scaler.transform(X_test_processed)

    # Make predictions
    predictions = model.predict(X_test_scaled)
    predictions_floor = np.floor(predictions)  # Convert to floor values

    print("✅ Predictions completed!")
    print(f"📊 Predicted {len(predictions)} samples")

    # Step 4: Evaluate Predictions
    print("\n📊 Step 4: Evaluating Predictions")
    print("-" * 36)

    # Calculate metrics for each component
    evaluation_results = []
    for i, target in enumerate(target_names):
        actual = y_test_floor.iloc[:, i]
        predicted = predictions_floor[:, i]

        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)

        evaluation_results.append({
            'Component': target,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2
        })

        print(f"🎯 {target}:")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE:  {mae:.4f}")
        print(f"   R²:   {r2:.4f}")

    # Overall performance
    avg_rmse = np.mean([r['RMSE'] for r in evaluation_results])
    avg_r2 = np.mean([r['R²'] for r in evaluation_results])
    print(f"\n📈 Overall Performance:")
    print(f"   Average RMSE: {avg_rmse:.4f}")
    print(f"   Average R²:   {avg_r2:.4f}")

    # Step 5: Show Sample Predictions
    print("\n🔍 Step 5: Sample Predictions")
    print("-" * 33)

    # Show 10 random samples
    sample_indices = np.random.choice(len(predictions), min(10, len(predictions)), replace=False)

    print("\nSample Predictions (Floor Values):")
    print("=" * 80)
    print(f"{'Index':<6} {'Component':<15} {'Actual':<8} {'Predicted':<10} {'Error':<8}")
    print("-" * 80)

    for idx in sample_indices[:5]:  # Show first 5 samples
        for i, target in enumerate(target_names):
            actual = y_test_floor.iloc[idx, i]
            predicted = predictions_floor[idx, i]
            error = abs(actual - predicted)

            print(f"{idx:<6} {target:<15} {actual:<8.0f} {predicted:<10.0f} {error:<8.1f}")
        print("-" * 80)

    # Step 6: Create Prediction Results DataFrame
    print("\n💾 Step 6: Saving Results")
    print("-" * 27)

    # Create comprehensive results
    results_df = pd.DataFrame()

    # Add original features (sample)
    sample_size = min(100, len(X_test_original))  # Save first 100 samples
    results_df = X_test_original.head(sample_size).copy()

    # Add predictions
    for i, target in enumerate(target_names):
        results_df[f'{target}_actual'] = y_test_floor.iloc[:sample_size, i].values
        results_df[f'{target}_predicted'] = predictions_floor[:sample_size, i]
        results_df[f'{target}_error'] = abs(results_df[f'{target}_actual'] - results_df[f'{target}_predicted'])

    # Save to CSV
    results_df.to_csv('prediction_results_sample.csv', index=False)
    print("✅ Results saved to 'prediction_results_sample.csv'")

    # Step 7: Create Visualization
    print("\n📊 Step 7: Creating Visualizations")
    print("-" * 36)

    # Create actual vs predicted plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('TTTF Predictions: Actual vs Predicted (Floor Values)', fontsize=16, fontweight='bold')

    for i, target in enumerate(target_names):
        if i < 4:  # Plot first 4 components
            row, col = divmod(i, 2)
            ax = axes[row, col]

            actual = y_test_floor.iloc[:, i]
            predicted = predictions_floor[:, i]

            # Scatter plot
            ax.scatter(actual, predicted, alpha=0.6, s=20)

            # Perfect prediction line
            min_val = min(actual.min(), predicted.min())
            max_val = max(actual.max(), predicted.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

            # Labels and title
            ax.set_xlabel('Actual TTF (weeks)')
            ax.set_ylabel('Predicted TTF (weeks)')
            ax.set_title(f'{target}\nR² = {evaluation_results[i]["R²"]:.3f}')
            ax.grid(True, alpha=0.3)
            ax.legend()

    # Remove empty subplots if less than 4 components
    for i in range(len(target_names), 4):
        row, col = divmod(i, 2)
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    plt.savefig('prediction_results_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("✅ Visualization saved as 'prediction_results_visualization.png'")

    print(f"\n🎉 Prediction Demo Completed Successfully!")
    print(f"📊 Evaluated {len(predictions)} test samples")
    print(f"📈 Average Model Performance: R² = {avg_r2:.4f}")
    print(f"💾 Results saved to 'prediction_results_sample.csv'")
    print("=" * 50)

    return results_df, evaluation_results


def create_sample_data_for_prediction():
    """Create a sample dataset for prediction testing"""
    print("\n🔧 Creating Sample Data for Prediction Testing")
    print("-" * 48)

    # Load the model first to get the valid model categories
    try:
        model_package = joblib.load('tttf_gb_model.pkl')
        label_encoders = model_package['label_encoders']

        # Get valid model categories from the label encoder
        if 'model' in label_encoders:
            valid_models = list(label_encoders['model'].classes_)
            print(f"📋 Valid model categories: {valid_models}")
        else:
            # Fallback to common model names
            valid_models = ['model_A', 'model_B', 'model_C']
            print(f"⚠️  Using default model categories: {valid_models}")

    except Exception as e:
        print(f"⚠️  Could not load model for validation, using default categories: {e}")
        valid_models = ['model_A', 'model_B', 'model_C']

    # Create synthetic sample data with realistic values
    np.random.seed(42)
    n_samples = 20

    sample_data = {
        'machineID': np.random.randint(1, 100, n_samples),
        'model': np.random.choice(valid_models, n_samples),  # Use valid model categories
        'age': np.random.randint(1, 10, n_samples),
        'volt': np.random.normal(170, 10, n_samples),
        'rotate': np.random.normal(1500, 200, n_samples),
        'pressure': np.random.normal(100, 15, n_samples),
        'vibration': np.random.normal(50, 10, n_samples),
        'error_count': np.random.poisson(2, n_samples),
        'failure_within_48h': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'days_since_comp1_maint': np.random.randint(1, 30, n_samples),
        'days_since_comp2_maint': np.random.randint(1, 30, n_samples),
        'days_since_comp3_maint': np.random.randint(1, 30, n_samples),
        'days_since_comp4_maint': np.random.randint(1, 30, n_samples)
    }

    sample_df = pd.DataFrame(sample_data)
    sample_df.to_csv('sample_data_for_prediction.csv', index=False)

    print(f"✅ Created sample data: {sample_df.shape}")
    print("💾 Saved as 'sample_data_for_prediction.csv'")
    print("\nSample data preview:")
    print(sample_df.head())

    return sample_df


def predict_custom_data(data_file='sample_data_for_prediction.csv'):
    """Make predictions on custom data file"""
    print(f"\n🔮 Making Predictions on Custom Data: {data_file}")
    print("-" * 60)

    try:
        # Load model
        model_package = joblib.load('tttf_gb_model.pkl')
        print("✅ Model loaded")

        # Load custom data
        custom_data = pd.read_csv(data_file)
        print(f"✅ Custom data loaded: {custom_data.shape}")

        # Preprocess and predict
        model = model_package['model']
        scaler = model_package['scaler']
        imputer = model_package['imputer']
        label_encoders = model_package['label_encoders']
        feature_names = model_package['feature_names']
        target_names = model_package['target_names']

        # Prepare data
        X_custom = custom_data[feature_names].copy()

        # Handle categorical features with improved error handling
        for cat_feat in ['model']:
            if cat_feat in feature_names and cat_feat in label_encoders:
                le = label_encoders[cat_feat]
                print(f"📋 Valid {cat_feat} categories: {list(le.classes_)}")

                # Check for unknown categories and handle them
                values = X_custom[cat_feat].fillna('').astype(str)
                unknown_values = set(values) - set(le.classes_)

                if unknown_values:
                    print(f"⚠️  Unknown {cat_feat} values found: {unknown_values}")
                    print(f"🔄 Replacing with most common category: {le.classes_[0]}")

                    # Replace unknown values with the first (most common) category
                    values = values.apply(lambda x: x if x in le.classes_ else le.classes_[0])

                X_custom[cat_feat] = le.transform(values)

        # Handle numerical features
        numerical_features = [f for f in feature_names if f != 'model']
        if numerical_features:
            X_custom[numerical_features] = imputer.transform(X_custom[numerical_features])

        # Scale and predict
        X_custom_scaled = scaler.transform(X_custom)
        predictions = model.predict(X_custom_scaled)
        predictions_floor = np.floor(predictions)

        # Create results
        results = custom_data.copy()
        for i, target in enumerate(target_names):
            results[f'{target}_predicted'] = predictions_floor[:, i]

        # Save results
        output_file = data_file.replace('.csv', '_with_predictions.csv')
        results.to_csv(output_file, index=False)

        print("✅ Predictions completed!")
        print(f"💾 Results saved to: {output_file}")

        # Show sample predictions
        print("\nSample Predictions:")
        print("-" * 40)
        for i in range(min(5, len(results))):
            print(f"\nSample {i + 1} (Machine ID: {results.iloc[i]['machineID']}):")
            for target in target_names:
                pred_col = f'{target}_predicted'
                print(f"  {target}: {results.iloc[i][pred_col]:.0f} weeks")

        return results

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        print(f"📋 Full error details:\n{traceback.format_exc()}")
        return None


def validate_model_compatibility():
    """Validate that the model and data are compatible"""
    print("\n🔍 Model Compatibility Check")
    print("-" * 30)

    try:
        # Load model
        model_package = joblib.load('tttf_gb_model.pkl')
        print("✅ Model loaded successfully")

        # Check components
        required_components = ['model', 'scaler', 'imputer', 'label_encoders', 'feature_names', 'target_names']
        missing_components = [comp for comp in required_components if comp not in model_package]

        if missing_components:
            print(f"⚠️  Missing model components: {missing_components}")
        else:
            print("✅ All required model components present")

        # Display model information
        print(f"📊 Feature names: {model_package['feature_names']}")
        print(f"🎯 Target names: {model_package['target_names']}")

        if 'model' in model_package['label_encoders']:
            print(f"🏷️  Valid model categories: {list(model_package['label_encoders']['model'].classes_)}")

        return True

    except Exception as e:
        print(f"❌ Model validation failed: {e}")
        return False


if __name__ == "__main__":
    # First validate model compatibility
    if not validate_model_compatibility():
        print("❌ Model validation failed. Please check your model file.")
        exit(1)

    print("\nChoose an option:")
    print("1. Use test data created during training (Recommended)")
    print("2. Create sample data and predict")
    print("3. Both options")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == '1':
        load_and_predict()
    elif choice == '2':
        sample_data = create_sample_data_for_prediction()
        if sample_data is not None:
            predict_custom_data('sample_data_for_prediction.csv')
    elif choice == '3':
        print("\n" + "=" * 60)
        print("OPTION 1: Using Test Data from Training")
        print("=" * 60)
        load_and_predict()

        print("\n" + "=" * 60)
        print("OPTION 2: Creating and Using Sample Data")
        print("=" * 60)
        sample_data = create_sample_data_for_prediction()
        if sample_data is not None:
            predict_custom_data('sample_data_for_prediction.csv')
    else:
        print("Invalid choice. Running Option 1 by default...")
        load_and_predict()