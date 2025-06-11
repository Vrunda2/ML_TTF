import pandas as pd
import numpy as np
import joblib
from datetime import datetime

print("🔮 TTTF Manual Prediction Tool")
print("=" * 50)


def load_model_info():
    """Load model and display information about required inputs"""
    try:
        model_package = joblib.load('tttf_gb_model.pkl')
        print("✅ Model loaded successfully!")

        feature_names = model_package['feature_names']
        target_names = model_package['target_names']
        label_encoders = model_package['label_encoders']

        print(f"🎯 This model predicts: {target_names}")
        print(f"📊 Required input features ({len(feature_names)}):")

        for i, feature in enumerate(feature_names, 1):
            if feature in label_encoders:
                valid_categories = list(label_encoders[feature].classes_)
                print(f"  {i:2d}. {feature} (categorical): {valid_categories}")
            else:
                print(f"  {i:2d}. {feature} (numerical)")

        return model_package

    except FileNotFoundError:
        print("❌ Model file 'tttf_gb_model.pkl' not found!")
        print("   Please run the training pipeline first.")
        return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


def get_user_input(model_package):
    """Get manual input from user for all required features"""
    feature_names = model_package['feature_names']
    label_encoders = model_package['label_encoders']

    print("\n📝 Please enter values for the following features:")
    print("-" * 50)

    input_data = {}

    for feature in feature_names:
        while True:
            try:
                if feature in label_encoders:
                    # Categorical feature
                    valid_categories = list(label_encoders[feature].classes_)
                    print(f"\n🏷️  {feature} (choose from: {valid_categories})")
                    value = input(f"   Enter {feature}: ").strip()

                    if value in valid_categories:
                        input_data[feature] = value
                        print(f"   ✅ {feature} = {value}")
                        break
                    else:
                        print(f"   ❌ Invalid value! Must be one of: {valid_categories}")

                else:
                    # Numerical feature
                    print(f"\n🔢 {feature} (numerical value)")

                    # Provide some guidance for common features
                    if 'age' in feature.lower():
                        print("   💡 Typical range: 1-10 years")
                    elif 'volt' in feature.lower():
                        print("   💡 Typical range: 150-190 volts")
                    elif 'rotate' in feature.lower():
                        print("   💡 Typical range: 1200-1800 RPM")
                    elif 'pressure' in feature.lower():
                        print("   💡 Typical range: 80-120 PSI")
                    elif 'vibration' in feature.lower():
                        print("   💡 Typical range: 30-70 units")
                    elif 'error' in feature.lower():
                        print("   💡 Typical range: 0-10 errors")
                    elif 'failure' in feature.lower():
                        print("   💡 Enter 0 (no) or 1 (yes)")
                    elif 'days_since' in feature.lower():
                        print("   💡 Typical range: 1-30 days")
                    elif 'machineID' in feature.lower():
                        print("   💡 Any machine identifier number")

                    value = input(f"   Enter {feature}: ").strip()

                    # Convert to appropriate numeric type
                    if 'failure' in feature.lower() or feature.lower() in ['machineid', 'error_count']:
                        value = int(float(value))
                    else:
                        value = float(value)

                    input_data[feature] = value
                    print(f"   ✅ {feature} = {value}")
                    break

            except ValueError:
                print("   ❌ Invalid input! Please enter a valid number.")
            except KeyboardInterrupt:
                print("\n\n👋 Prediction cancelled by user.")
                return None
            except Exception as e:
                print(f"   ❌ Error: {e}. Please try again.")

    return input_data


def predict_single_sample(model_package, input_data):
    """Make prediction for a single sample"""
    print("\n🔮 Making Prediction...")
    print("-" * 25)

    try:
        # Extract model components
        model = model_package['model']
        scaler = model_package['scaler']
        imputer = model_package['imputer']
        label_encoders = model_package['label_encoders']
        feature_names = model_package['feature_names']
        target_names = model_package['target_names']

        # Create DataFrame from input
        sample_df = pd.DataFrame([input_data])

        # Ensure correct feature order
        X_sample = sample_df[feature_names].copy()

        # Process categorical features
        for cat_feat in ['model']:
            if cat_feat in feature_names and cat_feat in label_encoders:
                le = label_encoders[cat_feat]
                X_sample[cat_feat] = le.transform(X_sample[cat_feat])

        # Process numerical features
        numerical_features = [f for f in feature_names if f not in label_encoders]
        if numerical_features:
            X_sample[numerical_features] = imputer.transform(X_sample[numerical_features])

        # Scale features
        X_scaled = scaler.transform(X_sample)

        # Make prediction
        prediction = model.predict(X_scaled)[0]  # Get first (and only) prediction
        prediction_floor = np.floor(prediction)

        print("✅ Prediction completed!")

        return prediction, prediction_floor

    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        print(f"📋 Error details:\n{traceback.format_exc()}")
        return None, None


def display_results(input_data, prediction, prediction_floor, target_names):
    """Display the prediction results in a nice format"""
    print("\n" + "=" * 60)
    print("🎯 PREDICTION RESULTS")
    print("=" * 60)

    # Display input summary
    print("\n📊 INPUT SUMMARY:")
    print("-" * 20)
    for feature, value in input_data.items():
        if isinstance(value, float):
            print(f"   {feature:<25}: {value:.2f}")
        else:
            print(f"   {feature:<25}: {value}")

    # Display predictions
    print("\n🔮 PREDICTED TIME TO FAILURE:")
    print("-" * 35)
    for i, target in enumerate(target_names):
        exact_weeks = prediction[i]
        floor_weeks = prediction_floor[i]

        print(f"   {target:<20}: {floor_weeks:.0f} weeks ({exact_weeks:.2f} exact)")

        # Convert to human-readable time
        if floor_weeks == 0:
            time_desc = "⚠️  IMMEDIATE ATTENTION NEEDED"
        elif floor_weeks <= 2:
            time_desc = "🔴 Very Soon (Critical)"
        elif floor_weeks <= 4:
            time_desc = "🟡 Soon (Monitor Closely)"
        elif floor_weeks <= 8:
            time_desc = "🟢 Normal Schedule"
        else:
            time_desc = "✅ Long Term"

        print(f"   {'':<20}   → {time_desc}")

    # Overall assessment
    min_weeks = min(prediction_floor)
    print(f"\n🚨 NEXT MAINTENANCE PRIORITY:")
    print("-" * 30)
    if min_weeks == 0:
        print("   ⚠️  IMMEDIATE maintenance required!")
    elif min_weeks <= 2:
        print("   🔴 Schedule maintenance within 2 weeks")
    elif min_weeks <= 4:
        print("   🟡 Schedule maintenance within 1 month")
    else:
        print("   ✅ Normal maintenance schedule")

    print("\n" + "=" * 60)


def save_prediction_log(input_data, prediction, prediction_floor, target_names):
    """Save prediction to a log file"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Create log entry
        log_entry = {
            'timestamp': timestamp,
            **input_data
        }

        # Add predictions
        for i, target in enumerate(target_names):
            log_entry[f'{target}_predicted'] = prediction_floor[i]
            log_entry[f'{target}_exact'] = prediction[i]

        # Save to CSV
        log_df = pd.DataFrame([log_entry])

        try:
            # Try to append to existing log
            existing_log = pd.read_csv('prediction_log.csv')
            updated_log = pd.concat([existing_log, log_df], ignore_index=True)
            updated_log.to_csv('prediction_log.csv', index=False)
        except FileNotFoundError:
            # Create new log file
            log_df.to_csv('prediction_log.csv', index=False)

        print(f"💾 Prediction logged to 'prediction_log.csv'")

    except Exception as e:
        print(f"⚠️  Could not save to log: {e}")


def main():
    """Main function to run the manual prediction tool"""

    # Load model and show info
    model_package = load_model_info()
    if model_package is None:
        return

    while True:
        print("\n" + "=" * 50)
        print("Choose an option:")
        print("1. 📝 Enter values manually")
        print("2. 🎲 Use example values (quick test)")
        print("3. 📋 Show feature requirements")
        print("4. 👋 Exit")

        choice = input("\nEnter choice (1-4): ").strip()

        if choice == '1':
            # Manual input
            input_data = get_user_input(model_package)
            if input_data is None:
                continue

        elif choice == '2':
            # Example values
            print("\n🎲 Using example values for quick test...")

            # Get valid model category
            if 'model' in model_package['label_encoders']:
                valid_models = list(model_package['label_encoders']['model'].classes_)
                example_model = valid_models[0]
            else:
                example_model = 'model_A'

            input_data = {
                'machineID': 12345,
                'model': example_model,
                'age': 5,
                'volt': 168.5,
                'rotate': 1567.2,
                'pressure': 98.7,
                'vibration': 45.3,
                'error_count': 2,
                'failure_within_48h': 0,
                'days_since_comp1_maint': 15,
                'days_since_comp2_maint': 8,
                'days_since_comp3_maint': 22,
                'days_since_comp4_maint': 12
            }

            print("📊 Example input values:")
            for feature, value in input_data.items():
                print(f"   {feature}: {value}")

        elif choice == '3':
            # Show requirements
            load_model_info()
            continue

        elif choice == '4':
            print("\n👋 Thank you for using the TTTF Prediction Tool!")
            break

        else:
            print("❌ Invalid choice. Please try again.")
            continue

        # Make prediction
        prediction, prediction_floor = predict_single_sample(model_package, input_data)

        if prediction is not None:
            # Display results
            display_results(input_data, prediction, prediction_floor,
                            model_package['target_names'])

            # Ask if user wants to save
            save_choice = input("\n💾 Save this prediction to log? (y/n): ").strip().lower()
            if save_choice in ['y', 'yes']:
                save_prediction_log(input_data, prediction, prediction_floor,
                                    model_package['target_names'])

        # Ask if user wants to continue
        continue_choice = input("\n🔄 Make another prediction? (y/n): ").strip().lower()
        if continue_choice not in ['y', 'yes']:
            print("\n👋 Thank you for using the TTTF Prediction Tool!")
            break


if __name__ == "__main__":
    main()