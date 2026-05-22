# organize_project.py - Run this to organize everything
import os
import shutil

print("="*60)
print("📁 Organizing FoodGuard Project")
print("="*60)

# Create folders
folders = [
    "scripts", "modules/honey", "modules/spices", "modules/milk",
    "data/raw", "data/processed", "outputs", "tests"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"✅ Created: {folder}")

# Files to move to scripts
scripts_files = [
    "honey_data_cleaning.py", "honey_data_preprocessing.py", "honey_eda.py",
    "honey_feature_engineering.py", "honey_feature_selection.py",
    "honey_model_training.py", "honey_model_evaluation.py",
    "generate_honey_dataset.py", "generate_spices_dataset.py",
    "load_honey_data.py", "save_engineered_data.py", "complete_feature_selection.py"
]

for file in scripts_files:
    if os.path.exists(file):
        shutil.move(file, f"scripts/{file}")
        print(f"✅ Moved: {file}")

# Move detector files
if os.path.exists("honey_model.py"):
    shutil.move("honey_model.py", "modules/honey/detector.py")
    print("✅ Moved: honey_model.py -> modules/honey/detector.py")

if os.path.exists("spices_model.py"):
    shutil.move("spices_model.py", "modules/spices/detector.py")
    print("✅ Moved: spices_model.py -> modules/spices/detector.py")

# Move data files
data_files = ["honey_dataset.csv", "spices_dataset.csv"]
for file in data_files:
    if os.path.exists(file):
        shutil.move(file, f"data/raw/{file}")
        print(f"✅ Moved: {file} -> data/raw/")

# Move processed data
processed_files = [
    "honey_dataset_cleaned.csv", "honey_dataset_engineered.csv",
    "honey_dataset_final.csv", "spices_dataset_cleaned.csv",
    "X_train_standard.npy", "X_test_standard.npy",
    "X_train_standard.csv", "X_test_standard.csv",
    "y_train.npy", "y_test.npy"
]

for file in processed_files:
    if os.path.exists(file):
        shutil.move(file, f"data/processed/{file}")
        print(f"✅ Moved: {file} -> data/processed/")

# Move outputs
outputs_files = ["model_metadata.json", "selected_features.txt"]
for file in outputs_files:
    if os.path.exists(file):
        shutil.move(file, f"outputs/{file}")
        print(f"✅ Moved: {file} -> outputs/")

if os.path.exists("honey_eda_plots"):
    shutil.move("honey_eda_plots", "outputs/honey_eda_plots")
    print("✅ Moved: honey_eda_plots -> outputs/")

# Create __init__.py files
init_files = ["modules/__init__.py", "modules/honey/__init__.py", 
              "modules/spices/__init__.py", "modules/milk/__init__.py"]

for init in init_files:
    with open(init, 'w') as f:
        f.write("# Module initialization file\n")
    print(f"✅ Created: {init}")

print("\n" + "="*60)
print("✅ ORGANIZATION COMPLETE!")
print("="*60)
print("\n📌 Next Steps:")
print("1. Update imports in app.py")
print("2. Run: streamlit run app.py")