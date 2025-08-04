# 🧼 Titanic Dataset Cleaning – Preprocessing & Outlier Removal

This project contains a Python script to clean and preprocess the Titanic dataset. It handles missing data, encodes categorical features, scales numerical columns, and removes outliers — all with minimal assumptions and straightforward logic.

## 📋 What the Script Does

### 1. 🧹 Initial Cleanup
- Drops `Name` and `Ticket` columns (not useful for analysis).

### 2. 🧾 Category Conversion
- Converts all `object`-type columns to `category`.
- Prints category value counts for manual inspection.

### 3. 🚢 Cabin Column Handling
- Filters `Cabin` entries to only keep values with a **letter followed by two digits** (e.g., `B45`, `C23`).
- Extracts:
  - `Cabin_letter`: the deck letter
  - `Cabin_number`: the numeric part
- Drops the original `Cabin` column.

### 4. ❌ Missing Values
- Fills missing `Age` values with the **mean**.
- Drops rows with missing `Embarked` values (only 2 rows).

### 5. 🔤 Label Encoding
- Applies `LabelEncoder` to all categorical columns.

### 6. 📊 Feature Scaling
- Uses `MinMaxScaler` to normalize `Age` and `Fare` columns (values scaled between 0 and 1).

### 7. 📈 Visualization
- Count plot of the `Survived` column.
- Boxplots for `Age` and `Fare` to spot outliers.

### 8. ✂️ Outlier Removal
- Uses IQR method to remove **upper-bound** outliers in `Fare` and `Age`.

### 9. 💾 Final Output
- Saves the cleaned DataFrame to `titanic_cleaned.csv`.

## 📁 Output File

- `titanic_cleaned.csv`: Cleaned and processed dataset, ready for analysis or machine learning tasks.

## 📌 Notes

- The script assumes that the dataset is in the same folder, named `Titanic-Dataset.csv`.
- Cleaning logic is minimal but effective for quick prototyping or exploratory data analysis.

---

Feel free to fork this and build your modeling pipeline on top of the cleaned data!
