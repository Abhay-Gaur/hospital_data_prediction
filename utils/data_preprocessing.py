import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(file_path, target_column=None):
    df = pd.read_csv(file_path)
    df = df.dropna().drop_duplicates()

    # Encode categorical columns
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # If user hasn’t chosen target column yet, return dataframe only
    if target_column is None:
        return df, None

    # Validate target column
    if target_column not in df.columns:
        raise ValueError(f"❌ Column '{target_column}' not found in dataset!")

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    return train_test_split(X, y, test_size=0.2, random_state=42), label_encoders
