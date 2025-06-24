import pandas as pd
import joblib

def preprocessing(df):
    selected_features = ["WorkLifeBalance", "JobSatisfaction", "JobLevel", "MonthlyIncome",
                         "Age", "MaritalStatus", "Department", "OverTime", "Attrition"]
    
    df = df[selected_features]
    df = df.drop("Attrition", axis=1)

    categorical_features = df.select_dtypes(include="object").columns
    df = pd.get_dummies(df, columns=categorical_features)

    numerical_features = df.select_dtypes(include=["int64", "float64"]).columns
    scaler = joblib.load("scaler.pkl")
    df[numerical_features] = scaler.transform(df[numerical_features])

    return df

def model_predict(df):
    df = preprocessing(df)

    model = joblib.load("model.pkl")
    predictions = model.predict(df)
    df["Attrition"] = predictions

    return df[["Attrition"]]

def main():
    input_data = pd.read_csv("https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset/refs/heads/main/employee/employee_data.csv")
    input_data = input_data[input_data["Attrition"].isna()].copy()
    predictions = model_predict(input_data)

    return predictions


if __name__ == "__main__":
    result = main()
    print(result)