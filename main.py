
# import streamlit as st
# import pandas as pd
# import os
# #Name the Application's Title
# st.title("Hospital Management System") 
 
# #Add side bar options
# option = st.sidebar.selectbox("Select an option:", ["Hospital Information System"])  
 
# # To display text
# st.write(f"You selected: {option}")
# st.write(f"    ")

# file_paths = {
#     "Hospital Information System": "csv_datas/HospitalMangementSystem.csv",
# }

# file_path = file_paths[option]

# if os.path.exists(file_path):
#     try:
#         # Read the CSV file
#         df = pd.read_csv(file_path)
        
#         # Display success message
#         st.success(f"✅ {option} loaded successfully from: {file_path}")
        
#         # Show basic file info
#         st.write(f"**File Info:** {df.shape[0]} rows × {df.shape[1]} columns")
        
#         # Display the data table
#         st.subheader(f"{option}")
#         st.dataframe(df, use_container_width=True)
        
#     except Exception as e:
#         st.error(f"❌ Error reading CSV file: {e}")
# else:
#     st.error(f"❌ File not found: {file_path}")
#     st.info("Please make sure your CSV files are in the 'data' folder with the correct names")

# st.write("---")

# st.write(f"Developed By : Abhay Gaur")



import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
from utils.data_preprocessing import load_and_preprocess_data
from utils.model_training import train_and_save_model
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Hospital Management System", layout="wide")

st.title("🏥 Hospital Management System (AI-Powered)")
st.markdown("---")

option = st.sidebar.selectbox("Select an option:", [
    "View Data", "Train Model", "Predict"
])

file_path = "csv_datas/HospitalMangementSystem.csv"

# ----- VIEW DATA -----
if option == "View Data":
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        st.success(f"✅ Data loaded successfully: {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df, use_container_width=True)
        st.write("### 📊 Summary Statistics")
        st.write(df.describe())

        # Visualization
        num_cols = df.select_dtypes(include='number').columns.tolist()
        if len(num_cols) >= 2:
            st.write("### 📈 Scatter plot")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            ax1.scatter(df[num_cols[0]], df[num_cols[1]], color='blue', marker='o')
            ax1.set_xlabel(num_cols[0])
            ax1.set_ylabel(num_cols[1])
            ax1.set_title(f"Scatter Plot of {num_cols[0]} vs {num_cols[1]}")
            st.pyplot(fig1)
            
            st.write("### Histogram count for Treatment")
            st.markdown("---")
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            sns.countplot(x='Treatment', data=df, palette='Set2', ax=ax2)
            ax2.set_title("Count of Patients by Treatment")
            ax2.set_xlabel("Treatment")
            ax2.set_ylabel("Count")
            st.pyplot(fig2)

            st.write("### Patients Using Services")
            class_counts = df['Services_Used'].value_counts()
            fig3, ax3 = plt.subplots(figsize=(6, 6))
            ax3.pie(
                class_counts,
                labels=class_counts.index,
                autopct='%1.1f%%',
                colors=['gold', 'lightblue', 'lightcoral']
            )
            ax3.set_title("Patients Class Distribution")
            st.pyplot(fig3)

            st.write("### Patient Recovery Status by Age")
            patients_by_age = df[df['Recovery_Status'] == "Recovered"].groupby('Age').size().cumsum()
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            ax4.plot(patients_by_age.index, patients_by_age.values, color='darkgreen', marker='o')
            ax4.set_title("Patients Recovered by Age")
            ax4.set_xlabel("Age")
            ax4.set_ylabel("Total Patients Recovered")
            st.pyplot(fig4)

    else:
        st.error("❌ CSV file not found!")

# ----- TRAIN MODEL -----
elif option == "Train Model":
    st.subheader("🧠 Train Machine Learning Model")
    st.subheader("🧠 Coming Soon")
    # if os.path.exists(file_path):
    #     df, _ = load_and_preprocess_data(file_path)
    #     target_column = st.selectbox("Select Target Column (to Predict)", df.columns)

    #     if st.button("Train Model"):
    #         try:
    #             (X_train, X_test, y_train, y_test), encoders = load_and_preprocess_data(file_path, target_column)
    #             model_choice = st.selectbox("Select Model", ["LinearRegression", "DecisionTree", "RandomForest", "GradientBoosting"])
    #             model, mae = train_and_save_model(X_train, X_test, y_train, y_test, model_choice)
    #             st.success(f"✅ Model '{model_choice}' trained successfully with MAE = {mae:.2f}")
    #         except ValueError as e:
    #             st.error(str(e))
    # else:
        # st.error("❌ CSV file not found!")

# ----- PREDICT -----
elif option == "Predict":
    st.subheader("🔮 Predict from Input")
    st.subheader("🔮 Coming Soon")
    # model_path = "models/trained_model.pkl"

    # if not os.path.exists(model_path):
    #     st.warning("⚠️ Model not trained yet. Train a model first.")
    # else:
    #     model = joblib.load(model_path)
    #     df, _ = load_and_preprocess_data(file_path)
    #     input_data = {}

    #     # Take numeric inputs dynamically
    #     for col in df.select_dtypes(include=['int64', 'float64']).columns:
    #         input_data[col] = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))

    #     input_df = pd.DataFrame([input_data])

    #     if st.button("Predict"):
    #         prediction = model.predict(input_df)[0]
    #         st.success(f"🧾 Predicted Value: **{prediction:.2f}**")
