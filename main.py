import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
from utils.data_preprocessing import load_and_preprocess_data
from utils.model_training import train_and_save_model
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from scipy import stats
from scipy.stats import norm, binom, poisson, chi2_contingency, ttest_ind, f_oneway
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(page_title="Hospital Management System", layout="wide")

st.title("🏥 Hospital Management System (AI-Powered)")
st.markdown("---")

option = st.sidebar.selectbox("Select an option:", [
    "View Data", "Train Model"
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

        # Drop duplicates-----------------------------------------------------------------------------------------------
        df.drop_duplicates(inplace=True)

        # 5. Data Cleaning-----------------------------------------------------------------------------------------------
        # Fill numerical columns with median
        df['Age'] = df['Age'].fillna(df['Age'].median())
        df['Charges'] = df['Charges'].fillna(df['Charges'].median())
        df['Readmission_Count'] = df['Readmission_Count'].fillna(df['Readmission_Count'].median())
        df['Visit_Duration'] = df['Visit_Duration'].fillna(df['Visit_Duration'].median())
        df['Treatment_Cost'] = df['Treatment_Cost'].fillna(df['Treatment_Cost'].median())

        # Fill categorical/text columns with 'Unknown' and normalize-----------------------------------------------------------------------------------------------
        df['Gender'] = df['Gender'].fillna("Unknown").str.strip().str.lower()
        df['Department'] = df['Department'].fillna("Unknown").str.strip().str.lower()
        df['Insurance_Status'] = df['Insurance_Status'].fillna("Unknown").str.strip().str.lower()
        df['Recovery_Status'] = df['Recovery_Status'].fillna("Unknown").str.strip().str.lower()
        df['Patient_Condition_Severity'] = df['Patient_Condition_Severity'].fillna("Unknown").str.strip().str.lower()

        # 6. Drop Irrelevant Columns (direct, no conditions)-----------------------------------------------------------------------------------------------
        df.drop(columns=['Name', 'Address'], inplace=True)

        df['age_group'] = pd.cut(df['Age'],
                         bins=[0, 18, 35, 50, 65, 120],
                         labels=['child', 'young_adult', 'adult', 'senior', 'elderly'],
                         right=True,
                         include_lowest=True)

        # Cost per Day-----------------------------------------------------------------------------------------------
        df['cost_per_day'] = df['Treatment_Cost'] / df['Visit_Duration'].replace(0, np.nan)

        # Chronic Condition Flag-----------------------------------------------------------------------------------------------
        df['chronic_flag'] = df['Chronic_Conditions'].apply(lambda x: 0 if x == 'none' else 1)

        # 9. Encoding Categorical Variables
        df_encoded = pd.get_dummies(df,
                                    columns=['Gender', 'Department', 'Insurance_Status',
                                            'Recovery_Status', 'Patient_Condition_Severity'],
                                    drop_first=True)
        

        # Visualization
        num_cols = df.select_dtypes(include='number').columns.tolist()
        if len(num_cols) >= 2:
            st.write("### Histograms")
            fig, ax = plt.subplots(figsize=(14, 10))
            df[['Charges', 'Visit_Duration', 'Treatment_Cost']].hist(ax=ax, bins=30)
            plt.suptitle("Numerical Feature Distributions of patients")
            st.pyplot(fig)

            st.write("### Correlation Heatmap")

            # Create a figure
            fig, ax = plt.subplots(figsize=(8, 6))

            # Plot heatmap on that figure
            sns.heatmap(df[['Charges', 'Visit_Duration', 'Treatment_Cost']].corr(),annot=True,cmap="coolwarm",ax=ax)
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)

            for col in ['Department', 'Insurance_Status', 'Patient_Condition_Severity']:
                fig, ax = plt.subplots(figsize=(4, 2)) 
                sns.countplot(x=col, data=df, order=df[col].value_counts().index)
                ax.set_title(f"Distribution of {col}")
                plt.xticks(rotation=45)
                st.pyplot(fig)

            st.write("### Recovery Rate by Gender")
            gender_table = (df.groupby("Gender")["Recovery_Status"].value_counts(normalize=True).unstack().round(3))
            st.dataframe(gender_table)

            fig_age = px.histogram(
                df,
                x="Age",
                nbins=30,
                color="Gender",
                color_discrete_map={"male": "steelblue", "female": "orchid"},
                marginal="box",
                title="Interactive Age Distribution"
            )

            st.plotly_chart(fig_age, use_container_width=True)

            fig_cost = px.scatter(
                df,
                x="Charges",
                y="Treatment_Cost",
                color="Patient_Condition_Severity",
                color_discrete_map={
                    "mild": "lightgreen",
                    "moderate": "skyblue",
                    "severe": "coral"
                },
                hover_data=["Gender", "Department", "Recovery_Status"],
                title="Charges vs Treatment Cost (Interactive)"
            )

            st.plotly_chart(fig_cost, use_container_width=True)

            fig_pie = px.pie(
                df,
                names="Recovery_Status",
                color="Recovery_Status",
                color_discrete_map={
                    "recovered": "seagreen",
                    "ongoing": "gold",
                    "unknown": "gray"
                },
                title="Recovery Status Distribution"
            )

            st.plotly_chart(fig_pie, use_container_width=True)

            text = " ".join(df['Symptoms'].dropna().astype(str))
            wc = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap='Set2',
                max_words=100
            ).generate(text)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            ax.set_title("Word Cloud of Patient Symptoms", fontsize=14, fontweight='bold')
            st.pyplot(fig)

            st.title("Hospital Management System – Statistical Analysis")
            st.header("📊 Measures of Central Tendency & Dispersion")

            cols = ['Age','Charges','Treatment_Cost','Visit_Duration','Readmission_Count']

            summary = df[cols].describe()

            variance = df[cols].var()
            variance.name = "variance"

            iqr = df[cols].quantile(0.75) - df[cols].quantile(0.25)
            iqr.name = "IQR"

            mode = df[cols].mode().iloc[0]
            mode.name = "mode"

            extended_summary = pd.concat([
                summary,
                variance.to_frame().T,
                iqr.to_frame().T,
                mode.to_frame().T
            ])

            st.write("### Extended Summary Statistics")
            st.dataframe(extended_summary)
            st.header("📈 Probability Distributions")

# ------------------ Normal Distribution (Age) ------------------
            st.subheader("Normal Distribution – Age")

            age_mean = df['Age'].mean()
            age_std = df['Age'].std()

            x = np.linspace(df['Age'].min(), df['Age'].max(), 100)
            y = norm.pdf(x, age_mean, age_std)

            fig1, ax1 = plt.subplots(figsize=(6,4))
            ax1.hist(df['Age'].dropna(), bins=30, density=True, alpha=0.6,
                    color='skyblue', edgecolor='black', label='Age Histogram')
            ax1.plot(x, y, color='coral', linewidth=2, label='Normal Curve')
            ax1.set_title("Age Distribution with Normal Curve")
            ax1.set_xlabel("Age")
            ax1.set_ylabel("Density")
            ax1.grid(True, linestyle='--', alpha=0.5)
            ax1.legend()

            st.pyplot(fig1)

# ------------------ Binomial Distribution (Recovery Status) ------------------
            st.subheader("Binomial Distribution – Recovery Status")

            n = len(df)
            p = (df['Recovery_Status'].str.lower() == 'recovered').mean()

            x_binom = np.arange(0, n+1)
            pmf_binom = binom.pmf(x_binom, n, p)

            fig2, ax2 = plt.subplots(figsize=(6,4))
            ax2.bar(x_binom[:50], pmf_binom[:50], color='lightcoral',
                    edgecolor='black', label='Binomial PMF')
            ax2.set_title(f"Binomial PMF: Recovery Outcomes (n={n})")
            ax2.set_xlabel("Number of Recovered Patients")
            ax2.set_ylabel("Probability")
            ax2.grid(True, linestyle='--', alpha=0.5)
            ax2.legend()

            st.pyplot(fig2)

            prob_at_most_60 = binom.cdf(60, n, p)
            st.write(f"**P(X ≤ 60 recovered out of {n}):** `{prob_at_most_60:.4f}`")

            # ------------------ Poisson Distribution (Readmission Count) ------------------
            st.subheader("Poisson Distribution – Readmission Count")

            lambda_val = df['Readmission_Count'].mean()
            x_poisson = np.arange(0, 11)
            pmf_poisson = poisson.pmf(x_poisson, lambda_val)

            fig3, ax3 = plt.subplots(figsize=(6,4))
            ax3.bar(x_poisson, pmf_poisson, color='teal', edgecolor='black',
                    label='Poisson PMF')
            ax3.set_title("Poisson Distribution of Readmission Counts")
            ax3.set_xlabel("Readmission Count")
            ax3.set_ylabel("Probability")
            ax3.grid(True, linestyle='--', alpha=0.5)
            ax3.legend()

            st.pyplot(fig3)

            st.header("Chi-Square Test: Gender vs Recovery Status")

            # Assuming df is already loaded earlier
            # Example: df = pd.read_csv("yourfile.csv")

            # Clean Data
            df = df[df['Gender'].notnull() & df['Recovery_Status'].notnull()]

            # Contingency table
            contingency_table = pd.crosstab(df['Gender'], df['Recovery_Status'])

            st.subheader("Contingency Table")
            st.dataframe(contingency_table)

            # Chi-square test
            chi2, p, dof, expected = chi2_contingency(contingency_table)

            st.subheader("Chi-Square Test Results")
            st.write(f"**Chi-square Statistic:** {chi2:.3f}")
            st.write(f"**Degrees of Freedom:** {dof}")
            st.write(f"**P-value:** {p:.3f}")

            # Decision with alpha = 0.05
            alpha = 0.05

            if p < alpha:
                st.success("Result: Significant association between Gender and Recovery Status (Reject H₀)")
            else:
                st.info("Result: No significant association between Gender and Recovery Status (Fail to reject H₀)")

            # Heatmap visualization
            st.subheader("Heatmap: Gender vs Recovery Status")

            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlGnBu', ax=ax)
            plt.title("Contingency Table Heatmap")
            st.pyplot(fig)


            st.header("Independent Sample T-Test: Charges vs Insurance Status")

            # Assuming df is loaded earlier in your Streamlit app
            # Example: df = pd.read_csv("yourfile.csv")

            # Convert 'Insurance_Status' to lowercase for safety
            df['Insurance_Status'] = df['Insurance_Status'].astype(str).str.lower()

            # Separate groups
            insured = df[df['Insurance_Status'] == 'yes']['Charges']
            not_insured = df[df['Insurance_Status'] == 'no']['Charges']

            st.subheader("Group Sizes")
            st.write(f"Insured Count: {len(insured)}")
            st.write(f"Not Insured Count: {len(not_insured)}")

            # T-test
            t_stat, p_value = stats.ttest_ind(insured, not_insured, equal_var=False)  # Welch's t-test is safer

            st.subheader("T-Test Results")
            st.write(f"**T-Statistic:** {t_stat:.3f}")
            st.write(f"**P-Value:** {p_value:.3f}")

            # Decision rule
            alpha = 0.05
            if p_value < alpha:
                st.success("Reject H₀: There is a **significant difference** in Charges between Insured and Uninsured patients.")
            else:
                st.info("Fail to reject H₀: **No significant difference** in Charges between both groups.")

            # Boxplot
            st.subheader("Boxplot: Charges by Insurance Status")

            fig, ax = plt.subplots(figsize=(7, 4))
            sns.boxplot(x='Insurance_Status', y='Charges', data=df, palette='Set2')
            plt.title("Charges by Insurance Status")

            st.pyplot(fig)            
            

# ----- TRAIN MODEL -----
elif option == "Train Model":
    st.subheader("🧠 Train Machine Learning Model")

    if os.path.exists(file_path):

        df = pd.read_csv(file_path)

        if st.button("Start Training"):
            st.write("🔄 Running preprocessing...")

            # -------------------------
            # STEP 1: Handle Missing Values
            # -------------------------
            df['Age'] = df['Age'].fillna(df['Age'].median())
            df['Charges'] = df['Charges'].fillna(df['Charges'].median())

            # -------------------------
            # STEP 2: Encode Categorical Variables
            # -------------------------
            encoder = LabelEncoder()
            df['Gender'] = encoder.fit_transform(df['Gender'].astype(str))
            df['Insurance_Status'] = encoder.fit_transform(df['Insurance_Status'].astype(str))
            df['Recovery_Status'] = encoder.fit_transform(df['Recovery_Status'].astype(str))

            # -------------------------
            # STEP 3: Features + Target
            # -------------------------
            X = df[['Age', 'Gender', 'Treatment_Cost',
                    'Visit_Duration', 'Insurance_Status', 'Readmission_Count']]
            X = pd.get_dummies(X, drop_first=True)
            y = df['Recovery_Status']

            # -------------------------
            # STEP 4: Scaling
            # -------------------------
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # -------------------------
            # STEP 5: Train-Test Split
            # -------------------------
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42)

            # ============================================================
            # 🔹 MODEL 1: Logistic Regression
            # ============================================================
            st.header("▶ Logistic Regression Results")

            from sklearn.linear_model import LogisticRegression
            log_model = LogisticRegression()
            log_model.fit(X_train, y_train)

            acc_log = log_model.score(X_test, y_test)
            st.metric("Accuracy value : ",acc_log)
            st.success("🎯 Logistic Regression Trained Successfully!")
            st.metric("Logistic Regression Accuracy", f"{acc_log*100:.2f}%")

            st.divider()   # visual separator

            # ============================================================
            # 🔹 MODEL 2: k-NN
            # ============================================================
            st.header("▶ k-Nearest Neighbors (k-NN) Results")

            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.model_selection import cross_val_score

            st.write("📌 Finding best value of K...")

            k_values = range(1, 21)
            cv_scores = []

            for k in k_values:
                knn = KNeighborsClassifier(n_neighbors=k)
                scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
                cv_scores.append(scores.mean())

            best_k = k_values[cv_scores.index(max(cv_scores))]

            st.write(f"🔍 Best K found: **{best_k}**")

            # Train final KNN
            knn_model = KNeighborsClassifier(n_neighbors=best_k)
            knn_model.fit(X_train, y_train)

            acc_knn = knn_model.score(X_test, y_test)
            st.metric("Accuracy vakue : ",acc_knn)
            st.success("🤖 KNN Model Trained Successfully!")
            st.metric("KNN Accuracy", f"{acc_knn*100:.2f}%")

            # ============================================================
            # SAVE MODELS
            # ============================================================
            import joblib
            os.makedirs("models", exist_ok=True)

            joblib.dump(log_model, "models/logistic_model.pkl")
            joblib.dump(knn_model, "models/knn_model.pkl")
            joblib.dump(scaler, "models/scaler.pkl")
            joblib.dump(encoder, "models/label_encoder.pkl")

            st.divider()
            st.info("""
📦 Saved Models:
- `logistic_model.pkl`
- `knn_model.pkl`
- `scaler.pkl`
- `label_encoder.pkl`
""")

    else:
        st.error("❌ CSV file not found!")


# ----- PREDICT -----
# elif option == "Predict":
#     st.subheader("🔮 Predict from Input")
#     st.subheader("🔮 Coming Soon")
#     # model_path = "models/trained_model.pkl"

#     # if not os.path.exists(model_path):
#     #     st.warning("⚠️ Model not trained yet. Train a model first.")
#     # else:
#     #     model = joblib.load(model_path)
#     #     df, _ = load_and_preprocess_data(file_path)
#     #     input_data = {}

#     #     # Take numeric inputs dynamically
#     #     for col in df.select_dtypes(include=['int64', 'float64']).columns:
#     #         input_data[col] = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))

#     #     input_df = pd.DataFrame([input_data])

#     #     if st.button("Predict"):
#     #         prediction = model.predict(input_df)[0]
#     #         st.success(f"🧾 Predicted Value: **{prediction:.2f}**")

# elif option == "Predict":
#     st.subheader("🔮 Predict Recovery Status")

#     model_path = "models/trained_model.pkl"
#     encoders_path = "models/label_encoders.pkl"

#     if not os.path.exists(model_path) or not os.path.exists(encoders_path):
#         st.warning("⚠️ Model or encoders not found. Please train the model first.")
#     else:
#         # Load model & encoders
#         model = joblib.load(model_path)
#         label_encoders = joblib.load(encoders_path)

#         df, _ = load_and_preprocess_data(file_path)
#         input_data = {}
#         st.write("### 🧾 Enter Patient Details")

#         # Skip target column
#         target_column = "Recovery_Status"

#         for col in df.columns:
#             if col == target_column:
#                 continue

#             # If the column was label-encoded
#             if col in label_encoders:
#                 options = list(label_encoders[col].classes_)
#                 selected_option = st.selectbox(f"{col}", options)
#                 encoded_value = label_encoders[col].transform([selected_option])[0]
#                 input_data[col] = encoded_value
#             else:
#                 # For numeric columns
#                 col_min, col_max, col_mean = df[col].min(), df[col].max(), df[col].mean()
#                 input_data[col] = st.number_input(f"{col}", float(col_min), float(col_max), float(col_mean))

#         input_df = pd.DataFrame([input_data])

#         if st.button("Predict"):
#             try:
#                 prediction = model.predict(input_df)[0]

#                 # Decode prediction if Recovery_Status was encoded
#                 if target_column in label_encoders:
#                     prediction_label = label_encoders[target_column].inverse_transform([int(prediction)])[0]
#                 else:
#                     prediction_label = prediction

#                 st.success(f"🧾 Predicted Recovery Status: **{prediction_label}**")

#             except Exception as e:
#                 st.error(f"❌ Error during prediction: {e}")
    