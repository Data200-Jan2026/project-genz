import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from scipy import stats

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Exam Score Prediction", page_icon="📚", layout="wide")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Exam_Score_Prediction.csv")
    return df

df = load_data()

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("📚 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Data Overview", "🔬 Statistical Tests", "📈 Linear Regression"])

# ═════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🎓 Exam Score Prediction App")
    st.markdown("---")

    st.markdown("""
    Welcome to the **Exam Score Prediction App**! This project explores factors that influence student exam scores.

    ### 📌 What this app does:
    - Gives you a quick **overview** of the dataset
    - Runs **statistical tests** (ANOVA, T-Test, Chi-Squared) to find meaningful relationships
    - Builds a **Linear Regression** model to predict exam scores

    ### 📂 Dataset Info
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Students", df.shape[0])
    col2.metric("Features", df.shape[1] - 1)
    col3.metric("Avg Exam Score", f"{df['exam_score'].mean():.2f}")

    st.markdown("### 👀 Quick Peek at the Data")
    st.dataframe(df.head(10), use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# DATA OVERVIEW PAGE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Data Overview":
    st.title("📊 Data Overview")
    st.markdown("---")

    st.subheader("Dataset Shape")
    st.write(f"Rows: **{df.shape[0]}** | Columns: **{df.shape[1]}**")

    st.subheader("Column Data Types")
    st.dataframe(df.dtypes.reset_index().rename(columns={"index": "Column", 0: "Type"}))

    st.subheader("Basic Statistics")
    st.dataframe(df.describe(), use_container_width=True)

    st.subheader("Missing Values")
    missing = df.isnull().sum().reset_index()
    missing.columns = ["Column", "Missing Count"]
    st.dataframe(missing)

    st.subheader("📊 Exam Score Distribution")
    fig, ax = plt.subplots()
    ax.hist(df["exam_score"], bins=20, color="steelblue", edgecolor="white")
    ax.set_xlabel("Exam Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Exam Scores")
    st.pyplot(fig)

    st.subheader("🔥 Correlation Heatmap (Numeric Columns)")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    numeric_df = df.select_dtypes(include=np.number).drop(columns=["student_id"], errors="ignore")
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

# ═════════════════════════════════════════════════════════════════════════════
# STATISTICAL TESTS PAGE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Statistical Tests":
    st.title("🔬 Statistical Tests")
    st.markdown("---")

    # ── T-Test ────────────────────────────────────────────────────────────────
    st.subheader("1️⃣ T-Test — Internet Access vs Exam Score")
    st.markdown("""
    **Question:** Does having internet access affect exam scores?  
    **Groups:** Students with internet (`yes`) vs without (`no`)
    """)
    group_yes = df[df["internet_access"] == "yes"]["exam_score"]
    group_no  = df[df["internet_access"] == "no"]["exam_score"]
    t_stat, t_p = stats.ttest_ind(group_yes, group_no)

    col1, col2 = st.columns(2)
    col1.metric("T-Statistic", f"{t_stat:.4f}")
    col2.metric("P-Value", f"{t_p:.4f}")
    if t_p < 0.05:
        st.success("✅ Significant difference — Internet access DOES affect exam scores (p < 0.05)")
    else:
        st.info("ℹ️ No significant difference found (p ≥ 0.05)")

    fig, ax = plt.subplots()
    ax.boxplot([group_yes, group_no], labels=["Internet: Yes", "Internet: No"])
    ax.set_ylabel("Exam Score")
    ax.set_title("Exam Score by Internet Access")
    st.pyplot(fig)

    st.markdown("---")

    # ── ANOVA ─────────────────────────────────────────────────────────────────
    st.subheader("2️⃣ ANOVA — Study Method vs Exam Score")
    st.markdown("""
    **Question:** Does the study method affect exam scores?  
    **Groups:** Each unique study method in the dataset.
    """)
    groups = [grp["exam_score"].values for _, grp in df.groupby("study_method")]
    f_stat, a_p = stats.f_oneway(*groups)

    col1, col2 = st.columns(2)
    col1.metric("F-Statistic", f"{f_stat:.4f}")
    col2.metric("P-Value", f"{a_p:.4f}")
    if a_p < 0.05:
        st.success("✅ Significant difference — Study method DOES affect exam scores (p < 0.05)")
    else:
        st.info("ℹ️ No significant difference found (p ≥ 0.05)")

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    df.boxplot(column="exam_score", by="study_method", ax=ax2)
    plt.suptitle("")
    ax2.set_title("Exam Score by Study Method")
    ax2.set_xlabel("Study Method")
    ax2.set_ylabel("Exam Score")
    st.pyplot(fig2)

    st.markdown("---")

    # ── Chi-Squared ───────────────────────────────────────────────────────────
    st.subheader("3️⃣ Chi-Squared Test — Gender vs Exam Difficulty")
    st.markdown("""
    **Question:** Is there a relationship between gender and exam difficulty?  
    **Method:** Chi-Squared test on a contingency table.
    """)
    contingency = pd.crosstab(df["gender"], df["exam_difficulty"])
    chi2, chi_p, dof, expected = stats.chi2_contingency(contingency)

    col1, col2, col3 = st.columns(3)
    col1.metric("Chi² Statistic", f"{chi2:.4f}")
    col2.metric("P-Value", f"{chi_p:.4f}")
    col3.metric("Degrees of Freedom", dof)

    st.write("**Contingency Table:**")
    st.dataframe(contingency)
    if chi_p < 0.05:
        st.success("✅ Significant association between Gender and Exam Difficulty (p < 0.05)")
    else:
        st.info("ℹ️ No significant association found (p ≥ 0.05)")

# ═════════════════════════════════════════════════════════════════════════════
# LINEAR REGRESSION PAGE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📈 Linear Regression":
    from sklearn.preprocessing import LabelEncoder

    st.title("📈 Linear Regression — Predicting Exam Score")
    st.markdown("---")

    # ── Encode categorical columns & train model ──────────────────────────────
    categorical_cols = ["gender", "course", "internet_access", "sleep_quality",
                        "study_method", "facility_rating", "exam_difficulty"]

    @st.cache_resource
    def train_lr_model(df):
        df_enc = df.copy()
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df_enc[col] = le.fit_transform(df_enc[col])
            encoders[col] = le
        X = df_enc.drop(["exam_score", "student_id"], axis=1)
        y = df_enc["exam_score"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model, encoders, list(X.columns)

    model, encoders, feature_cols = train_lr_model(df)

    # ── Sidebar inputs ────────────────────────────────────────────────────────
    st.sidebar.header("Enter Student Details")

    age        = st.sidebar.slider("Age", 15, 30, 20)
    s_hours    = st.sidebar.slider("Study Hours", 0.0, 12.0, 4.0, step=0.5)
    attendance = st.sidebar.slider("Class Attendance (%)", 0.0, 100.0, 75.0, step=1.0)
    sleep      = st.sidebar.slider("Sleep Hours", 0.0, 12.0, 7.0, step=0.5)

    st.sidebar.markdown("---")

    gender       = st.sidebar.selectbox("Gender", df["gender"].unique())
    course       = st.sidebar.selectbox("Course", df["course"].unique())
    internet     = st.sidebar.selectbox("Internet Access", df["internet_access"].unique())
    sleep_qual   = st.sidebar.selectbox("Sleep Quality", df["sleep_quality"].unique())
    study_method = st.sidebar.selectbox("Study Method", df["study_method"].unique())
    facility     = st.sidebar.selectbox("Facility Rating", df["facility_rating"].unique())
    difficulty   = st.sidebar.selectbox("Exam Difficulty", df["exam_difficulty"].unique())

    # ── Build input row ───────────────────────────────────────────────────────
    input_dict = {
        "age": age,
        "gender": encoders["gender"].transform([gender])[0],
        "course": encoders["course"].transform([course])[0],
        "study_hours": s_hours,
        "class_attendance": attendance,
        "internet_access": encoders["internet_access"].transform([internet])[0],
        "sleep_hours": sleep,
        "sleep_quality": encoders["sleep_quality"].transform([sleep_qual])[0],
        "study_method": encoders["study_method"].transform([study_method])[0],
        "facility_rating": encoders["facility_rating"].transform([facility])[0],
        "exam_difficulty": encoders["exam_difficulty"].transform([difficulty])[0],
    }
    input_df = pd.DataFrame([input_dict])[feature_cols]

    # ── Predict button ────────────────────────────────────────────────────────
    st.markdown("### 👈 Fill in the details on the sidebar, then click Predict!")

    if st.sidebar.button("🔮 Predict Exam Score"):
        prediction = model.predict(input_df)[0]
        st.success(f"📊 Predicted Exam Score: **{prediction:.2f} / 100**")
    else:
        st.info("Adjust the inputs on the sidebar and hit **Predict Exam Score** to get a result.")
