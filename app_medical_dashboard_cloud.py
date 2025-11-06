import streamlit as st
import tensorflow as tf
import numpy as np
import cv2, os, bcrypt, mysql.connector
from datetime import datetime
import pandas as pd
from keras.models import load_model
from io import BytesIO
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
import schedule, time
import threading

# ============ PAGE CONFIG ============
st.set_page_config(page_title="AI Medical Diagnosis Cloud", layout="wide", page_icon="🩺")

# ============ LOAD ENV & MYSQL CONNECTION ============
load_dotenv()
MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST") or "localhost",
    "user": os.getenv("MYSQL_USER") or "root",
    "password": os.getenv("MYSQL_PASSWORD") or "",
    "database": os.getenv("MYSQL_DATABASE") or "ai_medical_db" 
}

# --- Core Connection Function ---
def mysql_connect():
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        if conn.is_connected():
            return conn
        return None
    except mysql.connector.Error:
        return None

# ============ MYSQL TABLE SETUP ============
def init_mysql_db():
    temp_config = {k: v for k, v in MYSQL_CONFIG.items() if k != "database"}
    conn = None
    try:
        conn = mysql.connector.connect(**temp_config)
        cursor = conn.cursor()
        
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_CONFIG['database']}")
        conn.database = MYSQL_CONFIG['database']

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS doctors (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(255) UNIQUE,
                password VARCHAR(255)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id INT AUTO_INCREMENT PRIMARY KEY,
                doctor VARCHAR(255),
                name VARCHAR(255),
                age INT,
                gender VARCHAR(20),
                disease_type VARCHAR(255),
                diagnosis VARCHAR(255),
                confidence FLOAT,
                date DATETIME,
                image_path VARCHAR(255)
            )
        """)
        conn.commit()
    except mysql.connector.Error as err:
        st.sidebar.error(f"Critical MySQL Initialization Error: {err}. Check XAMPP/Database setup.")
    finally:
        if conn and conn.is_connected():
            conn.close()

init_mysql_db()

# ============ AUTHENTICATION (MySQL Only) ============
def register_user(username, password):
    conn = mysql_connect()
    if conn is None:
        st.error("Cannot register: Database connection failed.")
        return
        
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO doctors (username, password) VALUES (%s, %s)", (username, hashed_pw.decode()))
        conn.commit()
        st.success("✅ Registration successful! You can now log in.")
    except mysql.connector.IntegrityError:
        st.error("Username already exists!")
    except mysql.connector.Error as err:
        st.error(f"Registration Error: {err}")
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

def login_user(username, password):
    conn = mysql_connect()
    if conn is None:
        st.error("Login failed: Database connection failed.")
        return False
        
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM doctors WHERE username=%s", (username,))
    record = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if record and bcrypt.checkpw(password.encode(), record[0].encode()):
        return True
    return False

# ============ MODEL LOADING & CLASS MAPPING ============
HAM10000_LABELS = {
    0: 'akiec (Actinic Keratoses)', 1: 'bcc (Basal Cell Carcinoma)', 2: 'bkl (Benign Keratosis)', 
    3: 'df (Dermatofibroma)', 4: 'nv (Melanocytic Nevi)', 5: 'mel (Melanoma)', 6: 'vasc (Vascular Lesions)'
}

DISEASE_MODELS = {
    "Chest X-ray (Pneumonia)": r"C:\Ai_project\saved_models\model_VGG16.keras", 
    "Skin Cancer (HAM10000)": r"C:\Ai_project\saved_models\skin_cancer_model_finetuned.h5"
}

@st.cache_resource
def load_ai_model(path):
    if os.path.exists(path):
        return load_model(path)
    else:
        st.warning(f"⚠️ Model file for {os.path.basename(path)} not found at specified location.")
        return None
    
# --- Grad-CAM function (Unchanged) ---
def generate_gradcam(model, img_array, layer_name=None):
    try:
        if layer_name is None:
            for layer in reversed(model.layers):
                if len(layer.output.shape) == 4 and 'conv' in layer.name:
                    layer_name = layer.name
                    break
        if not layer_name:
            return None

        grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if predictions.shape[-1] > 1:
                class_index = tf.argmax(predictions[0])
                loss = predictions[:, class_index]
            else:
                loss = predictions[:, 0]
                
        grads = tape.gradient(loss, conv_outputs)[0]
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]

        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = np.maximum(heatmap, 0) / tf.reduce_max(heatmap)
        heatmap = cv2.resize(heatmap.numpy(), (224, 224))
        return heatmap

    except Exception:
        return None

# ============ DATABASE FUNCTIONS (MySQL Only) ============
def add_patient(doctor, name, age, gender, disease_type, diagnosis, confidence, image_path):
    conn = mysql_connect()
    if conn is None:
        st.error("Cannot save record: Database connection failed.")
        return

    cursor = conn.cursor()
    query = """
        INSERT INTO patients (doctor, name, age, gender, disease_type, diagnosis, confidence, date, image_path)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    values = (doctor, name, age, gender, disease_type, diagnosis, confidence, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), image_path)
    
    try:
        cursor.execute(query, values)
        conn.commit()
        st.success("✅ Patient record saved to MySQL database.")
    except mysql.connector.Error as err:
        st.error(f"Error saving patient record: {err}")
    finally:
        if cursor: cursor.close()
        if conn: conn.close()


def get_all_patients(doctor):
    conn = mysql_connect()
    if conn is None:
        return pd.DataFrame()

    query = "SELECT * FROM patients WHERE doctor=%s ORDER BY id DESC"
    df = pd.DataFrame()
    try:
        df = pd.read_sql(query, conn, params=(doctor,))
    except mysql.connector.Error:
        df = pd.DataFrame()
    finally:
        if conn: conn.close()
    return df

# ============ LOGIN UI ============
if "user" not in st.session_state:
    st.title("🩺 AI Medical Diagnosis Login")
    
    if mysql_connect() is None:
        st.error("🚨 Critical Error: Could not connect to MySQL database. Please ensure XAMPP is running.")
        st.stop()
        
    tab1, tab2 = st.tabs(["🔑 Login", "🆕 Register"])

    with tab1:
        user = st.text_input("Username")
        pw = st.text_input("Password", type="password")
        if st.button("Login"):
            if login_user(user, pw):
                st.session_state["user"] = user
                st.success(f"Welcome back, Dr. {user.capitalize()} 👨‍⚕️")
                st.rerun()
            else:
                st.error("Invalid credentials!")

    with tab2:
        new_user = st.text_input("Choose Username")
        new_pw = st.text_input("Choose Password", type="password")
        if st.button("Register"):
            if new_user and new_pw:
                register_user(new_user, new_pw)
            else:
                st.warning("Please fill both fields.")
    st.stop()

# ============ DASHBOARD START ============
doctor = st.session_state["user"]
st.sidebar.header(f"🧑‍⚕️ Dr. {doctor.capitalize()}")
st.sidebar.caption("System Status: Connected to MySQL")

if st.sidebar.button("🚪 Logout", use_container_width=True):
    st.session_state.pop("user", None)
    st.rerun()


# ============ MENU ============
menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🧠 Diagnosis", "📁 Case History", "📊 Analytics"]
)

# ============ PAGES ============
if menu == "🏠 Home":
    st.title("🩺 AI Medical Diagnosis Dashboard")
    st.markdown("---")
    st.write("Welcome to the unified cloud-powered AI system for medical image analysis.")
    st.info("Navigate to the **🧠 Diagnosis** tab to upload an image and receive an AI prediction.")

elif menu == "🧠 Diagnosis":
    st.header("🧠 New Diagnosis")
    st.markdown("---")

    col_pat, col_mod = st.columns([1.5, 2])
    
    # Patient Details Column
    with col_pat:
        st.subheader("Patient Information")
        name = st.text_input("Patient Name", placeholder="Enter Patient Full Name")
        age = st.number_input("Age", 0, 120, 30)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    # Model and File Upload Column
    with col_mod:
        st.subheader("AI Model Selection")
        disease_choice = st.selectbox("🧬 Select Disease Type", list(DISEASE_MODELS.keys()))
        file = st.file_uploader("Upload Medical Image (JPG/PNG)", type=["jpg", "jpeg", "png"])
        model = load_ai_model(DISEASE_MODELS[disease_choice])

        if model is None:
            st.error("🛑 Cannot run diagnosis: Model file not found.")
            st.stop()
        
        st.markdown("---")
        if st.button("🔬 **Run AI Diagnosis**", type="primary", use_container_width=True):
            if not name.strip():
                st.error("🛑 Please enter the **Patient Name** to run the diagnosis.")
            elif not file:
                st.error("🛑 Please **upload an image** for the AI to analyze.")
            else:
                with st.spinner('Analyzing image and generating Grad-CAM...'):
                    # --- DIAGNOSIS LOGIC ---
                    img_bytes = file.read()
                    img_np = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                    img_resized = cv2.resize(img_np, (224, 224))
                    img_tensor = np.expand_dims(img_resized / 255.0, axis=0)

                    predictions = model.predict(img_tensor)[0]
                    
                    # Determine diagnosis based on model type
                    if disease_choice == "Chest X-ray (Pneumonia)":
                        pred = predictions[0]
                        label = "Pneumonia" if pred > 0.5 else "Normal"
                        confidence = round(float(pred if label == "Pneumonia" else 1 - pred), 3)
                        diagnosis_output = f"**Final Diagnosis:** {label.upper()}"
                        
                    elif disease_choice == "Skin Cancer (HAM10000)":
                        predicted_index = np.argmax(predictions)
                        label = HAM10000_LABELS.get(predicted_index, "Unknown")
                        confidence = round(float(predictions[predicted_index]), 3)
                        diagnosis_output = f"**Final Diagnosis:** {label.upper()}"
                    
                    else:
                        label = "ERROR"
                        confidence = 0.0
                        diagnosis_output = "**Diagnosis Error**"

                    # --- VISUAL RESULTS ---
                    st.markdown("### ✅ AI Result")
                    
                    # Display the final prediction prominently
                    st.markdown(f"**{diagnosis_output}** with **{confidence*100:.1f}%** Confidence.")
                    
                    # --- START OF CORE FIX ---
                    # Only display the detailed probability breakdown if the model output length is 7
                    if disease_choice == "Skin Cancer (HAM10000)":
                        
                        # ⭐ CORE FIX: Check if the predictions array is the expected length (7) ⭐
                        if len(predictions) == 7:
                            st.markdown("**Top 3 Probabilities:**")
                            # This line is now safe because we verified the length is 7
                            prob_df = pd.Series(predictions, index=HAM10000_LABELS.values())
                            prob_df = (prob_df * 100).round(1).sort_values(ascending=False).head(3)
                            st.dataframe(prob_df,  use_container_width=True)
                        else:
                            st.warning(f"⚠️ Skin model error: Expected 7 predictions but got {len(predictions)}. Check model structure.")
                        
                    # --- END OF CORE FIX ---

                    # Grad-CAM Visualization
                    heatmap = generate_gradcam(model, img_tensor)
                    if heatmap is not None:
                        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                        overlay = cv2.addWeighted(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB), 0.6, heatmap_colored, 0.4, 0)

                        col_a, col_b = st.columns(2)
                        col_a.image(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB), caption="Original Image", use_container_width=True)
                        col_b.image(overlay, caption="Grad-CAM Heatmap (Area of Focus)", use_container_width=True)
                    else:
                        st.info("Grad-CAM visualization not available for this configuration.")

                    # Save Record
                    image_save_folder = os.path.join("database", "images")
                    os.makedirs(image_save_folder, exist_ok=True)
                    path = os.path.join(image_save_folder, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                    with open(path, "wb") as f:
                        f.write(img_bytes)
                        
                    add_patient(doctor, name, age, gender, disease_choice, label.upper(), confidence, path)
                
elif menu == "📁 Case History":
    st.header("📁 Patient Records & History")
    st.markdown("---")
    df = get_all_patients(doctor)
    if df.empty:
        st.info("No records found in the MySQL database yet.")
    else:
        # Search and filter options
        col_search, col_filter, _ = st.columns([2, 2, 4])
        name_filter = col_search.text_input("🔍 Search Patient Name", placeholder="e.g., John Doe")
        disease_filter = col_filter.selectbox("Filter by Disease Type", ["All"] + sorted(df["disease_type"].unique()))
        
        filtered_df = df.copy()
        if name_filter:
            filtered_df = filtered_df[filtered_df["name"].str.contains(name_filter, case=False, na=False)]
        if disease_filter != "All":
            filtered_df = filtered_df[filtered_df["disease_type"] == disease_filter]
        
        # Drop internal IDs and image paths for clean display
        display_df = filtered_df.drop(columns=['id', 'image_path', 'doctor'])
        
        st.dataframe(display_df, use_container_width=True)

        # Export button
        towrite = BytesIO()
        filtered_df.to_excel(towrite, index=False)
        towrite.seek(0)
        st.download_button("📤 Export Filtered Data to Excel", towrite, file_name=f"records_{doctor}_{datetime.now().strftime('%Y%m%d')}.xlsx")

elif menu == "📊 Analytics":
    st.header("📊 Doctor Performance Analytics")
    st.markdown("---")
    df = get_all_patients(doctor)
    if df.empty:
        st.info("No data available to generate analytics charts.")
    else:
        # ENHANCEMENT: Pandas transformation/apply 
        df['confidence_category'] = df['confidence'].apply(
            lambda x: 'High (>= 90%)' if x >= 0.9 else ('Medium (70-90%)' if x >= 0.7 else 'Low (< 70%)')
        )
        df['date_only'] = pd.to_datetime(df['date']).dt.date

        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("1. Cases by Disease Type")
            st.bar_chart(df["disease_type"].value_counts())
            
            st.subheader("2. Diagnosis Count by Gender")
            st.bar_chart(df["gender"].value_counts())

        with col_b:
            st.subheader("3. Diagnosis Confidence Distribution")
            category_order = ['High (>= 90%)', 'Medium (70-90%)', 'Low (< 70%)']
            counts = df["confidence_category"].value_counts().reindex(category_order, fill_value=0)
            st.bar_chart(counts)

            st.subheader("4. Average Confidence Over Time")
            daily_conf = df.groupby("date_only")["confidence"].mean().reset_index()
            daily_conf.rename(columns={'date_only': 'Date', 'confidence': 'Avg. Confidence'}, inplace=True)
            st.line_chart(daily_conf, x='Date', y='Avg. Confidence')


# --- SCHEDULER FUNCTIONS (Unchanged) ---
def export_daily_backup():
    conn = mysql_connect()
    if conn:
        try:
            df = pd.read_sql("SELECT * FROM patients", conn)
            backup_dir = "database/backups"
            os.makedirs(backup_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            df.to_excel(os.path.join(backup_dir, f"patient_backup_{timestamp}.xlsx"), index=False)
            print(f"✅ Daily MySQL backup saved successfully to {backup_dir}.")
        except Exception as e:
            print("❌ MySQL Backup failed:", e)
        finally:
            conn.close()
    else:
        print("❌ Backup skipped: Could not connect to MySQL.")

schedule.every().day.at("23:59").do(export_daily_backup)

def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(60)

threading.Thread(target=run_scheduler, daemon=True).start()