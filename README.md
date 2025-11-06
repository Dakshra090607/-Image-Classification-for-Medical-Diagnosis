# -Image-Classification-for-Medical-Diagnosis
This project is an AI-Powered Cloud Medical Diagnosis System that uses Transfer Learning and Deep Learning models to assist doctors in diagnosing medical conditions from images. The system is deployed via a Streamlit web dashboard and features user authentication, a patient record database (MySQL), and Grad-CAM for model explainability.
Setup Instructions
1.Clone the repository:
https://github.com/Dakshra090607/-Image-Classification-for-Medical-Diagnosis
2.Create a virtual environment & install dependencies:

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
3. Add .env file with MySQL credentials

5.Run the application:
streamlit run app.py
