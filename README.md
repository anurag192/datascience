# 🎓 Student Performance Indicator

An end-to-end machine learning project to predict student performance based on various academic and socio-economic features. This project includes data ingestion, preprocessing, model training, evaluation, deployment via a Flask app, and a production-ready pipeline.

---

## 🚀 Features

- 🔄 End-to-end machine learning pipeline
- 📊 Data ingestion and transformation
- 🧹 Data preprocessing (handling missing values, encoding, scaling)
- 🧠 Model training and evaluation
- 💾 Model persistence using `pickle`
- 🌐 Flask web app for real-time predictions
- 📁 Modular and scalable project structure

---

## 📁 Project Structure

```
├── src/                    # All source code
│   ├── components/         # Data ingestion, transformation, and model training scripts
│   ├── pipeline/           # Training and prediction pipelines
│   ├── utils.py            # Utility functions
│   └── logger.py           # Logging setup
│
├── templates/              # HTML templates for web interface
│   └── index.html
│
├── app.py                  # Flask app entry point
├── requirements.txt        # Python dependencies
├── setup.py                # Package configuration
├── .gitignore              # Files and folders to ignore in version control
├── README.md               # Project documentation
```

---

## ⚙️ Installation

1. **Clone the repository**:
```bash
git clone https://github.com/anurag192/datascienceproject.git
cd datascienceproject
```

2. **Create a virtual environment (optional but recommended)**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. **Install the dependencies**:
```bash
pip install -r requirements.txt
```

4. **Run the app**:
```bash
python app.py
```

---

## 🧠 Model Pipeline Overview

- **Data Ingestion**: Load raw data from source
- **Data Transformation**: Preprocessing like encoding, scaling
- **Model Training**: Train regression/classification model
- **Evaluation**: Measure performance using metrics like R², MSE, etc.
- **Prediction**: Deploy model via web interface to predict performance

---

## 🖥️ Web Interface

After running `app.py`, go to `http://localhost:5000` in your browser.  
You can input student data through the web form and get performance predictions.

---

## ✅ Example Use Case

Enter features such as:
- Parental education level
- Test preparation course
- Lunch type
- Gender
- Math, reading, and writing scores

Receive predicted **student performance level** instantly.

---


---
