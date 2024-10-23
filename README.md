# Heart Disease Predictor
This repository contains a machine learning model to predict the likelihood of heart disease based on various health 
metrics. The project demonstrates a comprehensive machine learning pipeline that includes data gathering to model deployment.

## Web Application
The Heart Disease Predictor is deployed as a Streamlit web application. You can access and use the predictor directly through your web browser:

1. Visit the following URL: 
   - [https://chris-huitt-hd-predictor.streamlit.app/](https://chris-huitt-hd-predictor.streamlit.app/)
2. Input the required health metrics into the provided fields.
   - You can find a range of example values in `test_cases.txt`.
3. Click the "Predict" button to see the heart disease risk prediction.

## Key Objectives
The project follows the steps below:
1. **Data Collection:** Collected health metrics and heart disease diagnoses from the UCI Heart Disease dataset.
2. **Exploratory Data Analysis (EDA):** Analyzed the dataset through descriptive analysis and data visualisations to understand the relationships between features and the target variable.
3. **Data Preprocessing:** Cleaned and transformed the data to prepare it for model development.
4. **Feature Selection:** Identified the most influential features for heart disease prediction.
5. **Model Development:** Tested multiple machine learning algorithms to identify the best-performing model.
   - Initially tested several algorithms: Logistic Regression, Random Forest, Gradient Boost, K-Nearest Neighbors, SVM, XGBoost, and LightGBM.
   - Select XGBoost as the final model based on model evaluation and performance metrics
6. **Model Deployment:** Deployed the model as a Streamlit web application for public use.

## Technical 
### Machine Learning Model
- **Libraries:** 
    - scikit-learn
    - numpy
    - pandas
    - matplotlib
    - seaborn
    - statsmodel
    - joblib
    - lightgbm
    - xgboost

#### Initial Model Exploration
- **Algorithms tested:** Logistic Regression, Random Forest, Gradient Boost, K-nearest Neighbors, SVM, XGBoost, LightGBM
- **Input features:** _Please refer to Data section below for details on all input features_
    - ID
    - Age
    - Sex
    - Dataset
    - Chest pain type (cp)
    - Resting blood pressure (trestbps)
    - Cholesterol (chol)
    - Fasting blood sugar (fbs)
    - Resting electrocardiographic results (restecg)
    - Maximum heart rate achieved (thalach)
    - Exercise-induced angina (exang)
    - ST depression induced by exercise relative to rest (oldpeak)
    - Slope of the peak exercise ST segment (slope)
    - Number of major vessels colored by fluoroscopy (ca)
    - Thalassemia (thal)
- **Target feature:** Presence of heart disease (0 = no, 1 = yes) (num)
#### Final Model
- **Algorithm used:** XGBoost
    - After tuning the models, XGBoost was selected the final model due to it having a very strong ROC-AUC (.8462), 
the best F1 score (.5127), and the best accuracy (.6977) and balanced precision and recall. 
- **Input features:** _Please refer to Data section below for details on all input features_
    - Age
    - Resting blood pressure (trestbps)
    - Cholesterol (chol)
    - Maximum heart rate achieved (thalach)
    - ST depression induced by exercise relative to rest (oldpeak)
    - Number of major vessels colored by fluoroscopy (ca)
    - Thalassemia (thal)
- **Target feature:** Presence of heart disease (0 = no, 1 = yes)
- **Output:** Level of risk for heart disease (0 = low, 1 = high)
### Web Application
- **Framework:** Streamlit
- **Deployment:** Streamlit Share
- **Features:**
    - Input fields for health metrics
    - Prediction of heart disease risk

## Project Structure
```
├── data/
│   ├── heart_disease_data.csv
│   └── preprocessed_heart_disease_data.csv
├── models/
│   ├── inital_models_info.joblib
│   ├── rank_1_xgboost_model_final.joblib
│   ├── rank_2_random_forest_model_final.joblib
│   ├── rank_3_gradient_boosting_model_final.joblib
│   └── top_3_models_info_final.joblib
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_feature_selection.ipynb
│   ├── 04_model_exploration.ipynb
│   ├── 05_model_evaluation.ipynb
│   └── 06_final_model_run.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── feature_selection.py
│   ├── model_dev.py
│   └── model_evaulation.py
├── hd_app.py
├── README.md
├── requirements.txt
└── test_cases.txt
```

## Notebook Details
1. `01_EDA.ipynb`: Exploratory Data Analysis
    - Descriptive statistics
    - Data visualizations
2. `02_data_preprocessing.ipynb`: Data Preprocessing
    - Data cleaning
    - Data transformation
3. `03_feature_selection.ipynb`: Feature Selection
4. `04_model_exploration.ipynb`: Initial Model Exploration, considering all features and all models
5. `05_model_evaluation.ipynb`: Model Evaluation, feature importance, and predictions
6. `06_final_model_dev.ipynb`: Final Model Development, considering top features as determined from `05_model_evaluation.ipynb`

## Key Findings
- Identified the most influential features for heart disease prediction
- Achieved 69.77% accuracy with the best-performing model (e.g., XGBoost)
- Developed a simplified model with 7 features

## How to Run
The Heart Disease Predictor is deployed as a Streamlit web application. You can access and use the predictor directly through your web browser:

1. Visit the following URL: 
   - [https://chris-huitt-hd-predictor.streamlit.app/](https://chris-huitt-hd-predictor.streamlit.app/)
2. Input the required health metrics into the provided fields.
   - You can find a range of example values in `test_cases.txt`.
3. Click the "Predict" button to see the heart disease risk prediction.

For developers interested in running the project locally:

1. Clone this repository: 
    ```bash 
    git clone https://github.com/c-huitt/heart-disease-predictor.git
   ```
2. Navigate to the project directory: `cd heart-disease-predictor`
    ```bash 
    cd heart-disease-predictor
   ```
3. Install required packages:
    ```bash 
    pip install -r requirements.txt
   ```
4. Run notebooks in order:
   - `1_EDA.ipynb`
   - `2_Feature_Selection.ipynb`
   - `3_Model_Development.ipynb`
   - `4_Model_Interpretation.ipynb`
5. Run the Streamlit app locally:
    ```bash 
    streamlit run hd_app.py
   ```
## Future Improvements
- Integrate additional data for more robust predictions and insights.
- Develop a secure web application for easy use by healthcare professionals with real patient data.
- Improve UX/UI design for better user experience.

## Data Glossary
- id: patient ID
- age: age in years
- sex: sex
     - 1 = male; 0 = female
- database: which database it came from
- cp: chest pain type
     - Value 1: typical angina
     - Value 2: atypical angina
     - Value 3: non-anginal pain
     - Value 4: asymptomatic
- trestbps: resting blood pressure (in mm Hg on admission to the hospital)
- chol: serum cholestoral in mg/dl
- fbs: (fasting blood sugar > 120 mg/dl)
     - 1 = true; 0 = false
- restecg: resting electrocardiographic results
     - Value 0: normal
     - Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
     - Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
- thalach: maximum heart rate achieved
- exang: exercise induced angina
     - 1 = yes; 0 = no
- oldpeak = ST depression induced by exercise relative to rest
- slope: the slope of the peak exercise ST segment
     - Value 1: upsloping
     - Value 2: flat
     - Value 3: downsloping
- ca: number of major vessels (0-3) colored by flourosopy
- thal:
     - 3 = normal; 6 = fixed defect; 7 = reversable defect
- num: diagnosis of heart disease (angiographic disease status)
     - Value 0: < 50% diameter narrowing
     - Value 1: > 50% diameter narrowingb(in any major vessel: attributes 59 through 68 are vessels)


## Contact
Chris Huitt - christopherhuitt@gmail.com

Project Link: [https://github.com/c-huitt/heart-disease-predictor](https://github.com/c-huitt/heart-disease-predictor)

## Acknowledgements
- [Dataset source: UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease/)
