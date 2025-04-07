import numpy as np
import pickle
import xgboost as xgb
from flask import Flask,render_template, jsonify, request
from keras.models import load_model
from keras import Sequential
from scipy.stats import mode

from azure.storage.blob import BlobServiceClient
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import io


#Initializing the application 
app = Flask(__name__) 


KEY_VAULT_URL = "https://key-azure-apps.vault.azure.net/" # Azure Key Vault URL
credential = DefaultAzureCredential()
client = SecretClient(vault_url=KEY_VAULT_URL, credential=credential) # Authenticate using Managed Identity

# Fetch secrets from Key Vault
#storage_account_key = client.get_secret("hrtdiseasestorageaccountkey").value
connection_string = client.get_secret("hrtdiseaseconnectionstring").value    
container_name ="models" 

    
def load_pickle_model_from_blob(model_name):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string) 
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=model_name)
    stream = io.BytesIO(blob_client.download_blob().readall())
    return pickle.load(stream)


def load_bytes_from_blob(model_name):    
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=model_name)
    return blob_client.download_blob().readall()


 
mdl_DT_new =  load_pickle_model_from_blob("DT.pkl") # pickle.load(open("model/DT.pkl","rb"))  
mdl_SVM_new = load_pickle_model_from_blob("SVM.pkl") #pickle.load(open("model/SVM.pkl","rb"))  
mdl_RF_new = load_pickle_model_from_blob("RF3.pkl") #pickle.load(open("model/RF.pkl","rb"))  
mdl_XG_new = load_pickle_model_from_blob("XG.pkl") #pickle.load(open("model/XG.pkl","rb"))  

cnn_model_bytes = load_bytes_from_blob("CNN3.keras") 
with open("temp_model.keras", "wb") as f:
    f.write(cnn_model_bytes)
    mdl_CNN_new = load_model("temp_model.keras")

scaler = load_pickle_model_from_blob("encoder_numerical.pkl")

@app.route('/') # Decorate Charater (Empty Route)

def index(): 
   return render_template ("index.html")

@app.route('/predict', methods = ["POST"])
def predict():   
    result = ""   
    
    age = request.form.get('age')
    gender = request.form.get('gender')
    gnrlhlth = request.form.get('gnrlhlth') 
    height = request.form.get('height')    
    weight = request.form.get('weight')
    bmi = request.form.get('bmi')
    stroke = request.form.get('stroke')
    angina = request.form.get('angina')
    codp = request.form.get('codp')
    chestpain = request.form.get('chestpain')
    sleephours = request.form.get('sleephours')
    smokestatus = request.form.get('smokestatus')  
    tdap = request.form.get('tdap')  
    covid = request.form.get('covid')  
    
    
    health_record = [
    age,             # AgeCategory 
    gender,          # Sex (encoded) 
    gnrlhlth,        # GeneralHealth (encoded)
    height,          # HeightInMeters    
    weight,          # WeightInKilograms
    bmi,             # BMI
    stroke,          # HadStroke (encoded)
    angina,          # HadAngina (encoded)
    codp,            # HadCOPD   (encoded)
    chestpain,       # ChestScan (encoded)    
    sleephours,      # SleepHours
    smokestatus,     # SmokerStatus (encoded)
    tdap,            # Tetanus, Diphtheria, and Pertussis (Tdap) vaccine in the last 10 years (encoded)
    covid            # Covid vaccine    
    ]     
    
    health_record_np = np.array(health_record).reshape(1, -1) # Reshaping   
    health_record_scaled = scaler.transform(health_record_np) # Scaling 
    health_record_for_CNN = health_record_scaled.reshape(1, health_record_scaled.shape[1], 1)


    # 1. Predicting from Decision Tree 
    dt_pred = mdl_DT_new.predict(health_record_scaled) 
    # 2. Predicting from SVM
    svm_pred = mdl_SVM_new.predict(health_record_scaled)  
    # 3. Predicting from Random Forest
    rf_pred = mdl_RF_new.predict(health_record_scaled)  
    # 4. Predicting from CNN 
    cnn_pred = mdl_CNN_new.predict(health_record_for_CNN)    
    cnn_clsfctn = (cnn_pred >= .5).astype(int)
    cnn_preds_flat = cnn_clsfctn.flatten()
    # 5. Predicting from XG Boost
    xg_pred = mdl_XG_new.predict(health_record_scaled)    
    # Collect all the predictions
    all_preds = np.array([dt_pred,svm_pred,rf_pred,cnn_preds_flat, xg_pred]).flatten() 

    # Majority voting on Heart 
    majority_pred = mode(all_preds)[0]   

    if majority_pred == 1: 
       heart_status = "Yes" 
       heart_status_color = 'alert alert-danger mt-3'
    else:    
       heart_status = "No"
       heart_status_color = 'alert alert-info mt-3'
    
    
    # Predict probabilities Over Random Forest Model   
    risk_thresholds = { # Define Resk Threshold 
        'high': 0.7,
        'medium': 0.4
    }

    # Define function to categorize risk levels
    def categorize_risk_health_recs(probability, thresholds):
        #positive_class_prob = probabilities[1]  # Assuming the positive class is at index 1
        if probability >= thresholds['high']:
            return 'High', 'alert alert-danger mt-3', probability
        elif probability >= thresholds['medium']:
            return 'Medium' , 'alert alert-warning mt-3', probability
        else:
            return 'Low' , 'alert alert-info mt-3', probability

    health_record_np = np.array(health_record).reshape(1, -1) # Convert the health record to a numpy array and reshape it for the model
    
    # Predict from Decision Tree 
    dt_proba = mdl_DT_new.predict_proba(health_record_scaled)[0][1]
    # Predict from SVM 
    svm_proba = mdl_SVM_new.predict(health_record_scaled)[0]
    # Predict from Random Forest
    rf_proba = mdl_RF_new.predict_proba(health_record_scaled)[0][1]
    # Predict from CNN
    cnn_proba = mdl_CNN_new.predict(health_record_for_CNN)[0][0]
    # Predict from XGBoost
    xg_proba = mdl_XG_new.predict_proba(health_record_scaled)[0][1]        
    # Average the probabilities 
    average_proba = (dt_proba + svm_proba +  rf_proba + cnn_proba + xg_proba) / 5  
 
    risk_category, risk_category_color, risk_probability = categorize_risk_health_recs(average_proba, risk_thresholds)

    # Passing Values Back to form
    frm_age = age
    frm_gender = gender
    frm_gnrlhlth = gnrlhlth
    frm_height = height
    frm_weight = weight
    frm_bmi = bmi
    frm_stroke= stroke
    frm_angina= angina
    frm_codp= codp
    frm_chestpain= chestpain
    frm_sleephours= sleephours
    frm_smokestatus= smokestatus
    frm_tdap= tdap
    frm_covid = covid

    # Risk Probability as a percentage    
    risk_probability = round(risk_probability ,2) * 100 
    result_dt = 'Yes' if dt_pred.flatten() == 1 else 'No'    #Decision Tree Prediction
    result_svm = 'Yes' if svm_pred.flatten() == 1 else 'No'  #SVM Prediction
    result_rf = 'Yes' if rf_pred.flatten() == 1 else 'No'    #Ramdom Forest Prediction
    result_cnn = 'Yes' if cnn_preds_flat.flatten() == 1 else 'No'  #CNN Prediction 
    result_xg = 'Yes' if xg_pred.flatten() == 1 else 'No'    #XG Prediction


    return render_template('index.html', heart_status = heart_status, heart_status_color = heart_status_color, risk_probability = risk_probability, 
                                         risk_category = risk_category, risk_category_color = risk_category_color, 
                                         result_dt = result_dt,
                                         result_svm = result_svm,
                                         result_rf = result_rf,
                                         result_cnn = result_cnn,
                                         result_xg = result_xg,
                                         frm_age = frm_age,
                                         frm_gender = frm_gender,
                                         frm_gnrlhlth = frm_gnrlhlth,
                                         frm_height = frm_height,
                                         frm_weight = frm_weight,
                                         frm_bmi = frm_bmi,
                                         frm_stroke = frm_stroke,
                                         frm_angina = frm_angina,
                                         frm_codp = frm_codp,
                                         frm_chestpain = frm_chestpain,
                                         frm_sleephours = frm_sleephours,
                                         frm_smokestatus = frm_smokestatus,
                                         frm_tdap = frm_tdap,
                                         frm_covid = frm_covid)

if __name__ == '__main__':
    app.run( debug=True)
