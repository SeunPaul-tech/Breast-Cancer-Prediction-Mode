{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "5edd2e8c-fb40-44cd-8d5c-348e76ea6c03",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-08-01 22:14:08.598 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\HP\\.anaconda\\Anaconda\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "C:\\Users\\HP\\.anaconda\\Anaconda\\Lib\\site-packages\\sklearn\\base.py:493: UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "ename": "ValueError",
     "evalue": "X has 5 features, but StandardScaler is expecting 30 features as input.",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[1], line 29\u001b[0m\n\u001b[0;32m     26\u001b[0m input_features \u001b[38;5;241m=\u001b[39m np\u001b[38;5;241m.\u001b[39marray([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean]])\n\u001b[0;32m     28\u001b[0m \u001b[38;5;66;03m# Scale the input\u001b[39;00m\n\u001b[1;32m---> 29\u001b[0m input_scaled \u001b[38;5;241m=\u001b[39m scaler\u001b[38;5;241m.\u001b[39mtransform(input_features)\n\u001b[0;32m     31\u001b[0m \u001b[38;5;66;03m# Predict\u001b[39;00m\n\u001b[0;32m     32\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m st\u001b[38;5;241m.\u001b[39mbutton(\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mPredict\u001b[39m\u001b[38;5;124m'\u001b[39m):\n",
      "File \u001b[1;32m~\\.anaconda\\Anaconda\\Lib\\site-packages\\sklearn\\utils\\_set_output.py:295\u001b[0m, in \u001b[0;36m_wrap_method_output.<locals>.wrapped\u001b[1;34m(self, X, *args, **kwargs)\u001b[0m\n\u001b[0;32m    293\u001b[0m \u001b[38;5;129m@wraps\u001b[39m(f)\n\u001b[0;32m    294\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mwrapped\u001b[39m(\u001b[38;5;28mself\u001b[39m, X, \u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs):\n\u001b[1;32m--> 295\u001b[0m     data_to_wrap \u001b[38;5;241m=\u001b[39m f(\u001b[38;5;28mself\u001b[39m, X, \u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n\u001b[0;32m    296\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28misinstance\u001b[39m(data_to_wrap, \u001b[38;5;28mtuple\u001b[39m):\n\u001b[0;32m    297\u001b[0m         \u001b[38;5;66;03m# only wrap the first output for cross decomposition\u001b[39;00m\n\u001b[0;32m    298\u001b[0m         return_tuple \u001b[38;5;241m=\u001b[39m (\n\u001b[0;32m    299\u001b[0m             _wrap_data_with_container(method, data_to_wrap[\u001b[38;5;241m0\u001b[39m], X, \u001b[38;5;28mself\u001b[39m),\n\u001b[0;32m    300\u001b[0m             \u001b[38;5;241m*\u001b[39mdata_to_wrap[\u001b[38;5;241m1\u001b[39m:],\n\u001b[0;32m    301\u001b[0m         )\n",
      "File \u001b[1;32m~\\.anaconda\\Anaconda\\Lib\\site-packages\\sklearn\\preprocessing\\_data.py:1043\u001b[0m, in \u001b[0;36mStandardScaler.transform\u001b[1;34m(self, X, copy)\u001b[0m\n\u001b[0;32m   1040\u001b[0m check_is_fitted(\u001b[38;5;28mself\u001b[39m)\n\u001b[0;32m   1042\u001b[0m copy \u001b[38;5;241m=\u001b[39m copy \u001b[38;5;28;01mif\u001b[39;00m copy \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m \u001b[38;5;28;01melse\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mcopy\n\u001b[1;32m-> 1043\u001b[0m X \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_validate_data(\n\u001b[0;32m   1044\u001b[0m     X,\n\u001b[0;32m   1045\u001b[0m     reset\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mFalse\u001b[39;00m,\n\u001b[0;32m   1046\u001b[0m     accept_sparse\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mcsr\u001b[39m\u001b[38;5;124m\"\u001b[39m,\n\u001b[0;32m   1047\u001b[0m     copy\u001b[38;5;241m=\u001b[39mcopy,\n\u001b[0;32m   1048\u001b[0m     dtype\u001b[38;5;241m=\u001b[39mFLOAT_DTYPES,\n\u001b[0;32m   1049\u001b[0m     force_all_finite\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mallow-nan\u001b[39m\u001b[38;5;124m\"\u001b[39m,\n\u001b[0;32m   1050\u001b[0m )\n\u001b[0;32m   1052\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m sparse\u001b[38;5;241m.\u001b[39missparse(X):\n\u001b[0;32m   1053\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mwith_mean:\n",
      "File \u001b[1;32m~\\.anaconda\\Anaconda\\Lib\\site-packages\\sklearn\\base.py:654\u001b[0m, in \u001b[0;36mBaseEstimator._validate_data\u001b[1;34m(self, X, y, reset, validate_separately, cast_to_ndarray, **check_params)\u001b[0m\n\u001b[0;32m    651\u001b[0m     out \u001b[38;5;241m=\u001b[39m X, y\n\u001b[0;32m    653\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m no_val_X \u001b[38;5;129;01mand\u001b[39;00m check_params\u001b[38;5;241m.\u001b[39mget(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mensure_2d\u001b[39m\u001b[38;5;124m\"\u001b[39m, \u001b[38;5;28;01mTrue\u001b[39;00m):\n\u001b[1;32m--> 654\u001b[0m     \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_check_n_features(X, reset\u001b[38;5;241m=\u001b[39mreset)\n\u001b[0;32m    656\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m out\n",
      "File \u001b[1;32m~\\.anaconda\\Anaconda\\Lib\\site-packages\\sklearn\\base.py:443\u001b[0m, in \u001b[0;36mBaseEstimator._check_n_features\u001b[1;34m(self, X, reset)\u001b[0m\n\u001b[0;32m    440\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m\n\u001b[0;32m    442\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m n_features \u001b[38;5;241m!=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mn_features_in_:\n\u001b[1;32m--> 443\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mValueError\u001b[39;00m(\n\u001b[0;32m    444\u001b[0m         \u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mX has \u001b[39m\u001b[38;5;132;01m{\u001b[39;00mn_features\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m features, but \u001b[39m\u001b[38;5;132;01m{\u001b[39;00m\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m\u001b[38;5;18m__class__\u001b[39m\u001b[38;5;241m.\u001b[39m\u001b[38;5;18m__name__\u001b[39m\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m \u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m    445\u001b[0m         \u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mis expecting \u001b[39m\u001b[38;5;132;01m{\u001b[39;00m\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mn_features_in_\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m features as input.\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m    446\u001b[0m     )\n",
      "\u001b[1;31mValueError\u001b[0m: X has 5 features, but StandardScaler is expecting 30 features as input."
     ]
    }
   ],
   "source": [
    "# app.py\n",
    "\n",
    "import streamlit as st\n",
    "import numpy as np\n",
    "import joblib\n",
    "\n",
    "# Load the trained model and scaler\n",
    "model = joblib.load('breast_cancer_model.pkl')\n",
    "scaler = joblib.load('scaler.pkl')\n",
    "\n",
    "# Title\n",
    "st.title(\"Breast Cancer Prediction App\")\n",
    "st.write(\"This app predicts whether a tumor is malignant or benign based on diagnostic features.\")\n",
    "\n",
    "# Collect input from the user\n",
    "st.header(\"Enter tumor measurements:\")\n",
    "\n",
    "# Input fields for selected features\n",
    "radius_mean = st.number_input('Radius Mean', min_value=0.0, value=14.0)\n",
    "texture_mean = st.number_input('Texture Mean', min_value=0.0, value=20.0)\n",
    "perimeter_mean = st.number_input('Perimeter Mean', min_value=0.0, value=90.0)\n",
    "area_mean = st.number_input('Area Mean', min_value=0.0, value=500.0)\n",
    "smoothness_mean = st.number_input('Smoothness Mean', min_value=0.0, value=0.1)\n",
    "\n",
    "# Combine inputs into an array\n",
    "input_features = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean]])\n",
    "\n",
    "# Scale the input\n",
    "input_scaled = scaler.transform(input_features)\n",
    "\n",
    "# Predict\n",
    "if st.button('Predict'):\n",
    "    prediction = model.predict(input_scaled)[0]\n",
    "    result = \"Malignant\" if prediction == 1 else \"Benign\"\n",
    "    st.success(f\"The model predicts the tumor is **{result}**.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d24767f3-5b97-4201-86e1-c43ad29862d4",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f9382486-b015-478a-b866-baa9eb78e98d",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
