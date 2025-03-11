import pandas as pd
import numpy as np
#import plotly.express as px
#import plotly.graph_objects as go
#import plotly.io as pio
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

sns.set()

#@st.cache_data


### Import data and display it
dfs = pd.read_csv('Solar_Footprints_V2_7811899327930675815.csv', index_col="OBJECTID") # data frame for solar data
dfp = pd.read_csv('Population-Density By County.csv') # data frame for US population data
# Filter rows with "California" in the "GEO.display-label" column
dfp = dfp[dfp["GEO.display-label"].str.contains("California", na=False)]

dfs=dfs.drop(['Combined Class','Substation Name GTET 100 Max Voltage','HIFLD ID (GTET 100 Max Voltage)','Substation Name GTET 200 Max Voltage',
         'HIFLD ID (GTET 200 Max Voltage)','Substation CASIO Name', 'HIFLD ID (CAISO)', 'Shape__Area', 'Shape__Length'],axis=1)
abbrev_dict = {"Distance to Substation (Miles) GTET 100 Max Voltage" : "Distance to GTET 100",           
               "Percentile (GTET 100 Max Voltage)" : "Percentile (GTET 100)",
               "Distance to Substation (Miles) GTET 200 Max Voltage" : "Distance to GTET 200",
               "Percentile (GTET 200 Max Voltage)" : "Percentile (GTET 200)",
               "Distance to Substation (Miles) CAISO" : "Distance to CAISO substation"}

dfs.rename(abbrev_dict, axis=1, inplace = True)

dfp=dfp[["GCT_STUB.display-label","Density per square mile of land area"]]
dfp.rename(columns={"GCT_STUB.display-label":"County","Density per square mile of land area":"Population Density"},inplace=True)

#dgeo=pd.DataFrame({"County":dfs.County.unique(),"Latitude":latitude,"Longitude":longitude})
dgeo=pd.read_csv('dgeo.csv')

dfs_temp=pd.merge(dfs, dgeo, on="County", how="left").set_index(dfs.index) 
df=pd.merge(dfs_temp, dfp, on="County", how="left")
df=df.set_index(dfs_temp.index)

st.title("Solar energy generation in California")
st.sidebar.title("Table of contents")
pages=["1.Data processing","2.Missing values", "3.DataVisualization", "4.Preprocessing","5.Resampling and scaling","6.Modeling","7.Metrics","8.ROU and shap"]
page=st.sidebar.radio("Go to", pages)

if page == pages[0]:
    st.write('### Presentation of data')
    st.write('Solar panels data after dropping unnecessary coloumns and shortening column names')
    if st.checkbox('Solar dataframe'):
        st.dataframe(dfs.head())

    st.write('Population density data')
    if st.checkbox('Population density'):
        st.dataframe(dfp.head())

    
    st.write('Like population density, the geographical location may have influence on the Solar stations')
    if st.checkbox('Geographical Data'):
        st.dataframe(dgeo.head())
    
    st.write('Merged DataFrame')
    if st.checkbox('Merged datafram'):
        st.dataframe(df.head())




if page == pages[1]:
    import io
    st.write('### Info and Description')
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    if st.checkbox('Info'):
        st.text(info_str)
    if st.checkbox('Describe'):
        st.dataframe(df.describe())

import geopandas as gpd
url = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
states = gpd.read_file(url)
california = states[states["name"] == "California"]

# Geographical plot of Total solar installations per coounty
total_solar=df[["County" ,"Urban or Rural","Latitude","Longitude"]].value_counts().reset_index()
total_solar.columns=["County" ,"Urban or Rural","Latitude","Longitude","frequency"]
total_solar_rural=total_solar[total_solar["Urban or Rural"]!="Urban"]
total_solar_urban=total_solar[total_solar["Urban or Rural"]!="Rural"]



image_gtet=Image.open("distance_gtet.png")
image_area1=Image.open("area_dist.png")
image_area2=Image.open("area_dist2.png")
if page == pages[2]:
    st.write('### Data visulaization')
    if st.checkbox('GTET distance'):
        fig= plt.figure(figsize=(18,10))
        plt.imshow(image_gtet)
        plt.axis("off")
        st.pyplot(fig)
    if st.checkbox('Area distribution'):
        fig, axes = plt.subplots(1, 2, figsize=(12, 10))    
        axes[0].imshow(image_area1)    
        axes[1].imshow(image_area2)    
        st.pyplot(fig)

    # Plotting
    if st.checkbox("Number of solar panels"):
        fig=plt.figure(figsize=(8,8))
        #fig=california.plot(edgecolor="black", facecolor="none", figsize=(8, 8))
        plt.plot(california.get_coordinates().x,california.get_coordinates().y)
        #plt.title("Number of solar panels")
        plt.scatter(total_solar_rural.Longitude,total_solar_rural.Latitude,s=total_solar_rural.frequency,c="r",label="Rural")
        plt.scatter(total_solar_urban.Longitude,total_solar_urban.Latitude,s=total_solar_urban.frequency,c="b",label="Urban",alpha=0.2)
        plt.axis("off")
        plt.legend()
        st.pyplot(fig)

    if st.checkbox("Relative area encroached by solar installations"):
        solar_area=df.groupby("County")["Acres"].sum()
        solar_area_df=pd.merge(solar_area,df.drop("Acres",axis=1),on=["County"],how="outer")
        fig=plt.figure(figsize=(8,8))
        plt.plot(california.get_coordinates().x,california.get_coordinates().y)
        #plt.title("Relative area encroached by solar installations")
        plt.scatter(solar_area_df.Longitude,solar_area_df.Latitude,s=solar_area_df.Acres/20,c="orange",label="area occupied by solar field")
        plt.scatter(solar_area_df.Longitude,solar_area_df.Latitude,s=solar_area_df["Population Density"]/10,c="g",label="population density",alpha=0.2)
        plt.legend()
        plt.axis("off")
        st.pyplot(fig)

from sklearn.preprocessing import OrdinalEncoder

df_enc=df

encoder_ordinal = OrdinalEncoder(categories=[["0 to 25th","25th to 50th","50th to 75th","75th to 100th"]])

df_enc["percentile_GTET100"] = pd.DataFrame(encoder_ordinal.fit_transform(df[["Percentile (GTET 100)"]])).set_index(df_enc.index)
df_enc["percentile_GTET200"] = pd.DataFrame(encoder_ordinal.fit_transform(df[["Percentile (GTET 200)"]])).set_index(df_enc.index)
df_enc["percentile_CAISO"] = pd.DataFrame(encoder_ordinal.fit_transform(df[["Percentile (CAISO)"]])).set_index(df_enc.index)

df_enc = pd.concat([df_enc, pd.get_dummies(df_enc["Install Type"], dtype=int, drop_first=True)], axis=1)

# Variables "Urban or Rural" and "Solar technoeconomic Intersection" are simply converted to binary
df_enc["Urban or Rural"]=df_enc["Urban or Rural"].map({"Urban":0,"Rural":1}).astype(int)
df_enc["Solar Technoeconomic Intersection"]=df_enc["Solar Technoeconomic Intersection"].map({"Within":1,"Outside":0}).astype(int)

# We also drop "County" variable
df_enc=df_enc.drop(["Percentile (GTET 100)","Percentile (GTET 200)","Percentile (CAISO)","Install Type","County"],axis=1)

# Reordering the columns
df_enc=pd.concat([df_enc.drop("Solar Technoeconomic Intersection",axis=1),df_enc["Solar Technoeconomic Intersection"]],axis=1)

if page==pages[3]:
    st.write('### Preprocessing')
    st.write('we performed ordinal encoding of categorical variables.')    
    corr_matrix = df_enc.corr()

    if st.checkbox('Heat correlation map'):
        # Create the heatmap
        fig=plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

        # Show the plot
        st.pyplot(fig)

        st.write('Due to negligible correlation of Longitude and Latitude variables with the target variable, we drop them.')
    

from sklearn.model_selection import train_test_split
df_enc=df_enc.drop(["Longitude","Latitude"], axis=1)

X=df_enc.drop("Solar Technoeconomic Intersection",axis=1)
y=df_enc["Solar Technoeconomic Intersection"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
   


if page == pages[4]:
    st.write('### Resampling and scaling')
    fig=plt.figure()
    sns.countplot(x="Solar Technoeconomic Intersection",data=pd.DataFrame(y_train)) 
    st.pyplot(fig)
    st.write('We are dealing with imbalanced classes so that we need to apply oversampling or undersampling on the training data.')
    st.write('After resampling, scaling were performed.')
    st.write('We prepare three sets of data for model training.')

    dict_={'Variable':['X_train_os_robust_scaling','X_train_os_min-max_scaling','X_train_us_robust_scaling'],'target':['y_train_os','y_train_os','y_train_us']}
    st.table(pd.DataFrame(dict_))
    
# Oversampling

from imblearn.over_sampling import RandomOverSampler

def random_oversample(X_train, y_train):
    """
    Applies RandomOverSampler to balance the class distribution by over-sampling the minority class.

    Args:
    - X_train: Features of the training dataset.
    - y_train: Target labels of the training dataset.

    Returns:
    - X_train_resampled: Resampled feature set.
    - y_train_resampled: Resampled target labels.
    """
    # Apply RandomOverSampler
    X_train_resampled_rOs, y_train_resampled_rOs = RandomOverSampler().fit_resample(X_train, y_train)

    # Return the resampled data for further processing or model training
    return X_train_resampled_rOs, y_train_resampled_rOs
X_train_os,y_train_os=random_oversample(X_train, y_train)

# Undersampling

from imblearn.under_sampling import ClusterCentroids
from collections import Counter

def cluster_centroids_undersample(X_train, y_train, random_state=888):
    """
    Applies ClusterCentroids undersampling on the majority class.

    Args:
    - X_train: Features of the training dataset.
    - y_train: Target labels of the training dataset.
    - random_state: Random seed for reproducibility (default is 888).

    Returns:
    - X_train_resampled_cc: Resampled feature set.
    - y_train_resampled_cc: Resampled target labels.
    """
    # Define the ClusterCentroids undersampler
    cluster_centroids = ClusterCentroids(random_state=random_state) #, n_init=10)
    
    # Apply the undersampling
    X_train_resampled_cc, y_train_resampled_cc = cluster_centroids.fit_resample(X_train, y_train)
    
    # Print the class distribution before and after resampling
    print("Original class distribution:", Counter(y_train))
    print("Resampled class distribution:", Counter(y_train_resampled_cc))
    
    # Plotting class distribution before and after resampling
    plt.figure(figsize=(8, 4))

    # Get the maximum y-value for setting the same y-limits
    original_counts = Counter(y_train).values()
    resampled_counts = Counter(y_train_resampled_cc).values()
    max_y_value = max(max(original_counts), max(resampled_counts))

    # Original distribution
    plt.subplot(1, 2, 1)
    plt.bar(Counter(y_train).keys(), original_counts, color="skyblue")
    plt.title("Original Class Distribution")
    plt.ylim(0, max_y_value)  # Set y-axis limit

    # Resampled distribution
    plt.subplot(1, 2, 2)
    plt.bar(Counter(y_train_resampled_cc).keys(), resampled_counts, color="lightcoral")
    plt.title("Resampled Class Distribution")
    plt.ylim(0, max_y_value)  # Set y-axis limit

    # Return the resampled data for further processing or model training
    return X_train_resampled_cc, y_train_resampled_cc

X_train_us, y_train_us = cluster_centroids_undersample(X_train, y_train)

##Scaling
#Robust Scaling
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def scale_data(X_train, X_test):

    # In our dataset all numerical columns contain outliers (need scaling)
    numerical_columns_with_outliers = [
        "Acres", "Distance to GTET 100", "Distance to GTET 200", 
        "Distance to CAISO substation", "Population Density"
    ]

    # Other columns that do not need to be scaled (all categorical)
    columns_not_to_scale = [
        "Urban or Rural", "percentile_GTET100", "percentile_GTET200", "percentile_CAISO", "Parking", "Rooftop"
    ]

    # Define the column transformer for scaling (for a production setup all other transformation steps like encoding etc. would also fit here)
    preprocessor = ColumnTransformer(
        transformers=[ 
            ("num_with_outliers", RobustScaler(), numerical_columns_with_outliers), # Apply RobustScaler to numerical columns with outliers      
            ("columns_not_to_scale", "passthrough", columns_not_to_scale) # Keep categorical columns without scaling
        ]
    )

    # Creating a pipeline with preprocessing step
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    # Fit the pipeline on the training data and transform X_train_resampled_cc
    # X_train_scaled = pipeline.fit_transform(X_train_us) # I think this should be X_train (added by Manuel)
    X_train_scaled = pipeline.fit_transform(X_train)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=numerical_columns_with_outliers + columns_not_to_scale)

    # Using the same preprocessor to transform the test set
    X_test_scaled = pipeline.transform(X_test)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=numerical_columns_with_outliers + columns_not_to_scale)

    # Check the scaled data shape
    print(f"Scaled X_train shape: {X_train_scaled.shape}")
    print(f"Scaled X_test shape: {X_test_scaled.shape}")

    pd.set_option('display.float_format', '{:.2f}'.format) # avoiding notation like 2.570000e+03 in the output
    #display("X_train_scaled_df:", X_train_scaled_df.describe())  # We can see that the training data is now scaled
    #display("X_test_scaled_df:", X_test_scaled_df.describe())  # We can see that the test data is now scaled, too and also has the same columns!
    
    return X_train_scaled_df, X_test_scaled_df

X_train_os_robust, X_test_robust = scale_data(X_train_os, X_test)
X_train_us_robust, X_test_robust = scale_data(X_train_us, X_test)


# MinMax scalling
from sklearn.preprocessing import StandardScaler

X_train_os_std = X_train_os.copy()
X_test_std = X_test.copy()
scaler = StandardScaler()
X_train_os_std[['Acres', 'Distance to GTET 100', 'Distance to GTET 200', 
                'Distance to CAISO substation', 'Population Density']] = scaler.fit_transform(
    X_train_os[['Acres', 'Distance to GTET 100', 'Distance to GTET 200', 
                'Distance to CAISO substation', 'Population Density']]
)
X_test_std[['Acres', 'Distance to GTET 100', 'Distance to GTET 200', 
            'Distance to CAISO substation', 'Population Density']] = scaler.transform(
    X_test[['Acres', 'Distance to GTET 100', 'Distance to GTET 200', 
            'Distance to CAISO substation', 'Population Density']]
)

#### Modeling
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.metrics import classification_report
import shap
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
results = []  # List to store results for final table
if page == pages[5]:
    st.write('### Oversampled Robust scaled')
    st.write("Best RandomForest Params: {'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 200}")
    st.write("Best SVM Params: {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'}")
    st.write("Best XGBoost Params: {'learning_rate': 0.2, 'max_depth': 7, 'n_estimators': 200}")
    st.write("Best KNN Params: {'metric': 'manhattan', 'n_neighbors': 3, 'weights': 'distance'}")

    st.write('### Undersampled Robust Scaled')
    st.write("Best RandomForest Params: {'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 200}")
    st.write("Best SVM Params: {'C': 1, 'gamma': 'auto', 'kernel': 'rbf'}")
    st.write("Best XGBoost Params: {'learning_rate': 0.2, 'max_depth': 5, 'n_estimators': 50}")
    st.write("Best KNN Params: {'metric': 'manhattan', 'n_neighbors': 7, 'weights': 'distance'}")

    st.write('### Oversampled Standard scaled')
    st.write("Best RandomForest Params: {'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 200}")
    st.write("Best SVM Params: {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'}")
    st.write("Best XGBoost Params: {'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 200}")
    st.write("Best KNN Params: {'metric': 'manhattan', 'n_neighbors': 3, 'weights': 'distance'}")

    st.write('### Summary Table')
    st.dataframe(pd.read_csv('summaery_5.csv'))

    

# Use SVM with best parameters
#svm_best = SVC(C=10, gamma='auto', kernel='rbf', probability=True)
import joblib
#joblib.dump(svm_best,'model')
# Train the model
svm_best=joblib.load('model')
svm_best.fit(X_train_os_robust, y_train_os)
    
# Predict on train and test data
y_train_pred = svm_best.predict(X_train_os_robust)
y_test_pred = svm_best.predict(X_test_robust)

    
if page == pages[6]:
    st.write('### Model Evaluation on Oversampled Robust Scaled Data')
    
    # Classification report
    st.write('#### Classification Report (Test Data)')
    st.text(classification_report(y_test, y_test_pred))
    
    # Confusion matrices
    cm_train = confusion_matrix(y_train_os, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)
    
    # Plot heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Confusion Matrix - Train Data')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=axes[1])
    axes[1].set_title('Confusion Matrix - Test Data')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    st.pyplot(fig)




y_pred_proba_test = svm_best.predict_proba(X_test_robust)[:, 1] # We decided for the SVC model
test_roc_auc = roc_auc_score(y_test, y_pred_proba_test)

image_shap = Image.open("shap.png")

if page == pages[7]:
    st.write('### ROU curve and SHAP plot')
    if st.checkbox('ROC curve'):    
        fig=plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba_test)
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {test_roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim(0, 1) # Sets the limits for both axes to focus on the range [0, 1]
        plt.ylim(0, 1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        #plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        st.pyplot(fig)

    if st.checkbox('Shap plot'):
        fig= plt.figure()
        plt.imshow(image_shap)
        plt.axis("off")
        st.pyplot(fig)
    





