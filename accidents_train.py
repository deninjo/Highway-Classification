import pickle
import pandas as pd
import numpy as np


from sklearn.svm import SVC
from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split



output_file = 'acc-model.bin'



# ---------------------------------data preparation--------------------------------------------------#
df = pd.read_excel('TRAFFIC ACCIDENTS DATA.xlsx')

# make column names and values uniform
df.rename(columns={'Road/ Highway': 'Road/Highway'}, inplace=True) # renaming a column

df[ 'Total people confirmed dead'] = df[ 'Total people confirmed dead'].apply(pd.to_numeric, errors='coerce')

df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = df.dtypes[df.dtypes == 'object'].index
for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')


# Convert to integer
df['total_people_confirmed_dead'] = df['total_people_confirmed_dead'].astype('Int64')

# Convert to 24-hour time format
# Step 1: Replace time strings to make them compatible
df['time_of_the_accidents'] = df['time_of_the_accidents'].replace(
    r'(\d{1,2})\.(\d{1,2})_?([ap]m)',  # Allow missing underscore
    r'\1:\2 \3',
    regex=True
)
# Step 2: Convert to datetime, handling AM/PM
df['time_of_the_accidents'] = pd.to_datetime(
    df['time_of_the_accidents'], format='%I:%M %p', errors='coerce'
)
# Step 3: Extract only the time in HH:MM format
df['time_of_the_accidents'] = df['time_of_the_accidents'].dt.strftime('%H:%M')


print("Preprocessed df: ")
print(df.info())




#-------------------------------------------data cleaning--------------------------------------------#
#dropping duplicates
df = df.drop_duplicates()

# dropping 'weather_conditions' column
del df['weather_conditions']

# Use .loc to fill the 'date' column forward
df.loc[:, 'date'] = df['date'].ffill()

# Remove rows where 'road/highway' is null
df = df.dropna(subset=['road/highway'])
df = df.reset_index(drop=True)

# Filter rows where road/highway is "kisii_keroka_road"
kisii_keroka_road_rows = df[df['road/highway'] == 'kisii_keroka_road']

# Use loc to apply backward fill on the slice
df.loc[kisii_keroka_road_rows.index, 'brief_accident_details/cause'] = kisii_keroka_road_rows['brief_accident_details/cause'].bfill()

# Filter rows where road/highway is "nakuru_eldoret_highway"
nakuru_eldoret_highway_rows = df[df['road/highway'] == 'nakuru_eldoret_highway']

# Use loc to apply forward fill on the slice as tha cause above would fit the victim
df.loc[nakuru_eldoret_highway_rows.index, 'brief_accident_details/cause'] = nakuru_eldoret_highway_rows['brief_accident_details/cause'].ffill()

# Drop rows where both 'victims' and 'total_people_confirmed_dead' are null
df = df.dropna(subset=['victims', 'total_people_confirmed_dead'], how='all')

# Fill 'total_people_confirmed_dead' with 1 where 'victims' is 'rider'
df.loc[df['victims'] == 'rider', 'total_people_confirmed_dead'] = 1

# Replace nulls in 'total_people_confirmed_dead' with the column's median
df['total_people_confirmed_dead'] = df['total_people_confirmed_dead'].fillna(df['total_people_confirmed_dead'].median())

# Dictionary to map similar names to a unified value
mapping_dict = {
    'passengers': 'passengers',
    'driver_and_passengers': 'driver_and_passengers',
    "passengers_and_driver": 'driver_and_passengers',
    'passenger': 'passengers',
    'pedestrians': 'pedestrians',
    'pedestrian': 'pedestrians',
    'occupants_of_the_small_vehicle_': 'occupants_of_the_small_vehicle_',
    'passengers_and_drivers': 'driver_and_passengers',
    'drivers/occupants': 'drivers/occupants',
    'passengers_and_pedestrians_': 'passengers_and_pedestrians',
    'drivers,passengers_and_pedestrians': 'driver_passenger_pedestrian',
    'rider': 'motorist',
    'driver_and_other_motorists_': 'driver_and_motorists',
    'motorist': 'motorist'
}

# Replace values in the 'victims' column based on the mapping
df['victims'] = df['victims'].replace(mapping_dict)

# Get the top 2 value counts
top_2_values = df['victims'].value_counts().head(2).index

# Randomly choose from the top 2 values and fill the NaNs
df.loc[df['victims'].isna(), 'victims'] = np.random.choice(top_2_values, size=df['victims'].isna().sum())

mapping_dict = {
    '16:30': 'afternoon',
    '17:50': 'evening',
    '18:00': 'evening',
    '12:30': 'afternoon',
    '02:00': 'night',
    '16:20': 'afternoon',
    '18:30': 'evening',
    '21:00': 'night',
    '22:00': 'night',
    '23:30': 'night',
    '13:00': 'afternoon',
    '19:30': 'night',
    '07:30': 'morning',
    '10:00': 'morning',
    '23:00': 'night',
    '04:30': 'night'
}

# Replace values in the 'time_of_the_accidents' column based on the mapping
df['time_of_the_accidents'] = df['time_of_the_accidents'].replace(mapping_dict)


# filling time based on the most frequent time for each 'road/highway'
# Group by 'road/highway' and get the most frequent time for each group
most_frequent_time = df.groupby('road/highway')['time_of_the_accidents'].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)

# Fill null values in 'time of accident' with the most frequent time per 'road/highway'
df['time_of_the_accidents'] = df['time_of_the_accidents'].fillna(most_frequent_time)

# filling time based on the most frequent time for each 'accident_spot'
# Group by 'accident_spot' and get the most frequent time for each group
most_frequent_time = df.groupby('accident_spot')['time_of_the_accidents'].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)

# Fill null values in 'time of accident' with the most frequent time per 'accident_spot'
df['time_of_the_accidents'] = df['time_of_the_accidents'].fillna(most_frequent_time)

# filling time based on the most frequent time for each 'area'
# Group by 'area' and get the most frequent time for each group
most_frequent_time = df.groupby('area')['time_of_the_accidents'].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)

# Fill null values in 'time of accident' with the most frequent time per 'area'
df['time_of_the_accidents'] = df['time_of_the_accidents'].fillna(most_frequent_time)

# filling time based on the most frequent time for each 'county'
# Group by 'county' and get the most frequent time for each group
most_frequent_time = df.groupby('county')['time_of_the_accidents'].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)

# Fill null values in 'time of accident' with the most frequent time per 'county'
df['time_of_the_accidents'] = df['time_of_the_accidents'].fillna(most_frequent_time)

# Forward fill for the remaining null values
df.loc[:, 'time_of_the_accidents'] = df['time_of_the_accidents'].ffill()

df = df.reset_index(drop=True)

print()
print("Clean df: ")
print(df.info())
print()




#------------------------------------------------model training-------------------------------------------------#
# Create the 'risk' target variable
df['risk'] = df['total_people_confirmed_dead'].apply(lambda x: 'High' if x > 5.75 else 'Low')
df.risk = (df.risk == 'High').astype(int)

# features and target variable
X = df.iloc[:,1:9]
y = df['risk']


# Initialize LabelEncoder
le = LabelEncoder()

# Initialize a dictionary to store LabelEncoders
encoders = {}

# Apply Label Encoding to each categorical column and save encoders
X_encoded = X.copy()
for col in X_encoded.columns:
    if X_encoded[col].dtype == 'object':  # Only for categorical columns
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col])
        encoders[col] = le  # Save the encoder for the column

# Apply SelectKBest to extract top 5/8 features
best = SelectKBest(score_func=chi2, k=5)
fit = best.fit(X_encoded, y)

# Prepare scores DataFrame
df_scores = pd.DataFrame(fit.scores_, columns=['Score'])
df_columns = pd.DataFrame(X.columns, columns=['Feature'])

# Join the scores and feature names
scores = pd.concat([df_columns, df_scores], axis=1)
print(scores.nlargest(5, 'Score'))
print()

# Define X and Y
x = X_encoded[['area', 'road/highway', 'brief_accident_details/cause', 'total_people_confirmed_dead', 'time_of_the_accidents']]
y = df['risk']

# Split the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=44)


# Initialize models
models = {
    'Logistic Regression': LogisticRegression(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier(),
}

# Using the Random Forest model from the models dictionary
random_forest_model = models['Random Forest']

# Train the model
random_forest_model.fit(X_train, y_train)

# Make predictions
y_pred = random_forest_model.predict(X_test)



#------------------------------------------model evaluation-------------------------------------------#
report = classification_report(y_test, y_pred, output_dict=True, zero_division=1)

# Extract the metrics for class '0' and '1', and store them along with the model name
metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision (0)': report['0']['precision'],
        'Recall (0)': report['0']['recall'],
        'F1-Score (0)': report['0']['f1-score'],
        'Precision (1)': report['1']['precision'],
        'Recall (1)': report['1']['recall'],
        'F1-Score (1)': report['1']['f1-score']
    }

model_metrics = [metrics]

# Convert the list of dictionaries into a pandas DataFrame
metrics_df = pd.DataFrame(model_metrics)

print("Evaluation of Random Forest:")
print(metrics_df)
print()

cv_scores = cross_val_score(random_forest_model, x, y, cv=5, scoring='accuracy')

# Print cross-validation scores
print(f'Cross-validation scores: {cv_scores}')
print(f'Mean CV accuracy: {np.mean(cv_scores):.4f}')
print()



#------------------------------saving the model----------------------------#
# write to the binary file
with open(output_file, 'wb') as f_out:
    pickle.dump((encoders, random_forest_model),f_out)
