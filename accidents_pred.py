import pickle
import numpy as np
import pandas as pd


model_file = 'acc-model.bin'
df = pd.read_excel('TRAFFIC ACCIDENTS DATA.xlsx')
features = ['area', 'road/highway', 'brief_accident_details/cause', 'total_people_confirmed_dead', 'time_of_the_accidents']



# -------data preparation--------#
def data_preparation(x):


    # make column names and values uniform
    x.rename(columns={'Road/ Highway': 'Road/Highway'}, inplace=True)  # renaming a column

    x['Total people confirmed dead'] = x['Total people confirmed dead'].apply(pd.to_numeric, errors='coerce')

    x.columns = x.columns.str.lower().str.replace(' ', '_')

    categorical_columns = x.dtypes[x.dtypes == 'object'].index
    for c in categorical_columns:
        x[c] = x[c].str.lower().str.replace(' ', '_')

    # Convert to integer
    x['total_people_confirmed_dead'] = x['total_people_confirmed_dead'].astype('Int64')

    # Convert to 24-hour time format
    # Step 1: Replace time strings to make them compatible
    x['time_of_the_accidents'] = x['time_of_the_accidents'].replace(
        r'(\d{1,2})\.(\d{1,2})_?([ap]m)',  # Allow missing underscore
        r'\1:\2 \3',
        regex=True
    )
    # Step 2: Convert to datetime, handling AM/PM
    x['time_of_the_accidents'] = pd.to_datetime(
        x['time_of_the_accidents'], format='%I:%M %p', errors='coerce'
    )
    # Step 3: Extract only the time in HH:MM format
    x['time_of_the_accidents'] = x['time_of_the_accidents'].dt.strftime('%H:%M')

    return x

df = data_preparation(df)


#-------data cleaning------------#
def data_cleaning(d):

    # dropping duplicates
    d = d.drop_duplicates()

    # dropping 'weather_conditions' column
    del d['weather_conditions']

    # Use .loc to fill the 'date' column forward
    d.loc[:, 'date'] = d['date'].ffill()

    # Remove rows where 'road/highway' is null
    d = d.dropna(subset=['road/highway'])
    d = d.reset_index(drop=True)

    # Filter rows where road/highway is "kisii_keroka_road"
    kisii_keroka_road_rows = d[d['road/highway'] == 'kisii_keroka_road']

    # Use loc to apply backward fill on the slice
    d.loc[kisii_keroka_road_rows.index, 'brief_accident_details/cause'] = kisii_keroka_road_rows[
        'brief_accident_details/cause'].bfill()

    # Filter rows where road/highway is "nakuru_eldoret_highway"
    nakuru_eldoret_highway_rows = d[d['road/highway'] == 'nakuru_eldoret_highway']

    # Use loc to apply forward fill on the slice as tha cause above would fit the victim
    d.loc[nakuru_eldoret_highway_rows.index, 'brief_accident_details/cause'] = nakuru_eldoret_highway_rows[
        'brief_accident_details/cause'].ffill()

    # Drop rows where both 'victims' and 'total_people_confirmed_dead' are null
    d = d.dropna(subset=['victims', 'total_people_confirmed_dead'], how='all')

    # Fill 'total_people_confirmed_dead' with 1 where 'victims' is 'rider'
    d.loc[d['victims'] == 'rider', 'total_people_confirmed_dead'] = 1

    # Replace nulls in 'total_people_confirmed_dead' with the column's median
    d['total_people_confirmed_dead'] = d['total_people_confirmed_dead'].fillna(
        d['total_people_confirmed_dead'].median())

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
    d['victims'] = d['victims'].replace(mapping_dict)

    # Get the top 2 value counts
    top_2_values = d['victims'].value_counts().head(2).index

    # Randomly choose from the top 2 values and fill the NaNs
    d.loc[d['victims'].isna(), 'victims'] = np.random.choice(top_2_values, size=d['victims'].isna().sum())

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
    d['time_of_the_accidents'] = d['time_of_the_accidents'].replace(mapping_dict)

    # filling time based on the most frequent time for each 'road/highway'
    # Group by 'road/highway' and get the most frequent time for each group
    most_frequent_time = d.groupby('road/highway')['time_of_the_accidents'].transform(
        lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)

    # Fill null values in 'time of accident' with the most frequent time per 'road/highway'
    d['time_of_the_accidents'] = d['time_of_the_accidents'].fillna(most_frequent_time)

    # filling time based on the most frequent time for each 'accident_spot'
    # Group by 'accident_spot' and get the most frequent time for each group
    most_frequent_time = d.groupby('accident_spot')['time_of_the_accidents'].transform(
        lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)

    # Fill null values in 'time of accident' with the most frequent time per 'accident_spot'
    d['time_of_the_accidents'] = d['time_of_the_accidents'].fillna(most_frequent_time)

    # filling time based on the most frequent time for each 'area'
    # Group by 'area' and get the most frequent time for each group
    most_frequent_time = d.groupby('area')['time_of_the_accidents'].transform(
        lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)

    # Fill null values in 'time of accident' with the most frequent time per 'area'
    d['time_of_the_accidents'] = d['time_of_the_accidents'].fillna(most_frequent_time)

    # filling time based on the most frequent time for each 'county'
    # Group by 'county' and get the most frequent time for each group
    most_frequent_time = d.groupby('county')['time_of_the_accidents'].transform(
        lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)

    # Fill null values in 'time of accident' with the most frequent time per 'county'
    d['time_of_the_accidents'] = d['time_of_the_accidents'].fillna(most_frequent_time)

    # Forward fill for the remaining null values
    d.loc[:, 'time_of_the_accidents'] = d['time_of_the_accidents'].ffill()

    d = d.reset_index(drop=True)


    return d

df = data_cleaning(df)

# Create the 'risk' target variable
df['risk'] = df['total_people_confirmed_dead'].apply(lambda x: 'High' if x > 5.75 else 'Low')
df.risk = (df.risk == 'High').astype(int)



# reading the label encoder and model from the binary file
with open(model_file, 'rb') as f_in:
    encoders, random_forest_model = pickle.load(f_in)


# Extract a row (e.g., row 43)
row = df.iloc[[68]]

print("Test data features: ")
print(row[features].to_string())
# Apply label encoding to the row
for col in row.columns:
    if col in encoders:  # Only apply encoding if there's an encoder for the column
        le = encoders[col]
        if row[col].iloc[0] in le.classes_:  # Check if the value exists in the encoder
            row.loc[:, col] = le.transform(row[col])

        else:
            # Handle unseen labels by adding them temporarily
            le.classes_ = np.append(le.classes_, row[col].iloc[0])
            row.loc[:, col] = le.transform(row[col])

X_test_row = row[features]

print()
print("Encoded Features: ")
print(X_test_row)


# Make predictions using the trained Random Forest model
prediction = random_forest_model.predict(X_test_row)[0]

proba = random_forest_model.predict_proba(X_test_row)
proba_high = proba[0][1] * 100

# Compare the predicted value with the actual value
actual_value = row['risk'].iloc[0]  # Extracts the only value from the series(w/o index)


print(f"High risk probability is {proba_high:.2f}%")
print(f"Prediction: {prediction}")
print(f"Actual: {actual_value}")
