import joblib
import pandas as pd
import numpy as np

# ----------- Load Saved Models -------------
scaler = joblib.load("models/scaler.pkl")
encoder = joblib.load("models/encoder.pkl")
model = joblib.load("models/linear_regression_model.pkl")

# ----------- Define Input Columns -------------
original_area = float(input("area: "))
bedrooms = float(input("bedrooms: "))
bathrooms = float(input("bathrooms: "))
stories = float(input("stories: "))
parking = float(input("parking: "))

# Categorical
mainroad = input("mainroad (yes/no): ").strip().lower()
guestroom = input("guestroom (yes/no): ").strip().lower()
basement = input("basement (yes/no): ").strip().lower()
hotwaterheating = input("hotwaterheating (yes/no): ").strip().lower()
airconditioning = input("airconditioning (yes/no): ").strip().lower()
prefarea = input("prefarea (yes/no): ").strip().lower()
furnishingstatus = input("furnishingstatus (furnished/unfurnished/semifurnished): ").strip().lower()

# ----------- Prepare Data -------------
# If area was transformed during training, apply log transform
area_log = np.log1p(original_area)

# Create input dict
input_data = {
    'area_log': area_log,  # or 'area_log' if that’s what the scaler was trained on
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'stories': stories,
    'parking': parking,
    'mainroad': mainroad,
    'guestroom': guestroom,
    'basement': basement,
    'hotwaterheating': hotwaterheating,
    'airconditioning': airconditioning,
    'prefarea': prefarea,
    'furnishingstatus': furnishingstatus
}

df_input = pd.DataFrame([input_data])

# Columns used during training
num_cols = ['bedrooms', 'bathrooms', 'stories', 'parking', 'area_log']
cat_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating',
            'airconditioning', 'prefarea', 'furnishingstatus']

# ----------- Preprocessing -------------
df_input[num_cols] = scaler.transform(df_input[num_cols])
df_cat_encoded = pd.DataFrame(
    encoder.transform(df_input[cat_cols]),
    columns=encoder.get_feature_names_out(cat_cols),
    index=df_input.index
)
X_final = pd.concat([df_input[num_cols], df_cat_encoded], axis=1)

# ----------- Predict -------------
prediction = model.predict(X_final)[0]
print(f"\n Predicted House Price: ₹{prediction:,.2f}")