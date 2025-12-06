# gui_house_price.py
import pandas as pd
from sklearn.linear_model import LinearRegression
import tkinter as tk
from tkinter import ttk, messagebox

# ------------------------
# 1) Load and preprocess dataset
# ------------------------
df = pd.read_csv("data/train.csv")
df = df[['GrLivArea','BedroomAbvGr','Neighborhood','SalePrice']].dropna()
df = pd.get_dummies(df, columns=['Neighborhood'], drop_first=True)

X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

# Train model on full dataset
model = LinearRegression()
model.fit(X, y)

# ------------------------
# 2) GUI Setup
# ------------------------
root = tk.Tk()
root.title("House Price Predictor")
root.geometry("400x400")

tk.Label(root, text="Enter House Details", font=("Helvetica", 14)).pack(pady=10)

# Area input
tk.Label(root, text="House Area (sq ft)").pack()
area_entry = tk.Entry(root)
area_entry.pack()

# Bedrooms input
tk.Label(root, text="Number of Bedrooms").pack()
bed_entry = tk.Entry(root)
bed_entry.pack()

# Neighborhood dropdown
tk.Label(root, text="Select Neighborhood").pack()
neigh_list = [col.replace('Neighborhood_','') for col in X.columns if col.startswith('Neighborhood_')]
neigh_combo = ttk.Combobox(root, values=neigh_list, state="readonly")
neigh_combo.pack()
neigh_combo.current(0)  # default first neighborhood

# Result label
result_label = tk.Label(root, text="", font=("Helvetica", 12), fg="blue")
result_label.pack(pady=20)

# ------------------------
# 3) Prediction function
# ------------------------
def predict_price():
    try:
        area = float(area_entry.get())
        bedrooms = int(bed_entry.get())
        neighborhood = neigh_combo.get()
        
        # Prepare input
        input_data = {'GrLivArea': area, 'BedroomAbvGr': bedrooms}
        for n in neigh_list:
            input_data[f'Neighborhood_{n}'] = 1 if n == neighborhood else 0
        
        input_df = pd.DataFrame([input_data])
        price = model.predict(input_df)[0]
        
        result_label.config(text=f"Predicted House Price: ₹{round(price,2)}")
    except Exception as e:
        messagebox.showerror("Error", f"Invalid input! {e}")

# ------------------------
# 4) Predict Button
# ------------------------
tk.Button(root, text="Predict Price", command=predict_price, bg="green", fg="white").pack(pady=10)

root.mainloop()
