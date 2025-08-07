# 🏠 Energy Efficiency Regression

This project uses machine learning to predict **Heating Load** and **Cooling Load** for buildings based on structural and design parameters. The data is derived from building simulations conducted in **Autodesk Ecotect**.

---

## 📊 Dataset Overview

- **Samples**: 768 simulated building shapes  
- **Features**: 8 design and architectural parameters  
- **Targets**:  
  - `Y1`: Heating Load (kWh/m²)  
  - `Y2`: Cooling Load (kWh/m²)

Simulations vary based on glazing area, glazing distribution, orientation, compactness, and more.

### 🔧 Features

| Feature | Description                   |
|---------|-------------------------------|
| X1      | Relative Compactness          |
| X2      | Surface Area (m²)             |
| X3      | Wall Area (m²)                |
| X4      | Roof Area (m²)                |
| X5      | Overall Height (m)            |
| X6      | Orientation (categorical)     |
| X7      | Glazing Area (ratio)          |
| X8      | Glazing Area Distribution     |

- All features are numeric (some encoded categories).
- No missing values.
- Can optionally be used for classification by rounding targets.

---