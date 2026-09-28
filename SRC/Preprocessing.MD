## 1. Preprocessing Techniques Applied

* **Missing Value Imputation:** 
  * Unstructured text (`Return_Reason`): Missing values represent retained purchases (`label = 0`) and are imputed with `"No issue reported"`.
  * Numerical attributes (`Product_Price`, `Session_Dwell_Time`): Imputed using median values to handle skewness.
* **Continuous Feature Scaling:** Applied `RobustScaler` to continuous numerical features (`Product_Price`, `Discount_Applied`, `Session_Dwell_Time`, `Customer_Tenure`) to mitigate extreme price outliers without distorting scale.
* **Categorical Encoding:** Applied `OneHotEncoder` to low-cardinality categorical variables (`Payment_Method`, `Shipping_Region`) to create sparse binary columns.
* **Stratified Data Splitting:** Executed a 70% Train, 15% Validation, and 15% Test stratified split to preserve the natural binary class ratio (~83% non-returns, ~17% returns).

---

## 2. Script Linkage & Execution Guide

* **Associated Script:** `src/01_preprocessing.py`
* **Execution Command:**
  ```bash
  python src/01_preprocessing.py