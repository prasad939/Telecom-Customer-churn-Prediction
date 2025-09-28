# Project Overview
This project develops a machine learning model to predict customer churn for a telecommunications company. Churn prediction helps identify customers who are likely to discontinue service, enabling the company to take proactive retention actions.


## Features
- Uses Random Forest classifier trained on customer usage, service, and billing data.

- Data preprocessing includes tenure grouping and one-hot encoding of categorical features.

- Web application interface built with Flask for interactive churn prediction.

- Provides prediction results with confidence scores based on user inputs.

- Supports input fields based on common telecom customer attributes (e.g., MonthlyCharges, Contract type).

  ## Dataset
- Dataset used: first_telc.csv (Telco customer data with features relevant to churn).

- Key features include gender, tenure, payment method, internet service type, and customer support options.

- Target variable: customer churn status (binary classification).

  ### How to Run
- 1.Install required packages (listed in requirements.txt).

- 2.Train the model or use provided saved model model.sav.
