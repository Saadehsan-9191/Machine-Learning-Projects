Diamonds Predictor Application 

Project Overview 

This project analyses the Diamonds dataset from Kaggle and builds multiple machine learning models to predict diamond price, classify diamond clarity, perform customer segmentation, and provide an AI assistant for diamond analysis. The project includes complete EDA, preprocessing, ANN models, clustering, and a Gradio web application. 

Folder Structure 

Diamonds-Predictor/ 
│ 
├── README.md 
├── EDA_Report.html 
├── EDA_Summary.csv 
├── diamonds_app.py 
└── diamonds_app.py 
 

Dataset 

Source: Kaggle – shivam2503/diamonds Rows: 53,940 Features: 

Numeric: carat, depth, table, x, y, z, price 

Categorical: cut, colour, clarity 


Exploratory Data Analysis (EDA) 

The following analyses were performed: 

Missing value check 
Duplicate removal 
Outlier detection using boxplots 
Distribution visualization using histograms 
Correlation heatmap 
Automated EDA report using Sweetviz 

Key findings: 

Strong correlation between carat and price 
No missing values 
Some outliers in x, y, z 
Clear patterns between cut, colour, clarity, and price 

Data Preprocessing 

Steps completed: 

Removed duplicate rows 
Removed rows with zero dimensions (x, y, z) 
One‑Hot Encoding for categorical features: cut, colour, clarity 
Standard Scaling for numeric features: carat, depth, table, x, y, z 
Train/Test split (80/20) 

Regression Model (Price Prediction) 

An Artificial Neural Network (ANN) was built to predict diamond prices. 

Model Architecture 

Dense(64, relu) 
Dense(32, relu) 
Dense(1, linear) 

Evaluation Metrics 

Mean Absolute Error (MAE) 
Mean Squared Error (MSE) 
Root Mean Squared Error (RMSE) 
R² Score 

The model demonstrated strong predictive performance. 

Classification Model (Clarity Prediction) 

An ANN classifier was built to predict diamond clarity. 

Model Architecture 

Dense(64, relu) 
Dense(32, relu) 
Dense(number_of_classes, SoftMax) 

Evaluation Metrics 

Accuracy 
Precision 
Recall 
F1‑Score 
Confusion Matrix 

The model achieved solid accuracy and learned clarity patterns effectively. 

Customer Segmentation (Clustering) 

KMeans clustering was applied using scaled numeric features including price. 

Evaluation 

Silhouette Score 
Cluster visualization (carat vs price) 

Cluster Interpretation 

Cluster 0: High‑carat, high‑price diamonds 
Cluster 1: Mid‑range diamonds 
Cluster 2: Small, low‑price diamonds 
Cluster 3: Diamonds with unusual proportions 

These clusters represent natural market segments such as luxury, mid‑tier, and budget diamonds. 

AI Assistant 

A dataset‑powered assistant was built to: 

Accept natural language queries 
Filter diamonds based on query intent 
Provide human‑like explanations 

Example queries: 

“Show me expensive diamonds” 
“Find large premium diamonds” 
“Cheap diamonds with Ideal cut” 

Gradio Web Application 

The Gradio app includes: 

Price Prediction 
Predicts diamond price using the ANN regression model. 
Clarity Prediction 
Predicts diamond clarity using the ANN classification model. 
AI Assistant 
Searches the dataset and explains results in natural language. 

How to Run the Project 

Install dependencies 
Run the Gradio app 
python diamonds_app.py 
Access the application at:  http://127.0.0.1:7860 
 
Model Comparison Summary 

Regression Model
Evaluated using MAE, MSE, RMSE, and R². The model shows strong predictive accuracy.

Classification Model
Evaluated using Accuracy, Precision, Recall, and F1‑Score. The model performs well in predicting clarity categories.

Clustering Model
Evaluated using Silhouette Score. The clusters show clear segmentation into meaningful groups.

Conclusion 

This project demonstrates a complete machine learning workflow including EDA, preprocessing, ANN regression, ANN classification, clustering, and deployment using Gradio. It provides a full analytical system for diamond pricing, clarity prediction, and market segmentation. 

 