Sales Prediction Project

A web-based application for predicting sales using a Random Forest model. Features a responsive frontend with Tailwind CSS, Chart.js for visualization, and a Flask backend with JSON data exchange.

Prerequisites





Python 3.8+



Node.js (optional, for local development)



Web browser

Setup





Backend:





Navigate to backend/.



Install dependencies: pip install -r requirements.txt



Train the model: python model.py



Start Flask server: python app.py (runs on http://localhost:5000)



Frontend:





Open frontend/index.html in a browser or serve via Flask for production.



Data:





Sample data in backend/data/training_data.json. Replace with your own sales data (format: ad_spend, prev_sales, season, target_sales).

Usage





Enter advertising spend, previous sales, and select a season (Q1-Q4).



Click "Predict Sales" to get a prediction and view feature importance in a bar chart.



API endpoint: POST /predict (expects JSON: {"features": {"ad_spend": 1000, "prev_sales": 5000, "season": "Q1"}}).

Features





AI Model: Random Forest with feature scaling and seasonal encoding.



Responsive UI: Built with Tailwind CSS for mobile and desktop compatibility.



Visualization: Chart.js displays feature importance (e.g., how much ad spend impacts sales).



Error Handling: Input validation on frontend and backend.



CORS Support: Enabled for cross-origin requests.

Notes





Replace training_data.json with real sales data.



For production, deploy Flask with Gunicorn and serve frontend via a static file server (e.g., Nginx).



Enhance model with more features (e.g., customer demographics) for better accuracy.
