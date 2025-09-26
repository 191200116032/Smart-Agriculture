## Development Environment

- **IDE**: PyCharm
- **Python Version**: 3.13.7

## Model Information

The crop recommendation model is trained and saved at: `models/crop_recommendation_saved_model.pkl`

To retrain and save the model again, run `main.py`:
```bash
python main.py
```
This will re-train the model using the dataset at `dataset/crop_recommendation/crop_recommendation.csv` and save the new model in the `models/` directory.

## Running the Application

1. First, install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run ui/app.py
   ```
