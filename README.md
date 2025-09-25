## Development Environment

- **IDE**: PyCharm
- **Python Version**: 3.13.7

## Model Information

The model is already trained and saved at: `core/saved_model.pkl`

To retrain and save the model again, run `main.py`:
```python
train_and_save_model()
```

## Running the Application

1. First, install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the app:
   ```bash
   streamlit run ui/app.py
   ```