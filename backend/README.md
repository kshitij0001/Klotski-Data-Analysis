# Klotski Data Science Backend

This backend provides:
- A Python solver for the Klotski puzzle
- Data generation for machine learning/EDA
- A FastAPI server for frontend integration

## Usage

### CLI


# Klotski Data Science Backend

This backend provides:
- A Python solver for the Klotski puzzle
- Data generation for machine learning/EDA
- A FastAPI server for frontend integration

## Usage

### CLI
python -m klotski --preset

### API
Run the server:
uvicorn api.app:app --reload

POST to `/solve` with a board config.

### Data Generation
See [`scripts/generate_data.py`](file:///backend/scripts/generate_data.py) for dataset creation.

## Requirements
See [`requirements.txt`](file:///backend/requirements.txt).

## Project Structure
- `klotski/` — core logic
- `api/` — FastAPI server
- `data/` — generated datasets
- `dashboard/` — Streamlit analytics (optional)