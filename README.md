# Medical Insurance Cost Prediction

A machine learning application that predicts medical insurance costs from user-provided information. It uses a Streamlit frontend, a FastAPI backend, a trained machine learning model, and Docker Compose to run the services together.

## Project Overview

A user enters insurance-related features through the web interface. The frontend sends the input to the backend API, which prepares the data and requests a prediction from the trained model. The prediction is returned to the interface.

### Architecture

```text
User
 |
 v
Streamlit Frontend (:8501)
 |
 | POST /predict
 v
FastAPI Backend (:8000)
 |
 v
Input preparation (Pandas DataFrame)
 |
 v
Trained ML Model
 |
 v
Predicted insurance cost
```

Docker Compose runs the frontend and backend as separate services.

## Tech Stack

- **Python** — application and machine learning code
- **Streamlit** — user interface
- **FastAPI** — prediction API
- **Uvicorn** — ASGI server for the API
- **Pandas** — input data preparation
- **Docker** — containerization
- **Docker Compose** — running the services together
- **scikit-learn / joblib** — include these if they are used by the model and listed in your dependencies

## Features

- Web interface for entering prediction inputs
- Backend API endpoint at `POST /predict`
- Converts incoming input into a one-row, six-feature DataFrame, as shown in the successful test log
- Loads the trained model when the backend starts
- Returns a predicted insurance cost
- Runs frontend and backend in separate Docker containers

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) with Docker Compose
- Git, if you want to clone the repository

## Run with Docker Compose

1. Clone the repository:

   ```bash
   git clone <YOUR_GITHUB_REPOSITORY_URL>
   cd Ml_med_Insurance
   ```

2. Ensure the repository includes the Compose file, both Dockerfiles, frontend and backend source code, model file, and dependencies.

3. Build the images:

   ```bash
   docker compose build
   ```

4. Start the application:

   ```bash
   docker compose up
   ```

5. Open these URLs:

   - **Streamlit frontend:** http://localhost:8501
   - **FastAPI interactive documentation:** http://localhost:8000/docs

To stop the services, press `Ctrl+C`. If you started them in detached mode, run:

```bash
docker compose down
```

To rebuild after code or dependency changes:

```bash
docker compose up --build
```

## API

The prediction endpoint is:

```http
POST /predict
```

The exact request schema and response format are defined in the backend code. While the application is running, visit http://localhost:8000/docs to inspect the API schema and test a request.

## Verification

A local Docker test successfully completed a prediction request. The backend logs showed:

- The trained model loaded successfully.
- Input was converted into a DataFrame with shape `(1, 6)`.
- The prediction completed successfully.
- The API returned `200 OK` for `POST /predict`.

This verifies that the tested frontend-to-backend prediction flow worked in that run. It does **not** establish model accuracy, generalization, or production readiness.

## Configuration and Troubleshooting

- **Frontend URL:** Open `http://localhost:8501` in your browser. `0.0.0.0` is the address the server listens on; it is not normally the browser URL to open.
- **Backend URL inside Docker:** When both services share the Compose network, the frontend should use the backend service name and port, for example `http://backend:8000/predict`, if `backend` is the service name in `docker-compose.yml`.
- **Module import errors:** Linux containers are case-sensitive. Make sure the startup module path matches directory and file names exactly.
- **Missing dependencies:** Ensure packages imported by your application are included in the dependency file used by the Docker build.
- **Compose `version` warning:** Recent Docker Compose versions ignore the top-level `version` field. It can be removed from the Compose file.

## Project Structure

Adjust this example tree to match your actual repository before publishing:

```text
Ml_med_Insurance/
├── Backend/
│   ├── main.py
│   └── predictor.py
├── Frontend/
│   └── app.py
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
├── requirements.txt
└── README.md
```

Include any additional directories containing the trained model or other source files.

## Future Improvements

- Add input validation and clearer error messages.
- Test edge cases and invalid requests.
- Evaluate and document model performance using appropriate metrics.
- Add automated tests for the API and prediction pipeline.
- Add CI checks and deploy the application to a hosting platform.
- Document the training data, model limitations, and intended use.

## Disclaimer

This project is for educational and demonstration purposes. Predictions are estimates from a machine learning model and should not be treated as professional insurance, medical, or financial advice.

## Author

**Chukwuneta Emmanuel**

- GitHub: [Your GitHub Profile](https://github.com/Emblogs)
