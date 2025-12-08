from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, validator, ConfigDict
import uvicorn
import pickle
import numpy as np
import pandas as pd
from typing import List
from datetime import datetime

MODEL_PATH = "models/best_model.pkl"
THRESHOLD = 0.36  # Optimal threshold from the latest evaluation (≈0.3617)

app = FastAPI(
    title="🌌 Pulsar Star Classification API",
    version="1.2.0",
    contact={
        "name": "Michael Chadolias",
        "url": "https://github.com/mchadolias/detect-pulsar-api",
    },
    docs_url="/docs",
    redoc_url="/redoc",
)


with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Use the exact feature names from the error message
FEATURE_NAMES = [
    "ip_mean",  # Integrated Profile Mean
    "ip_std",  # Integrated Profile Standard Deviation
    "ip_kurtosis",  # Integrated Profile Excess Kurtosis
    "ip_skewness",  # Integrated Profile Skewness
    "dm_mean",  # DM-SNR Curve Mean
    "dm_std",  # DM-SNR Curve Standard Deviation
    "dm_kurtosis",  # DM-SNR Curve Excess Kurtosis
    "dm_skewness",  # DM-SNR Curve Skewness
]


class HTRUInputSingle(BaseModel):
    features: List[float] = Field(
        ...,
        description="List of 8 numerical HTRU2 features for a single sample.",
    )

    @validator("features")
    def validate_features(cls, v):
        if len(v) != 8:
            raise ValueError("Exactly 8 features are required")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "features": [
                        99.3671875,
                        41.57220208,
                        1.547196967,
                        4.154106043,
                        27.55518395,
                        61.71901588,
                        2.20880796,
                        3.662680136,
                    ]
                }
            ]
        }
    )


class HTRUInputBatch(BaseModel):
    samples: List[List[float]] = Field(
        ...,
        description="List of samples. Each sample contains 8 numerical HTRU2 features.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "samples": [
                    [
                        99.3671875,
                        41.57220208,
                        1.547196967,
                        4.154106043,
                        27.55518395,
                        61.71901588,
                        2.20880796,
                        3.662680136,
                    ],
                    [140.0, 45.0, 1.8, 3.9, 25.0, 60.0, 2.1, 3.5],
                    [80.0, 35.0, 0.5, 2.0, 30.0, 50.0, 1.5, 2.5],
                ]
            }
        }

    @validator("samples")
    def validate_samples(cls, v):
        for i, sample in enumerate(v):
            if len(sample) != 8:
                raise ValueError(f"Sample {i} must contain exactly 8 features, got {len(sample)}")
        return v


class PredictionResponse(BaseModel):
    probability: float = Field(ge=0.0, le=1.0)
    is_pulsar: bool

    class Config:
        json_schema_extra = {
            "examples": [
                {"probability": 0.96047443151474, "is_pulsar": True},
                {"probability": 0.123456789, "is_pulsar": False},
            ]
        }


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
                    {"probability": 0.96047443151474, "is_pulsar": True},
                    {"probability": 0.9847214818000793, "is_pulsar": True},
                    {"probability": 0.123456789, "is_pulsar": False},
                ]
            }
        }


def predict_single(features: List[float]) -> float:
    """Predict probability for a single sample"""
    # Convert to DataFrame with proper column names
    features_df = pd.DataFrame([features], columns=FEATURE_NAMES)
    probability = model.predict_proba(features_df)[0, 1]
    return float(probability)


def predict_batch(samples: List[List[float]]) -> List[float]:
    """
    Predict probabilities for multiple samples.

    Args:
        samples: List of samples, where each sample is a list of 8 numerical features
                in the order: [ip_mean, ip_std, ip_kurtosis, ip_skewness,
                              dm_mean, dm_std, dm_kurtosis, dm_skewness]

    Returns:
        List of probabilities (0.0 to 1.0) for each sample being a pulsar

    Raises:
        ValueError: If samples have incorrect dimensions or feature count
        Exception: If model prediction fails

    Example:
        >>> samples = [
        ...     [99.3671875, 41.57220208, 1.547196967, 4.154106043, 27.55518395, 61.71901588, 2.20880796, 3.662680136],
        ...     [140.0, 45.0, 1.8, 3.9, 25.0, 60.0, 2.1, 3.5],
        ...     [80.0, 35.0, 0.5, 2.0, 30.0, 50.0, 1.5, 2.5]
        ... ]
        >>> probabilities = predict_batch(samples)
        >>> print(probabilities)
        [0.96047443151474, 0.9847214818000793, 0.123456789]
        >>> [prob >= 0.5 for prob in probabilities]
        [True, True, False]
    """
    try:
        # Validate input dimensions
        if not samples:
            raise ValueError("Empty samples list provided")

        # Check all samples have exactly 8 features
        for i, sample in enumerate(samples):
            if len(sample) != 8:
                raise ValueError(f"Sample {i} has {len(sample)} features, expected 8")

        # Convert to DataFrame with proper column names
        samples_df = pd.DataFrame(samples, columns=FEATURE_NAMES)

        # Get probabilities for class 1 (pulsar)
        probabilities = model.predict_proba(samples_df)[:, 1]

        # Convert numpy floats to Python floats for JSON serialization
        return [float(prob) for prob in probabilities]

    except ValueError as ve:
        # Re-raise validation errors with clear messages
        raise ve
    except Exception as e:
        # Handle model prediction errors
        raise Exception(f"Model prediction failed: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
def predict_single_sample(payload: HTRUInputSingle) -> PredictionResponse:
    """
    Predict if a single sample is a pulsar.

    ## 📊 Expected Input Format

    The sample must contain 8 numerical HTRU2 features in this exact order:

    1. **ip_mean** - Mean of the integrated profile
    2. **ip_std** - Standard deviation of the integrated profile
    3. **ip_kurtosis** - Excess kurtosis of the integrated profile
    4. **ip_skewness** - Skewness of the integrated profile
    5. **dm_mean** - Mean of the DM-SNR curve
    6. **dm_std** - Standard deviation of the DM-SNR curve
    7. **dm_kurtosis** - Excess kurtosis of the DM-SNR curve
    8. **dm_skewness** - Skewness of the DM-SNR curve

    ## 🚀 Example Request

    ```json
    {
      "features": [
        99.3671875,
        41.57220208,
        1.547196967,
        4.154106043,
        27.55518395,
        61.71901588,
        2.20880796,
        3.662680136
      ]
    }
    ```

    ## 📈 Example Response

    ```json
    {
      "probability": 0.96047443151474,
      "is_pulsar": true
    }
    ```

    ## 💡 Performance Notes

    - **High ROC–AUC**: 0.9747 on the held-out test set
    - **Recall-focused**: F₂-score 0.8923 and recall 0.8994 at threshold ≈ 0.36
    - **Balanced performance**: F₁-score 0.8819 with precision 0.8651
    - **Fast Inference**: Real-time prediction capabilities

    ## 🎯 Interpretation Guide

    - **Probability ≥ 0.36**: Classified as pulsar (`is_pulsar: true`)
    - **Probability < 0.36**: Classified as non-pulsar (`is_pulsar: false`)
    - **Confidence Levels**:
      - 0.8-1.0: High confidence pulsar
      - 0.6-0.8: Moderate confidence pulsar
      - 0.36-0.6: Borderline / low-confidence pulsar
      - 0.2-0.36: Borderline non-pulsar
      - 0.0-0.2: High confidence non-pulsar

    ## 🔬 Test Cases

    - **High probability pulsar**: Features showing strong pulsar characteristics
    - **Low probability**: Features typical of noise or RFI artifacts
    - **Borderline cases**: Samples near the decision boundary (0.4-0.6)
    """
    try:
        probability = predict_single(payload.features)
        return PredictionResponse(
            probability=probability,
            is_pulsar=probability >= THRESHOLD,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.post("/predict_batch", response_model=BatchPredictionResponse)
def predict_batch_samples(payload: HTRUInputBatch) -> BatchPredictionResponse:
    """
    Predict multiple samples at once.

    ## 📊 Expected Input Format

    Each sample must contain 8 numerical HTRU2 features in this exact order:

    1. **ip_mean** - Mean of the integrated profile
    2. **ip_std** - Standard deviation of the integrated profile
    3. **ip_kurtosis** - Excess kurtosis of the integrated profile
    4. **ip_skewness** - Skewness of the integrated profile
    5. **dm_mean** - Mean of the DM-SNR curve
    6. **dm_std** - Standard deviation of the DM-SNR curve
    7. **dm_kurtosis** - Excess kurtosis of the DM-SNR curve
    8. **dm_skewness** - Skewness of the DM-SNR curve

    ## 🚀 Example Request

    ```json
    {
      "samples": [
        [99.3671875, 41.57220208, 1.547196967, 4.154106043, 27.55518395, 61.71901588, 2.20880796, 3.662680136],
        [140.0, 45.0, 1.8, 3.9, 25.0, 60.0, 2.1, 3.5],
        [80.0, 35.0, 0.5, 2.0, 30.0, 50.0, 1.5, 2.5]
      ]
    }
    ```

    ## 📈 Example Response

    ```json
    {
      "predictions": [
        {
          "probability": 0.96047443151474,
          "is_pulsar": true
        },
        {
          "probability": 0.9847214818000793,
          "is_pulsar": true
        },
        {
          "probability": 0.123456789,
          "is_pulsar": false
        }
      ]
    }
    ```

    ## 💡 Performance Notes

    - **Same Metrics**: Batch mode uses the same Random Forest model and threshold (≈ 0.36)
    - **Test Set**: F₂ = 0.8923, F₁ = 0.8819, ROC–AUC = 0.9747, PR–AUC = 0.9306
    - **Order Preservation**: Response order exactly matches input sample order
    - **Ideal Use Cases**: Processing multiple telescope observations, dataset validation

    ## 🎯 Interpretation Guide

    - **Probability ≥ 0.36**: Classified as pulsar (`is_pulsar: true`)
    - **Probability < 0.36**: Classified as non-pulsar (`is_pulsar: false`)
    - **Confidence Levels**:
      - 0.8-1.0: High confidence pulsar
      - 0.6-0.8: Moderate confidence pulsar
      - 0.36-0.6: Borderline / low-confidence pulsar
      - 0.2-0.36: Borderline non-pulsar
      - 0.0-0.2: High confidence non-pulsar

    """
    try:
        probabilities = predict_batch(payload.samples)
        predictions = [
            PredictionResponse(probability=prob, is_pulsar=prob >= THRESHOLD)
            for prob in probabilities
        ]
        return BatchPredictionResponse(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {str(e)}")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "imbalance_treatment": "cost-sensitive learning with class weights",
        "performance": {
            "threshold": THRESHOLD,
            "f2_score": 0.8923,
            "f1_score": 0.8819,
            "recall": 0.8994,
            "precision": 0.8651,
            "roc_auc": 0.9747,
            "pr_auc": 0.9306,
        },
        "dataset_info": {
            "total_samples": 17898,
            "class_distribution": {"non_pulsars": 0.9084, "pulsars": 0.0916},
            "imbalance_ratio": "9.9:1",
            "background": "HTRU2 Pulsar Dataset",
        },
    }


@app.get("/features")
def get_feature_names():
    """Endpoint to check what feature names the model expects with example data"""
    return {
        "feature_names": FEATURE_NAMES,
        "descriptions": {
            "ip_mean": "Mean of the integrated profile",
            "ip_std": "Standard deviation of the integrated profile",
            "ip_kurtosis": "Excess kurtosis of the integrated profile",
            "ip_skewness": "Skewness of the integrated profile",
            "dm_mean": "Mean of the DM-SNR curve",
            "dm_std": "Standard deviation of the DM-SNR curve",
            "dm_kurtosis": "Excess kurtosis of the DM-SNR curve",
            "dm_skewness": "Skewness of the DM-SNR curve",
        },
        "example_data": {
            "single_prediction_example": {
                "description": "Copy and paste this example for /predict endpoint",
                "curl_command": """curl -X 'POST' 'http://localhost:9696/predict' \\
  -H 'Content-Type: application/json' \\
  -d '{
    "features": [
      99.3671875,
      41.57220208, 
      1.547196967,
      4.154106043,
      27.55518395,
      61.71901588,
      2.20880796,
      3.662680136
    ]
  }'""",
                "payload": {
                    "features": [
                        99.3671875,
                        41.57220208,
                        1.547196967,
                        4.154106043,
                        27.55518395,
                        61.71901588,
                        2.20880796,
                        3.662680136,
                    ]
                },
            },
            "batch_prediction_example": {
                "description": "Copy and paste this example for /predict_batch endpoint",
                "curl_command": """curl -X 'POST' 'http://localhost:9696/predict_batch' \\
  -H 'Content-Type: application/json' \\
  -d '{
    "samples": [
      [99.3671875, 41.57220208, 1.547196967, 4.154106043, 27.55518395, 61.71901588, 2.20880796, 3.662680136],
      [140.0, 45.0, 1.8, 3.9, 25.0, 60.0, 2.1, 3.5],
      [80.0, 35.0, 0.5, 2.0, 30.0, 50.0, 1.5, 2.5]
    ]
  }'""",
                "payload": {
                    "samples": [
                        [
                            99.3671875,
                            41.57220208,
                            1.547196967,
                            4.154106043,
                            27.55518395,
                            61.71901588,
                            2.20880796,
                            3.662680136,
                        ],
                        [140.0, 45.0, 1.8, 3.9, 25.0, 60.0, 2.1, 3.5],
                        [80.0, 35.0, 0.5, 2.0, 30.0, 50.0, 1.5, 2.5],
                    ]
                },
            },
            "test_cases": {
                "high_probability_pulsar": {
                    "description": "Features that typically indicate a pulsar (high probability)",
                    "features": [
                        99.3671875,
                        41.57220208,
                        1.547196967,
                        4.154106043,
                        27.55518395,
                        61.71901588,
                        2.20880796,
                        3.662680136,
                    ],
                },
                "low_probability_pulsar": {
                    "description": "Features that typically indicate non-pulsar (low probability)",
                    "features": [80.0, 35.0, 0.5, 2.0, 30.0, 50.0, 1.5, 2.5],
                },
                "borderline_case": {
                    "description": "Features that are close to the decision boundary",
                    "features": [110.0, 40.0, 1.2, 3.5, 26.0, 55.0, 1.8, 3.0],
                },
            },
        },
    }


@app.get("/examples")
def get_quick_examples():
    """Quick testing examples using JSON files"""
    return {
        "quick_test_commands": {
            # GET endpoints (information retrieval)
            "health_check": "curl http://localhost:9696/health",
            "get_features": "curl http://localhost:9696/features",
            "get_examples": "curl http://localhost:9696/examples",
            # POST endpoints using JSON files only
            "single_prediction": "curl -X POST 'http://localhost:9696/predict' -H 'Content-Type: application/json' -d @examples/single_prediction.json",
            "batch_prediction": "curl -X POST 'http://localhost:9696/predict_batch' -H 'Content-Type: application/json' -d @examples/batch_prediction.json",
            "test_cases": "curl -X POST 'http://localhost:9696/predict_batch' -H 'Content-Type: application/json' -d @examples/test_cases.json",
        },
        "available_json_files": {
            "examples/single_prediction.json": "Single sample prediction with high-probability pulsar features",
            "examples/batch_prediction.json": "Batch prediction with mixed pulsar and non-pulsar samples",
            "examples/test_cases.json": "Multiple test cases including high/low probability and borderline samples",
        },
        "json_file_structure": {
            "single_prediction.json": {
                "format": '{"features": [f1, f2, f3, f4, f5, f6, f7, f8]}',
                "example": '{"features": [99.367, 41.572, 1.547, 4.154, 27.555, 61.719, 2.208, 3.662]}',
            },
            "batch_prediction.json": {
                "format": '{"samples": [[f1..f8], [f1..f8], ...]}',
                "example": '{"samples": [[99.367, 41.572, 1.547, 4.154, 27.555, 61.719, 2.208, 3.662], [80.0, 35.0, 0.5, 2.0, 30.0, 50.0, 1.5, 2.5]]}',
            },
        },
        "usage_instructions": [
            "1. Download the JSON files from the examples directory",
            "2. Use curl with -d @filename.json to send the file content",
            "3. All JSON files are pre-configured with valid test data",
            "4. Modify the JSON files to test with your own feature values",
        ],
        "important_notes": [
            "Use GET for information endpoints (/health, /features, /examples)",
            "Use POST for prediction endpoints (/predict, /predict_batch)",
            "All JSON files contain properly formatted 8-feature samples",
            "Files are located in the 'examples/' directory",
        ],
    }


@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🌌 Pulsar Classification API</title>
        <style>
            body { 
                font-family: 'Segoe UI', Arial, sans-serif; 
                max-width: 1200px; 
                margin: 0 auto; 
                padding: 20px; 
                background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
                color: white;
            }
            .header { 
                text-align: center; 
                padding: 40px 0; 
                background: rgba(255,255,255,0.1); 
                border-radius: 15px;
                margin-bottom: 30px;
            }
            .card { 
                background: rgba(255,255,255,0.1); 
                padding: 20px; 
                margin: 15px 0; 
                border-radius: 10px; 
                border-left: 4px solid #4CAF50;
            }
            .endpoint { 
                background: rgba(0,0,0,0.3); 
                padding: 15px; 
                margin: 10px 0; 
                border-radius: 8px; 
                font-family: monospace;
            }
            .btn { 
                display: inline-block; 
                padding: 10px 20px; 
                margin: 5px; 
                background: #4CAF50; 
                color: white; 
                text-decoration: none; 
                border-radius: 5px; 
            }
            .metrics { 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                gap: 15px; 
                margin: 20px 0; 
            }
            .metric-card { 
                background: rgba(255,255,255,0.15); 
                padding: 15px; 
                border-radius: 8px; 
                text-align: center;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🌌 Pulsar Star Classification API</h1>
            <p>Machine Learning API for detecting pulsar stars with high recall and ROC–AUC ≈ 0.97</p>
        </div>
        
        <div class="card">
            <h2>🚀 Quick Start</h2>
            <p>Test the API with these endpoints:</p>
            <a class="btn" href="/docs">API Documentation</a>
            <a class="btn" href="/redoc">Alternative Docs</a>
            <a class="btn" href="/health">Health Check</a>
            <a class="btn" href="/examples">Examples</a>
        </div>
        
        <div class="card">
            <h2>📊 Model Performance (Test Set)</h2>
            <div class="metrics">
                <div class="metric-card">
                    <h3>F₂-score</h3>
                    <p>0.8923</p>
                    <small>Recall-focused metric</small>
                </div>
                <div class="metric-card">
                    <h3>Recall</h3>
                    <p>89.9%</p>
                    <small>Pulsars detected</small>
                </div>
                <div class="metric-card">
                    <h3>Precision</h3>
                    <p>86.5%</p>
                    <small>Correct pulsar IDs</small>
                </div>
                <div class="metric-card">
                    <h3>ROC–AUC</h3>
                    <p>0.9747</p>
                    <small>Discrimination power</small>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>🔌 API Endpoints</h2>
            <div class="endpoint">
                <strong>POST</strong> /predict - Single prediction
            </div>
            <div class="endpoint">
                <strong>POST</strong> /predict_batch - Batch predictions
            </div>
            <div class="endpoint">
                <strong>GET</strong> /features - Feature specifications
            </div>
            <div class="endpoint">
                <strong>GET</strong> /health - Service status
            </div>
            <div class="endpoint">
                <strong>GET</strong> /examples - Predefined test examples
            </div>
        </div>
        
        <div class="card">
            <h2>🔬 Technical Details</h2>
            <p><strong>Model:</strong> Random Forest classifier (threshold ≈ 0.36, F₂-optimised)</p>
            <p><strong>Dataset:</strong> HTRU2 Pulsar Dataset (17,898 samples)</p>
            <p><strong>Features:</strong> 8 radio telescope measurements</p>
            <p><strong>Deployment:</strong> Docker + FastAPI</p>
        </div>


        <div class="card">
            <h2>📁 GitHub Project</h2>
            <p><strong>Repository:</strong> 
                <a href="https://github.com/mchadolias/detect-pulsar-api" target="_blank" style="color: #4CAF50;">
                    github.com/mchadolias/detect-pulsar-api
                </a>
            </p>
            <p><strong>Description:</strong> ML classification for pulsar detection from radio telescope data</p>
            <p><strong>Features:</strong></p>
            <ul style="margin-left: 20px;">
                <li>Complete ML pipeline from data acquisition to deployment</li>
                <li>FastAPI REST API with real-time inference</li>
                <li>Docker containerization for easy deployment</li>
                <li>Comprehensive documentation and examples</li>
            </ul>
            <div style="margin-top: 15px;">
                <a href="https://github.com/mchadolias/detect-pulsar-api" target="_blank" class="btn" style="background: #333; color: white;">
                    📂 View on GitHub
                </a>
                <a href="https://github.com/mchadolias/detect-pulsar-api/issues" target="_blank" class="btn" style="background: #6e5494; color: white;">
                    🐛 Report Issues
                </a>
            </div>
        </div>
    </body>
    </html>
    """


if __name__ == "__main__":
    uvicorn.run("predict:app", host="0.0.0.0", port=9696)
