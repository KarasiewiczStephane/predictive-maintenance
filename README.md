# Predictive Maintenance

Equipment failure prediction system using machine learning on sensor data. Detects degradation patterns in temperature, vibration, pressure, and RPM readings to predict failures before they occur.

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────────┐
│  Sensor CSV  │───>│  DataLoader   │───>│  Feature Eng.  │───>│   Trainer    │
│   (input)    │    │  (validate)   │    │  (transform)   │    │  (RF/XGB)    │
└─────────────┘    └──────────────┘    └───────────────┘    └──────┬───────┘
                                                                    │
                                                                    v
┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────────┐
│   Client     │<──│  FastAPI      │<──│  Predictor     │<──│   Model      │
│  (request)   │    │  (REST API)   │    │  (inference)   │    │  (.joblib)   │
└─────────────┘    └──────────────┘    └───────────────┘    └──────────────┘
```

## Quick Start

```bash
# Clone and install
git clone git@github.com:KarasiewiczStephane/predictive-maintenance.git
cd predictive-maintenance
pip install -r requirements.txt

# Generate sample data
python scripts/generate_sample_data.py

# Train a model
make train

# Start the API
make serve
```

## Usage

### Training

```bash
# Train with default settings (Random Forest)
python -m src.main --train sample/equipment_sensors.csv

# Override model type via environment
MODEL_TYPE=xgboost python -m src.main --train sample/equipment_sensors.csv
```

### API Prediction

```bash
# Start the server
python -m src.main --serve

# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "equipment_id": "EQ-001",
    "temperature": 85.0,
    "vibration": 7.5,
    "pressure": 95.0,
    "rpm": 2800.0
  }'

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '[
    {"equipment_id": "EQ-001", "temperature": 85.0, "vibration": 7.5, "pressure": 95.0, "rpm": 2800.0},
    {"equipment_id": "EQ-002", "temperature": 72.0, "vibration": 4.2, "pressure": 102.0, "rpm": 3050.0}
  ]'
```

### Docker

```bash
# Build and run
docker compose up --build

# Or manually
docker build -t predictive-maintenance .
docker run -p 8000:8000 predictive-maintenance
```

## Project Structure

```
predictive-maintenance/
├── src/
│   ├── config.py              # Pydantic Settings configuration
│   ├── main.py                # CLI entry point (train/serve)
│   ├── data/
│   │   ├── loader.py          # CSV loading and validation
│   │   └── features.py        # Feature engineering pipeline
│   ├── models/
│   │   ├── trainer.py         # Model training (RF/XGBoost)
│   │   └── evaluation.py      # Metrics and threshold tuning
│   ├── api/
│   │   ├── server.py          # FastAPI endpoints
│   │   └── middleware.py       # Request logging middleware
│   └── utils/
│       └── logging.py         # Structured logging utilities
├── tests/                     # Unit and integration tests
├── configs/                   # YAML environment configs
├── scripts/                   # Data generation scripts
├── data/sample/               # Sample sensor data
├── .github/workflows/ci.yml   # CI/CD pipeline
├── Dockerfile                 # Multi-stage production build
├── docker-compose.yml         # Local development setup
├── Makefile                   # Common commands
├── pyproject.toml             # Ruff and pytest config
└── requirements.txt           # Python dependencies
```

## Configuration

All settings are configurable via environment variables or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `data` | Base directory for data files |
| `MODEL_DIR` | `models` | Directory for saved models |
| `MODEL_TYPE` | `random_forest` | Algorithm (`random_forest` or `xgboost`) |
| `TEST_SIZE` | `0.2` | Train/test split ratio |
| `N_ESTIMATORS` | `100` | Number of trees |
| `MAX_DEPTH` | `10` | Maximum tree depth |
| `WINDOW_SIZES` | `5,10,30` | Rolling window sizes for features |
| `API_HOST` | `0.0.0.0` | API server host |
| `API_PORT` | `8000` | API server port |
| `LOG_LEVEL` | `INFO` | Logging level |
| `LOG_JSON_FORMAT` | `false` | JSON structured logging |

See `.env.example` for the full list.

## Development

```bash
# Install dependencies
make install

# Run linter
make lint

# Run tests
make test

# Run tests with coverage report
make test-cov

# Clean up
make clean
```

## License

MIT
