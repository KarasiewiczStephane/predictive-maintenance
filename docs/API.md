# API Documentation

Base URL: `http://localhost:8000`

Interactive docs available at: `http://localhost:8000/docs` (Swagger UI)

## Endpoints

### GET /health

Check service health and model availability.

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### POST /predict

Predict equipment failure from a single sensor reading.

**Request Body:**

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `equipment_id` | string | required | Equipment identifier |
| `temperature` | float | -50 to 200 | Temperature reading |
| `vibration` | float | 0 to 100 | Vibration reading |
| `pressure` | float | 0 to 500 | Pressure reading (PSI) |
| `rpm` | float | 0 to 10000 | Rotations per minute |

**Example:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "equipment_id": "EQ-001",
    "temperature": 85.0,
    "vibration": 7.5,
    "pressure": 95.0,
    "rpm": 2800.0
  }'
```

**Response:**

```json
{
  "equipment_id": "EQ-001",
  "failure_probability": 0.7234,
  "prediction": "failure_risk",
  "confidence": 0.4468
}
```

### POST /predict/batch

Predict failure for multiple sensor readings.

**Request Body:** Array of sensor readings (same schema as `/predict`).

**Response:** Array of prediction responses.

## Error Responses

| Status | Description |
|--------|-------------|
| 422 | Validation error (invalid input values) |
| 503 | Model not loaded |

## Response Headers

| Header | Description |
|--------|-------------|
| `X-Request-ID` | Unique request tracking identifier |
