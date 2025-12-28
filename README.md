# Federated Learning Platform for Cross-Institutional Medical Diagnosis

A privacy-preserving federated learning platform enabling collaborative ML model training across multiple hospitals without centralizing raw patient data.

## Features

- **Federated Learning**: PyTorch + Flower-based distributed training
- **Privacy**: Differential Privacy (DP-SGD) via Opacus, Secure Aggregation
- **Database**: PostgreSQL with SQLAlchemy ORM for metadata/compliance tracking
- **API**: FastAPI with OAuth 2.0 authentication
- **Deployment**: Docker & Kubernetes ready

## Quick Start

### Prerequisites
- Python 3.10+
- PostgreSQL 14+
- Docker (optional)

### Installation

```bash
# Clone and setup
cd federated-medical-diagnosis
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Setup environment
copy .env.example .env
# Edit .env with your PostgreSQL credentials

# Run database migrations
alembic upgrade head

# Start the server
python -m server.main
```

### Running a Federated Training Session

```bash
# Terminal 1: Start FL Server
python -m server.main

# Terminal 2: Start Hospital Client 1
python -m clients.hospital_client --hospital-id hospital-1 --data-path ./data/hospital1

# Terminal 3: Start Hospital Client 2
python -m clients.hospital_client --hospital-id hospital-2 --data-path ./data/hospital2
```

## Project Structure

```
federated-medical-diagnosis/
├── server/           # Central Federation Server
├── clients/          # Hospital Client Nodes
├── database/         # PostgreSQL Layer
├── models/           # ML Models
├── privacy/          # DP-SGD, Secure Aggregation
├── orchestration/    # Round Management
├── monitoring/       # Compliance & Audit
├── config/           # Configuration
├── deployment/       # Docker & K8s
└── tests/            # Test Suite
```

## Privacy & Compliance

- **Differential Privacy**: ε-δ guarantees via DP-SGD
- **Secure Aggregation**: Encrypted model updates
- **Audit Logging**: Append-only compliance logs
- **HIPAA/GDPR**: Compliant metadata-only storage

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/hospitals` | POST | Register hospital |
| `/models` | POST | Create model |
| `/training/start` | POST | Start FL round |
| `/training/status/{round_id}` | GET | Get round status |
| `/metrics/{model_id}` | GET | Get model metrics |

## Configuration

See `.env.example` for all configuration options including:
- Database connection
- Privacy parameters (epsilon, delta, noise multiplier)
- Server settings
- Authentication secrets

## License

MIT License - See LICENSE file
