# Quick Start Guide

## 🎯 Quick Setup (5 minutes)

### Step 1: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Setup Environment
```bash
cp .env.example .env
```

### Step 4: Create Logs Directory
```bash
mkdir -p logs
```

## 🚀 Running the Project

### Option A: Run Complete Pipeline
```bash
python main.py
```
This runs the end-to-end analysis pipeline on sample synthetic data.

### Option B: Run Streamlit Dashboard
```bash
streamlit run examples/streamlit_dashboard.py
```
Launches interactive web dashboard at `http://localhost:8501`

### Option C: Start API Server
```bash
uvicorn src.api_server:app --reload
```
API available at `http://localhost:8000`
- Docs: `http://localhost:8000/docs`

### Option D: Docker Compose (All Services)
```bash
docker-compose up
```
Starts:
- PostgreSQL (port 5432)
- API Server (port 8000)
- Streamlit Dashboard (port 8501)

## 📊 Next Steps

1. **Add Your Data**
   - Place CSV files in `data/raw/`
   - Update code to load your data

2. **Customize Processing**
   - Edit `src/data_processor.py` for your needs
   - Add custom features in `src/analytics.py`

3. **Train Models**
   - Modify `src/ml_models.py`
   - Add new algorithms

4. **Create Reports**
   - Use `src/visualizer.py` for charts
   - Export to `reports/` folder

5. **Deploy**
   - Update `Dockerfile`
   - Push to cloud (AWS, GCP, Azure)

## 📚 Key Modules

| Module | Purpose |
|--------|---------|
| `data_loader.py` | Load CSV, Excel, SQL data |
| `data_processor.py` | Cleaning, normalization, feature engineering |
| `analytics.py` | Statistical analysis, grouping, tests |
| `ml_models.py` | Training and evaluation |
| `visualizer.py` | Matplotlib, Seaborn, Plotly charts |
| `api_server.py` | FastAPI REST endpoints |

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_models.py::TestMLModels -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## 🐛 Troubleshooting

### Import Errors
```bash
# Add project to path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Missing Logs Directory
```bash
mkdir -p logs
```

### Database Connection Issues
```bash
# Update .env with correct DATABASE_URL
# PostgreSQL: postgresql://user:password@localhost:5432/dbname
# SQLite: sqlite:///data.db
```

## 📖 Example Usage

```python
from src.data_loader import DataLoader
from src.data_processor import DataProcessor
from src.analytics import Analytics
from src.ml_models import MLModels

# Load data
df = DataLoader.load_csv('data/raw/mydata.csv')

# Process
df_clean = DataProcessor.remove_duplicates(df)

# Analyze
stats = Analytics.descriptive_stats(df_clean)

# Model
X_train, X_test, y_train, y_test = MLModels.train_test_split_data(X, y)
model = MLModels.train_xgboost(X_train, y_train)
metrics = MLModels.evaluate_classification(model, X_test, y_test)
```

## 📞 Support

See README.md for detailed documentation.
