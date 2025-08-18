# Investment Allocation API

This FastAPI backend provides investment allocation recommendations based on user's age and investment amount, using the Groq LLM API for intelligent financial advice.

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory and add your Groq API key:
```
GROQ_API_KEY=your_groq_api_key_here
```

4. Run the application:
```bash
uvicorn main:app --reload
```

## API Usage

### Get Investment Allocation

**Endpoint:** `POST /allocate-investment`

**Request Body:**
```json
{
    "age": 30,
    "investment_amount": 100000,
    "market_forecast": "Optional market forecast data"
}
```

**Response:**
```json
{
    "allocation_percentage": {
        "stocks": 60,
        "bonds": 30,
        "cash": 10
    },
    "recommendation": "Detailed explanation of the allocation strategy"
}
```

## API Documentation

Once the server is running, you can access the interactive API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
