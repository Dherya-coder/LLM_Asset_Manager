from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from typing import Optional
import json
import re
from openai import AzureOpenAI  # Azure OpenAI SDK

# Load environment variables
load_dotenv()

app = FastAPI(title="Investment Allocation API")

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Azure OpenAI client
try:
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")  # model deployment name

    if not api_key or not endpoint or not deployment:
        raise ValueError("Azure OpenAI environment variables are not set properly")

    client = AzureOpenAI(
        api_key=api_key,
        api_version="2025-01-01-preview",  # latest stable
        azure_endpoint=endpoint
    )
except Exception as e:
    print(f"Error initializing Azure OpenAI client: {str(e)}")
    client = None

# Hardcoded market trends
MARKET_TRENDS = {
    "NIFTY 50": "Expected to increase by 10-12% in the next year due to strong economic growth and corporate earnings but riskier",
    "Gold": "Expected to increase by 8-10% in the next year due to global economic uncertainty and inflation concerns",
    "Bonds": "Expected to yield 6-7% in the next year with moderate risk",
    "Mutual Funds": "Expected to increase by 8-10% in the next year less risky due to diversification over multiple stocks by experts"
    # "Real Estate": "Expected to appreciate by 5-7% in major metropolitan areas",
    # "Cryptocurrency": "High volatility expected with potential 20-30% gains but significant risk",
    # "Fixed Deposits": "Offering 6-7% returns with minimal risk",
    # "International Markets": "Expected to grow by 10-12% with focus on US and European markets"
}

class InvestmentRequest(BaseModel):
    age: int
    investment_amount: float
    market_forecast: Optional[str] = None

class InvestmentResponse(BaseModel):
    allocation_percentage: dict
    recommendation: str

def extract_json_from_text(text):
    json_pattern = r'\{[\s\S]*\}'
    match = re.search(json_pattern, text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Fallback if JSON extraction fails
    allocation = {}
    for asset in MARKET_TRENDS.keys():
        pattern = rf"{asset}.*?(\d+)%"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            allocation[asset] = int(match.group(1))

    if allocation:
        return {"allocation": allocation, "explanation": text}

    return {
        "allocation": {"Cash": 100},
        "explanation": "Unable to parse specific allocations. Please review the full response for details."
    }

@app.post("/allocate-investment", response_model=InvestmentResponse)
async def allocate_investment(request: InvestmentRequest):
    if not client:
        raise HTTPException(status_code=500, detail="Azure OpenAI client not initialized properly")
        
    try:
        market_trends_text = "\n".join([f"- {asset}: {trend}" for asset, trend in MARKET_TRENDS.items()])
        
        prompt = f"""
        As a financial advisor, analyze the following investment scenario and provide allocation recommendations:
        
        Investor Age: {request.age}
        Investment Amount: ${request.investment_amount:,.2f}
        
        Current Market Trends and Forecasts (Next 1 Year):
        {market_trends_text}
        take additional information regarding returns from various asset classes from yfinance with tickers of each asset class as follows
    NIFTY 50 - "^NSEI", Gold - "GC=F" (Gold Futures), US 10Y Bonds - "^TNX", Mutual Funds - "^HDFCQUAL.BO".
        Additional Market Context:
        {request.market_forecast if request.market_forecast else "No additional market forecast provided"}
        
        Please provide:
        1. A detailed allocation percentage breakdown across different asset classes (stocks, bonds, cash, etc.)
        2. A brief explanation for the allocation strategy, taking into account the provided market trends
        
        Format your response as a JSON object with 'allocation' and 'explanation' fields.
        """

        # Azure OpenAI chat completion
        completion = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),  # deployment name
            messages=[
                {"role": "system", "content": "You are a professional financial advisor. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )

        response_text = completion.choices[0].message.content
        allocation_data = extract_json_from_text(response_text)

        return InvestmentResponse(
            allocation_percentage=allocation_data.get('allocation', {}),
            recommendation=allocation_data.get('explanation', '')
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
