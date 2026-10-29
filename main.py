
import os
import httpx
from io import BytesIO

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from google import genai
from google.genai import types

load_dotenv()

app = FastAPI(title="Stark AI Trading API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None


@app.get("/")
def health_check():
    return {
        "status": "online",
        "app": "Stark AI Trading API",
        "ai_configured": client is not None,
    }


@app.post("/analyze-chart")
async def analyze_chart(
    file: UploadFile = File(...),
    symbol: str = Form("EURUSD"),
):
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="Gemini API key is not configured in backend/.env",
        )

    if file.content_type not in {"image/png", "image/jpeg", "image/webp"}:
        raise HTTPException(
            status_code=400,
            detail="Upload a PNG, JPG, or WEBP chart image.",
        )

    image_bytes = await file.read()

    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller.")

    try:
        image = Image.open(BytesIO(image_bytes))
        image.verify()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image.thumbnail((1600, 1600))

        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        image_bytes = buffer.getvalue()
    except Exception:
        raise HTTPException(status_code=400, detail="The uploaded image could not be read.")
  
    live_context = "Live market quote unavailable; do not use live-price confirmation."

    allowed_symbols = {"EURUSD", "GBPUSD", "USDJPY", "XAUUSD"}

    if symbol.upper() in allowed_symbols:
        try:
            async with httpx.AsyncClient(timeout=10) as market_client:
                market_response = await market_client.get(
                    f"https://biquote.io/api/{symbol.upper()}"
                )
                market_response.raise_for_status()
                market_data = market_response.json()

            if not market_data.get("stale", True):
                live_context = (
                    f"Selected market: {symbol.upper()}\n"
                    f"Live bid: {market_data.get('bid', 'Unavailable')}\n"
                    f"Live ask: {market_data.get('ask', 'Unavailable')}\n"
                    f"Quote timestamp: {market_data.get('timestamp', 'Unavailable')}\n"
                    "Use this quote only as current market context, "
                    "not as proof of a BUY or SELL signal."
                )
        except (httpx.HTTPError, ValueError):
            pass
    
    prompt = """
    You are Stark AI Trading, an advanced technical chart analyst.

    Analyze only what is actually visible in the uploaded chart.

    Return this exact format:

    Signal: BUY / SELL / WAIT
    Symbol/Timeframe: Visible value or Not visible
    Trend: Bullish / Bearish / Sideways / Unclear
    Pattern: Identified pattern or None confirmed
    Support: Visible price level(s) or Unclear
    Resistance: Visible price level(s) or Unclear
    Entry: Price/zone or N/A
    SL: Price or N/A
    TP: Price or N/A
    Reason: One concise sentence
    Confidence: Low / Medium / High

    ANALYSIS RULES:
    - Examine candle bodies, wicks, and neighboring candles.
    - Check for patterns such as engulfing candles, hammer,
      shooting star, doji, and morning/evening star.
    - Name a pattern only when its shape and context support it.
    - If individual candles are too small or blurry to distinguish reliably,
      report "Pattern unclear" instead of guessing from the full chart.
    - Assess swing highs/lows, trend structure, support,
      resistance, and Fibonacci levels only when clearly labeled.
    - Never invent or mislabel Fibonacci ratios. If a ratio or
      its anchor points are unclear, do not name the ratio.
    - Distinguish a possible pattern from a confirmed signal.
    - Use WAIT when evidence is mixed or confirmation is missing.
    - Never invent prices, indicators, or chart details.
    - Entry, SL, and TP must use readable chart levels and
      a logically consistent setup; otherwise return N/A. 
    - Do not claim to know live prices from a screenshot.
    - Use LIVE MARKET CONTEXT to compare the selected market's current bid/ask with the chart. State whether the quote supports, conflicts with, or cannot confirm the chart bias. Never change BUY/SELL to WAIT based on the quote alone, and never invent live prices.
    - Keep the entire response under 50 words.
    - Be concise, objective, and honest about uncertainty.
    - No trade outcome is guaranteed.
    """


    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg",
                ),
                f"{prompt}\n\nLIVE MARKET CONTEXT:\n{live_context}",
            ],
        )
        return {"analysis": response.text or "The AI returned no analysis."}
    except Exception as e:
        print(f"Gemini analysis error: {type(e).__name__}: {e}", flush=True)
        raise HTTPException(
            status_code=502,
            detail=f"Gemini error: {type(e).__name__}: {e}",
        )

@app.get("/market-price/{symbol}")
async def market_price(symbol: str):
    symbol = symbol.upper()

    allowed_symbols = {"EURUSD", "GBPUSD", "USDJPY", "XAUUSD"}
    if symbol not in allowed_symbols:
        raise HTTPException(
            status_code=400,
            detail="Unsupported symbol"
        )

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                f"https://biquote.io/api/{symbol}"
            )
            response.raise_for_status()
            data = response.json()

        if data.get("stale", True):
            raise HTTPException(
                status_code=503,
                detail="Market quote is stale"
            )

        return data

    except httpx.HTTPError:
        raise HTTPException(
            status_code=502,
            detail="Could not fetch market price"
        )
@app.get("/market-candles/{symbol}")
async def market_candles(
    symbol: str,
    interval: str = "15m",
    limit: int = 100,
):
    symbol = symbol.upper()

    allowed_symbols = {"EURUSD", "GBPUSD", "USDJPY", "XAUUSD"}
    allowed_intervals = {"1m", "5m", "15m", "30m", "1h", "4h", "1d"}

    if symbol not in allowed_symbols:
        raise HTTPException(
            status_code=400,
            detail="Unsupported symbol",
        )

    if interval not in allowed_intervals:
        raise HTTPException(
            status_code=400,
            detail="Unsupported candle interval",
        )

    if not 1 <= limit <= 200:
        raise HTTPException(
            status_code=400,
            detail="Limit must be between 1 and 200",
        )

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                f"https://biquote.io/api/{symbol}/ohlc",
                params={"interval": interval, "limit": limit},
            )
            response.raise_for_status()
            data = response.json()

        return data

    except httpx.HTTPError:
        raise HTTPException(
            status_code=502,
            detail="Could not fetch market candles",
        )