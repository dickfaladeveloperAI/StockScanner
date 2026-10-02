import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.signal import argrelextrema
from google import genai
from google.genai import types

WATCHLIST = [
    {"symbol": "2330.TW", "name": "台積電"},
    {"symbol": "2317.TW", "name": "鴻海"},
    {"symbol": "2454.TW", "name": "聯發科"}
]

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("找不到 GEMINI_API_KEY 環境變數")

# 初始化全新 Google GenAI Client
client = genai.Client(api_key=API_KEY)

def analyze_stock(symbol, name):
    df = yf.download(symbol, period="150d", interval="1d", progress=False)
    if df.empty or len(df) < 60:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()

    order = 5
    high_idx = argrelextrema(df['High'].values, np.greater, order=order)[0]
    low_idx = argrelextrema(df['Low'].values, np.less, order=order)[0]

    pivots = []
    for idx in high_idx[-3:]:
        pivots.append({"type": "高點", "price": round(float(df['High'].iloc[idx]), 2), "date": str(df.index[idx].strftime('%Y-%m-%d'))})
    for idx in low_idx[-3:]:
        pivots.append({"type": "低點", "price": round(float(df['Low'].iloc[idx]), 2), "date": str(df.index[idx].strftime('%Y-%m-%d'))})
    pivots.sort(key=lambda x: x['date'])

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    prompt = f"""
你是一位專精波浪理論與台股價量結構的量化分析師。
請根據以下提供的 {name} ({symbol}) 技術特徵進行推算：
- 最新收盤價: {round(float(latest['Close']), 2)}
- 均線現況: 5MA={round(float(latest['MA5']), 2)}, 20MA={round(float(latest['MA20']), 2)}, 60MA={round(float(latest['MA60']), 2)}
- 近期成交量變動: 最新={int(latest['Volume'])}, 前一日={int(prev['Volume'])}
- 關鍵波段轉折序列 (近幾個波峰與波谷): {pivots}

請嚴格以繁體中文與 JSON 格式輸出：
{{
  "wave_status": "推測波浪位置 (如：主升第3浪推進中、第4浪修正、B浪反彈等)",
  "support_levels": [整數或浮點數支撐價1, 支撐價2],
  "resistance_levels": [整數或浮點數壓力價1, 壓力價2],
  "sentiment": "強勢多頭 / 震盪整理 / 偏空防守",
  "score": 1到10的數字 (買進評分),
  "rationale": "一句話簡評價量與型態核心邏輯"
}}
"""

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )
    analysis = json.loads(response.text)

    kline_data = []
    for idx, row in df.tail(80).iterrows():
        kline_data.append({
            "time": idx.strftime('%Y-%m-%d'),
            "open": round(float(row['Open']), 2),
            "high": round(float(row['High']), 2),
            "low": round(float(row['Low']), 2),
            "close": round(float(row['Close']), 2),
            "volume": int(row['Volume']),
            "ma5": round(float(row['MA5']), 2) if not np.isnan(row['MA5']) else None,
            "ma20": round(float(row['MA20']), 2) if not np.isnan(row['MA20']) else None,
            "ma60": round(float(row['MA60']), 2) if not np.isnan(row['MA60']) else None,
        })

    return {
        "symbol": symbol,
        "name": name,
        "latest_price": round(float(latest['Close']), 2),
        "analysis": analysis,
        "klines": kline_data
    }

def main():
    os.makedirs("docs", exist_ok=True)
    all_results = []
    for item in WATCHLIST:
        print(f"正在分析 {item['name']}...")
        res = analyze_stock(item['symbol'], item['name'])
        if res:
            all_results.append(res)

    with open("docs/data.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print("分析完成，資料已輸出至 docs/data.json")

if __name__ == "__main__":
    main()
