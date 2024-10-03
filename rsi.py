import pyupbit
import pandas as pd
import time

def get_rsi(df, period=14):
    """RSI 계산 함수"""
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def get_current_rsi(ticker, period=14):
    """실시간 1분봉 기준 RSI 조회"""
    df = pyupbit.get_ohlcv(ticker, interval="minute1", count=period+1)  # 1분봉 데이터 가져오기
    rsi = get_rsi(df, period).iloc[-1]  # 최신 RSI 값 계산
    return rsi

# 1초 간격으로 RSI 값을 출력하는 코드
ticker = "KRW-BTC"  # 원하는 암호화폐 티커로 변경 가능

while True:
    try:
        rsi_value = get_current_rsi(ticker)
        current_time = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"{current_time} - 현재 1분봉 RSI 값: {rsi_value:.2f}")
        time.sleep(1)  # 1초 대기
    except Exception as e:
        print(f"에러 발생: {e}")
        time.sleep(1)  # 에러 발생 시에도 1초 대기 후 재시도
