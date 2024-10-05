import time
import numpy as np
import pandas as pd
import pyupbit


key_file_path = r'C:\Users\winne\OneDrive\바탕 화면\upbit_key.txt'
# 2. 로그인
# 2.1 텍스트 파일에서 Upbit API 키 읽기
with open(key_file_path, 'r') as file:
    access = file.readline().strip()
    secret = file.readline().strip()

# 2.2 API 인증키 입력
upbit = pyupbit.Upbit(access, secret)

# 3. 종목 리스트 (상위 시가 총액 5종목)
tickers = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-ADA", "KRW-DOGE"]

# 4. 변수 파트
interval = "minute1"
rsi_period = 20
rsi_threshold = 24  # RSI 27 이하일 때 매수
initial_invest_ratio = 0.01  # 잔고의 1%
target_profit_rate = 0.0045  # 0.45%
maintain_profit_rate = -0.005  # -0.5%
stop_loss_rate = -0.025  # -2.5%

# 5. RSI 계산 함수 (캐싱 적용)
rsi_cache = {}

def get_rsi(ticker):
    global rsi_cache
    now = int(time.time() / 3)  # 분 단위로 캐싱
    cache_key = f"{ticker}_{now}"
    if cache_key in rsi_cache:
        return rsi_cache[cache_key]
    else:
        df = pyupbit.get_ohlcv(ticker, interval=interval, count=rsi_period + 1)
        delta = df['close'].diff()

        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)

        AU = up.ewm(com=(rsi_period - 1), min_periods=rsi_period).mean()
        AD = down.ewm(com=(rsi_period - 1), min_periods=rsi_period).mean()
        RS = AU / AD

        RSI = 100 - (100 / (1 + RS))
        rsi_value = RSI.iloc[-1]
        rsi_cache[cache_key] = rsi_value
        return rsi_value

# 6. 투자 전략 실행
while True:
    try:
        krw_balance = upbit.get_balance("KRW")
        for ticker in tickers:
            # 현재 보유 수량 및 평균 매수가
            balance = upbit.get_balance(ticker)
            avg_buy_price = upbit.get_avg_buy_price(ticker)

            # 현재가 및 RSI 조회 (API 요청 수 최소화)
            current_price = pyupbit.get_current_price(ticker)
            rsi = get_rsi(ticker)
            print(f"{ticker} RSI: {rsi:.2f}")

            # 6.1 초기 매수
            if balance == 0 and rsi < rsi_threshold:
                invest_amount = krw_balance * initial_invest_ratio
                if invest_amount > 5000:  # 최소 주문 금액 체크
                    order = upbit.buy_market_order(ticker, invest_amount)
                    print(f"{ticker} 매수 주문 완료 - 금액: {invest_amount} KRW")
                else:
                    print(f"{ticker} 매수 실패 - 잔액 부족")
                time.sleep(0.1)

            # 6.2 추가 매수 및 수익률 관리
            elif balance > 0:
                profit_rate = (current_price - avg_buy_price) / avg_buy_price
                print(f"{ticker} 수익률: {profit_rate*100:.2f}%")

                # 목표 수익률 도달 시 매도
                if profit_rate >= target_profit_rate:
                    order = upbit.sell_market_order(ticker, balance)
                    print(f"{ticker} 매도 주문 완료 - 수익 실현")
                    time.sleep(0.1)
                    continue

                # 손절매가 도달 시 매도
                if profit_rate <= stop_loss_rate:
                    order = upbit.sell_market_order(ticker, balance)
                    print(f"{ticker} 매도 주문 완료 - 손절매 실행")
                    time.sleep(0.1)
                    continue

                # 수익률 유지 위해 추가 매수
                if profit_rate <= maintain_profit_rate:
                    krw_balance = upbit.get_balance("KRW")
                    invest_amount = krw_balance * initial_invest_ratio
                    if invest_amount > 5000:
                        order = upbit.buy_market_order(ticker, invest_amount)
                        print(f"{ticker} 추가 매수 완료 - 금액: {invest_amount} KRW")
                    else:
                        print(f"{ticker} 추가 매수 실패 - 잔액 부족")
                    time.sleep(0.1)

            time.sleep(1)  # API 요청 간 간격 조정

        time.sleep(1)  # 루프 간 간격 조정
    except Exception as e:
        print(f"에러 발생: {e}")
        time.sleep(1)