import time
import numpy as np
import pandas as pd
import pyupbit

# 2. 로그인
# 2.1 텍스트 파일에서 Upbit API 키 읽기
with open("upbit_keys.txt") as f:
    lines = f.readlines()
    access_key = lines[0].strip()
    secret_key = lines[1].strip()

# 2.2 API 인증키 입력
upbit = pyupbit.Upbit(access_key, secret_key)

# 3. 종목 리스트 (상위 시가 총액 5종목)
tickers = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-ADA", "KRW-DOGE"]

# 4. 변수 파트
interval = "minute1"
rsi_period = 14
rsi_threshold = 27  # 초기 매수 시 RSI 27 이하일 때 매수
initial_invest_ratio = 0.01  # 잔고의 1%
target_profit_rate = 0.0045  # 0.45%
maintain_profit_rate = -0.005  # -0.5%
stop_loss_rate = -0.025  # -2.5%
rsi_diff_threshold = 2  # RSI 차이 임계값 설정

# 종목별로 마지막 추가 매수 시의 RSI 값을 저장할 딕셔너리
last_additional_buy_rsi = {ticker: None for ticker in tickers}

# 5. RSI 계산 함수 (캐싱 적용)
rsi_cache = {}

def get_rsi(ticker):
    global rsi_cache
    now = int(time.time() / 60)  # 분 단위로 캐싱
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
                    # 마지막 추가 매수 RSI 값을 현재 RSI로 설정
                    last_additional_buy_rsi[ticker] = rsi
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
                    # 추가 매수 RSI 값 초기화
                    last_additional_buy_rsi[ticker] = None
                    time.sleep(0.1)
                    continue

                # 손절매가 도달 시 매도
                if profit_rate <= stop_loss_rate:
                    order = upbit.sell_market_order(ticker, balance)
                    print(f"{ticker} 매도 주문 완료 - 손절매 실행")
                    # 추가 매수 RSI 값 초기화
                    last_additional_buy_rsi[ticker] = None
                    time.sleep(0.1)
                    continue

                # 수익률 유지 위해 추가 매수 (RSI가 임계값 이상 낮아졌을 때만)
                if profit_rate <= maintain_profit_rate:
                    # 이전 추가 매수 RSI보다 현재 RSI가 rsi_diff_threshold 이상 낮은지 확인
                    if last_additional_buy_rsi[ticker] is None or rsi < last_additional_buy_rsi[ticker] - rsi_diff_threshold:
                        krw_balance = upbit.get_balance("KRW")
                        invest_amount = krw_balance * initial_invest_ratio
                        if invest_amount > 5000:
                            order = upbit.buy_market_order(ticker, invest_amount)
                            print(f"{ticker} 추가 매수 완료 - 금액: {invest_amount} KRW")
                            # 마지막 추가 매수 RSI 값 업데이트
                            last_additional_buy_rsi[ticker] = rsi
                        else:
                            print(f"{ticker} 추가 매수 실패 - 잔액 부족")
                        time.sleep(0.1)
                    else:
                        print(f"{ticker} 추가 매수 조건 미충족 - RSI가 이전보다 {rsi_diff_threshold} 이상 낮지 않음")
                else:
                    print(f"{ticker} 추가 매수 조건 미충족 - 수익률이 유지 수익률보다 높음")

            time.sleep(0.1)  # API 요청 간 간격 조정

        time.sleep(1)  # 루프 간 간격 조정
    except Exception as e:
        print(f"에러 발생: {e}")
        time.sleep(1)
