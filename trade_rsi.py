import time
import numpy as np
import pyupbit

# 최소 거래 금액 설정
MIN_TRADE_AMOUNT = 500  # 최소 거래 금액 500 KRW

# API 키 설정 파일 경로
key_file_path = r'C:\Users\winne\OneDrive\바탕 화면\upbit_key.txt'

# API 키 읽기
with open(key_file_path, 'r') as file:
    access = file.readline().strip()
    secret = file.readline().strip()

# 거래할 암호화폐 종목 설정
tickers = [
    "KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-EOS", "KRW-ADA", 
    "KRW-DOGE", "KRW-LOOM", "KRW-SHIB", "KRW-NEO", 
    "KRW-ARDR", "KRW-GAS", "KRW-HBAR", "KRW-STPT", "KRW-SEI",
    "KRW-ZRO", "KRW-HIVE", "KRW-SOL", "KRW-HIFI", "KRW-TFUEL", 
    "KRW-WAVES"
]

# 종목별 거래 상태 관리 딕셔너리 초기화
trade_state = {
    ticker: {
        "buy1_price": None,  # 첫 매수 가격
        "total_amount": 0,  # 총 매수량
        "total_cost": 0,  # 총 매수 금액 (가중 평균 계산용)
        "buy_executed_count": 0,  # 매수 실행 횟수
        "sell_executed": False,  # 매도 완료 여부
        "sell_order_id": None  # 매도 주문 ID 저장
    } for ticker in tickers
}

# 사용자 설정 변수들
interval = "minute1"
rsi_period = 14
rsi_threshold = 30  # RSI 30 이하일 때 매수
initial_buy_percent = 0.01

profit_threshold = 0.3  # 수익률이 0.3% 이상일 때 매도
loss_threshold_after_final_buy = -2.5  # 손실률이 -2% 이하일 때 매도

# 단계별 추가 매수 조건 설정
additional_buy_conditions = [
    {"trigger_loss": -1, "buy_ratio": 0.01},    # 손실률 -1%에서 1% 추가 매수
    {"trigger_loss": -1.2, "buy_ratio": 0.015},  # 손실률 -1.2%에서 1.5% 추가 매수
    {"trigger_loss": -1.3, "buy_ratio": 0.02},   # 손실률 -1.3%에서 2% 추가 매수
    {"trigger_loss": -1.4, "buy_ratio": 0.025},  # 손실률 -1.4%에서 2.5% 추가 매수
    {"trigger_loss": -1.5, "buy_ratio": 0.03},   # 손실률 -1.5%에서 3% 추가 매수
    {"trigger_loss": -1.7, "buy_ratio": 0.035},  # 손실률 -1.7%에서 3.5% 추가 매수
    {"trigger_loss": -2, "buy_ratio": 0.04}      # 손실률 -2%에서 4% 추가 매수
]

# 잔고 캐싱을 위한 전역 변수
balance_cache = {}
cache_timestamp = 0
cache_duration = 60  # 60초 동안 캐시 유지

# 잔고 조회 함수 (캐싱 적용)
def get_cached_balance(currency):
    global balance_cache, cache_timestamp
    
    current_time = time.time()
    if current_time - cache_timestamp > cache_duration:
        print("잔고 캐시 만료, 새로 조회 중...")
        balance_cache = upbit.get_balances()
        cache_timestamp = current_time

    for b in balance_cache:
        if b['currency'] == currency:
            if b['balance'] is not None:
                return float(b['balance'])
    return 0

def get_balance(currency):
    return get_cached_balance(currency)

# 현재 가격 조회 함수
def safe_get_current_price(ticker):
    time.sleep(1/30)
    try:
        price = pyupbit.get_current_price(ticker)
        if price is None:
            print(f"{ticker}의 현재 가격을 가져올 수 없습니다.")
            return None
        return price
    except Exception as e:
        print(f"{ticker}의 현재 가격 조회 중 오류 발생: {e}")
        return None

# 암호화폐 지정가 매수 함수
def buy_crypto(ticker, amount):
    if amount < MIN_TRADE_AMOUNT:
        print(f"{ticker} 매수 금액 {amount} KRW는 최소 거래 금액 {MIN_TRADE_AMOUNT} KRW 이하입니다.")
        return None
    time.sleep(1/8)
    try:
        return upbit.buy_market_order(ticker, amount)
    except Exception as e:
        print(f"{ticker} 매수 오류 발생: {e}")
        return None

# 암호화폐 지정가 매도 함수
def sell_crypto(ticker, price, amount):
    total_amount = price * amount
    if total_amount < MIN_TRADE_AMOUNT:
        print(f"{ticker} 매도 금액 {total_amount} KRW는 최소 거래 금액 {MIN_TRADE_AMOUNT} KRW 이하입니다.")
        return None
    time.sleep(1/8)
    try:
        return upbit.sell_limit_order(ticker, price, amount)
    except Exception as e:
        print(f"{ticker} 매도 오류 발생: {e}")
        return None

# 매도 후 상태 초기화 함수
def reset_trade_state(ticker):
    trade_state[ticker]['buy1_price'] = None
    trade_state[ticker]['total_amount'] = 0
    trade_state[ticker]['total_cost'] = 0
    trade_state[ticker]['buy_executed_count'] = 0
    trade_state[ticker]['sell_executed'] = False
    trade_state[ticker]['sell_order_id'] = None
    print(f"{ticker}의 거래 상태가 초기화되었습니다.")

# 평균 매수 가격 계산 함수
def calculate_avg_buy_price(ticker):
    total_cost = trade_state[ticker]['total_cost']
    total_amount = trade_state[ticker]['total_amount']
    if total_amount == 0:
        return None
    return total_cost / total_amount

# 매수 후 상태 업데이트 함수
def update_buy_state(ticker, buy_price, buy_amount):
    total_cost = trade_state[ticker]['total_cost']
    total_amount = trade_state[ticker]['total_amount']

    new_cost = buy_price * buy_amount
    trade_state[ticker]['total_cost'] = total_cost + new_cost
    trade_state[ticker]['total_amount'] += buy_amount / buy_price

    avg_price = calculate_avg_buy_price(ticker)
    print(f"{ticker}의 새로운 가중 평균 매수가: {avg_price:.2f} KRW")

# 매수 실행 로직
def execute_buy(ticker, buy_amount):
    current_price = safe_get_current_price(ticker)
    if current_price is None:
        return False

    buy_order = buy_crypto(ticker, buy_amount)
    if buy_order:
        trade_state[ticker]['total_cost'] += buy_amount
        trade_state[ticker]['total_amount'] += buy_amount / current_price
        trade_state[ticker]['buy_executed_count'] += 1
        print(f"{ticker} 매수 완료: {buy_amount} KRW 매수, 가격: {current_price} KRW")
        return True
    return False

# 추가 매수 실행 로직 (손실률에 따른 추가 매수)
def execute_additional_buy(ticker, current_price, profit_rate):
    buy_executed_count = trade_state[ticker]['buy_executed_count']
    
    if buy_executed_count == 0:
        return  # 초기 매수가 되지 않았다면 추가 매수하지 않음

    for i, condition in enumerate(additional_buy_conditions):
        if profit_rate <= condition['trigger_loss'] and buy_executed_count == i + 1:
            buy_amount = krw_balance * condition['buy_ratio']
            print(f"{ticker}: 손실률 {profit_rate:.2f}%, 추가 매수 {i + 2}차 진행 - {buy_amount} KRW")
            execute_buy(ticker, buy_amount)
            break

# 매도 실행 로직
def execute_sell(ticker):
    current_price = safe_get_current_price(ticker)
    if current_price is None:
        return

    crypto_balance = get_balance(ticker.split("-")[1])
    if crypto_balance > 0:
        sell_order = sell_crypto(ticker, current_price, crypto_balance)
        if sell_order:
            trade_state[ticker]['sell_order_id'] = sell_order['uuid']
            print(f"{ticker} 매도 주문 완료: {current_price} KRW")
            reset_trade_state(ticker)  # 매도 후 상태 초기화
            return True  # 매도 완료 후 초기화 완료 상태 반환
    return False

# 매도 조건 확인 함수
def should_sell(ticker, current_price):
    avg_buy_price = calculate_avg_buy_price(ticker)
    if avg_buy_price is None:
        return False, None
    
    profit_rate = ((current_price - avg_buy_price) / avg_buy_price) * 100

    if profit_rate >= profit_threshold:
        return True, "profit"

    if trade_state[ticker]['buy_executed_count'] >= 7 and profit_rate <= loss_threshold_after_final_buy:
        return True, "loss"
    
    return False, None

# RSI 계산 함수
def calculate_rsi(ticker, period=rsi_period, interval=interval):
    df = pyupbit.get_ohlcv(ticker, interval=interval, count=period + 1)
    if df is None or df.empty:
        print(f"{ticker}의 {interval} 데이터를 가져올 수 없습니다.")
        return None

    delta = df['close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = np.mean(gain[-period:])
    avg_loss = np.mean(loss[-period:])

    if avg_loss == 0:
        return 100

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

# RSI 매수 신호 감지 함수 (첫 매수에서만 사용)
def detect_rsi_buy_signal(ticker, rsi_threshold=rsi_threshold, interval=interval):
    rsi = calculate_rsi(ticker, interval=interval)
    if rsi is None:
        return False

    if rsi <= rsi_threshold:
        print(f"{ticker}: RSI 매수 신호 감지 - RSI 값: {rsi:.2f}")
        return True
    else:
        print(f"{ticker}: RSI 매수 신호 미충족 - RSI 값: {rsi:.2f}")
        return False

# 초기 매수 실행 함수
def execute_buy_on_rsi_signal(ticker):
    if trade_state[ticker]['buy_executed_count'] > 0:
        print(f"{ticker} 이미 첫 매수가 완료된 종목입니다.")
        return False

    if detect_rsi_buy_signal(ticker):
        buy_amount = krw_balance * initial_buy_percent
        print(f"{ticker} RSI 매수 신호 감지: {buy_amount} KRW 매수 진행")
        buy_order = buy_crypto(ticker, buy_amount)
        if buy_order is not None:
            current_price = safe_get_current_price(ticker)
            update_buy_state(ticker, current_price, buy_amount)
            print(f"{ticker} RSI 매수 완료: 가격: {current_price} KRW")
        time.sleep(3)
        return True
    return False

# 단일 티커에 대한 매수/매도 로직 실행 함수
def trade_single_ticker(ticker):
    current_price = safe_get_current_price(ticker)
    if current_price is None:
        return

    avg_buy_price = calculate_avg_buy_price(ticker)
    if avg_buy_price is None:
        return

    profit_rate = ((current_price - avg_buy_price) / avg_buy_price) * 100

    if trade_state[ticker]['buy_executed_count'] == 0:
        execute_buy_on_rsi_signal(ticker)
    else:
        execute_additional_buy(ticker, current_price, profit_rate)

    should_sell_now, reason = should_sell(ticker, current_price)
    if should_sell_now:
        if execute_sell(ticker):  # 매도 완료 시 초기화된 상태에서 다시 매수 진행
            print(f"{ticker} 매도 완료 후 다시 매수 신호 감지 대기 중...")
            return  # 매도 후 초기화되었으므로 바로 매수로 돌아가 다시 반복

# 업비트 로그인
upbit = pyupbit.Upbit(access, secret)
print("자동 거래 시작")

# Step 1: 초기 구매 결정
krw_balance = get_balance("KRW")

# Step 2: 각 티커에 대해 1분마다 거래 실행
while True:
    start_time = time.time()
    
    for ticker in tickers:
        print(f"{ticker}의 매수 상태 확인 중...")
        trade_single_ticker(ticker)  # 매도 및 추가 매수는 손실률 기준으로 실행
        time.sleep(1)

    elapsed_time = time.time() - start_time
    if elapsed_time < 60:
        time.sleep(60 - elapsed_time)
