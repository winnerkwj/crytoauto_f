import time
import numpy as np
import pyupbit
from datetime import datetime, timedelta

# 최소 거래 금액 설정
MIN_TRADE_AMOUNT = 500  # 최소 거래 금액 500 KRW

# 최소 잔고 기준 설정 (5000 KRW)
MIN_BALANCE_AMOUNT = 5000  # 5000 KRW 이상 잔고가 있는 암호화폐만 추가

# API 키 설정 파일 경로
key_file_path = r'C:\Users\winne\OneDrive\바탕 화면\upbit_key.txt'

# API 키 읽기
with open(key_file_path, 'r') as file:
    access = file.readline().strip()
    secret = file.readline().strip()

# 자동으로 추가할 사용자 지정 종목들 (잔고가 5000원 이상일 때 추가)
user_defined_tickers = []

# 거래할 암호화폐 종목 설정 (초기에는 비워둠)
tickers = []

# 종목별 거래 상태 관리 딕셔너리 초기화 함수
def initialize_trade_state():
    global trade_state
    trade_state = {
        ticker: {
            "buy1_price": None,  # 첫 매수 가격 (가중 평균 매수 가격으로 대체)
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
rsi_threshold = 26  # RSI 30 이하일 때 매수
initial_buy_percent = 0.01

profit_threshold = 0.45  # 수익률이 0.45% 이상일 때 매도
loss_threshold_after_final_buy = -2.5  # 손실률이 -2.5% 이하일 때 매도

# 단계별 추가 매수 조건 설정
additional_buy_conditions = [
    {"trigger_loss": -1},    # 손실률 -1%에서 추가 매수
    {"trigger_loss": -1.2},  # 손실률 -1.2%에서 추가 매수
    {"trigger_loss": -1.3},   # 손실률 -1.3%에서 추가 매수
    {"trigger_loss": -1.4},  # 손실률 -1.4%에서 추가 매수
    {"trigger_loss": -1.5},   # 손실률 -1.5%에서 추가 매수
    {"trigger_loss": -1.7},  # 손실률 -1.7%에서 추가 매수
    {"trigger_loss": -2}      # 손실률 -2%에서 추가 매수
]

# 잔고 캐싱을 위한 전역 변수
balance_cache = {}
cache_timestamp = 0
cache_duration = 60  # 60초 동안 캐시 유지

# KRW 잔고 이전 값 (입출금 감지용)
krw_last_balance = 0

# 잔고 조회 함수 (KRW는 실시간으로, 다른 암호화폐는 캐싱 적용)
def get_cached_balance(currency, retry_count=3):
    global balance_cache, cache_timestamp, krw_last_balance
    current_time = time.time()
    attempt = 0

    while attempt < retry_count:
        try:
            if currency == "KRW":
                # KRW 잔고는 항상 실시간으로 조회
                print("KRW 잔고를 실시간으로 조회 중...")
                balances = upbit.get_balances()
                for b in balances:
                    if b['currency'] == "KRW" and b['balance'] is not None:
                        current_krw_balance = float(b['balance'])
                        # 입출금 여부 감지
                        if current_krw_balance != krw_last_balance:
                            print(f"KRW 잔고 변동 감지: 이전 잔고 {krw_last_balance} KRW, 현재 잔고 {current_krw_balance} KRW")
                            krw_last_balance = current_krw_balance  # 새로운 잔고로 업데이트
                        return current_krw_balance
                return 0
            else:
                # 다른 암호화폐는 캐싱 사용 (캐시 시간 만료 시 재조회)
                if current_time - cache_timestamp > cache_duration:
                    print("잔고 캐시 만료, 새로 조회 중...")
                    balance_cache = upbit.get_balances()
                    cache_timestamp = current_time
                    print("잔고 갱신 성공")
                    
                for b in balance_cache:
                    if b['currency'] == currency and b['balance'] is not None:
                        return float(b['balance'])
                return 0
        except Exception as e:
            attempt += 1
            print(f"잔고 조회 실패 (시도 {attempt}/{retry_count}): {e}")
            time.sleep(1)
    print(f"잔고 조회 실패, {retry_count}번 시도 후 중단")
    return 0

# 잔고 기준으로 추가할 사용자 지정 티커를 찾는 함수
def update_user_defined_tickers_with_balance():
    global user_defined_tickers
    user_defined_tickers = []  # 매번 초기화

    try:
        balances = upbit.get_balances()
        for balance in balances:
            currency = balance['currency']
            if currency == 'KRW':  # KRW는 제외
                continue
            if balance['balance'] is not None:
                # 잔고가 5000원 이상일 때만 추가
                balance_amount = float(balance['balance']) * float(balance['avg_buy_price'])
                if balance_amount >= MIN_BALANCE_AMOUNT:
                    ticker = f"KRW-{currency}"
                    user_defined_tickers.append(ticker)

                    # 잔고만큼 초기 매수로 인식
                    initialize_buy_state_from_balance(ticker, float(balance['balance']), float(balance['avg_buy_price']))

        print(f"5000원 이상 잔고 보유 종목: {user_defined_tickers}")
    except Exception as e:
        print(f"잔고 조회 오류: {e}")

# 기존 보유 종목을 초기 매수로 인식하는 함수
def initialize_buy_state_from_balance(ticker, balance_amount, avg_buy_price):
    """
    기존 보유 암호화폐의 잔고와 평균 매수 가격을 초기 매수로 인식하는 함수.
    """
    if ticker not in trade_state:
        print(f"{ticker}의 거래 상태가 초기화되지 않았습니다. 상태를 먼저 초기화하세요.")
        return

    # 기존 잔고를 초기 매수로 인식
    trade_state[ticker]['buy1_price'] = avg_buy_price
    trade_state[ticker]['total_amount'] = balance_amount
    trade_state[ticker]['total_cost'] = balance_amount * avg_buy_price
    trade_state[ticker]['buy_executed_count'] = 1  # 최소 1번의 매수가 완료된 것으로 간주

    print(f"{ticker}: 초기 매수 상태 설정 완료 - 평균 매수 가격: {avg_buy_price:.2f} KRW, 잔고: {balance_amount} 개")

# 시가총액 상위 40개 암호화폐 종목 조회 및 등록 함수 (1일 1회 업데이트)
def update_top_40_tickers():
    global tickers
    try:
        # 상위 40개 티커 가져오기
        top_40_tickers = pyupbit.get_tickers(fiat="KRW")[:40]

        # 사용자 지정 티커를 잔고 기준으로 업데이트
        update_user_defined_tickers_with_balance()

        # 사용자 지정 종목을 중복 없이 추가
        tickers = list(set(top_40_tickers + user_defined_tickers))

        print(f"시가총액 상위 40개 및 사용자 지정 티커 업데이트 완료: {tickers}")
    except Exception as e:
        print(f"티커 업데이트 오류: {e}")

# 현재 가격 조회 함수
def safe_get_current_price(ticker, retry_count=3):
    for attempt in range(retry_count):
        time.sleep(1/30)
        try:
            price = pyupbit.get_current_price(ticker)
            if price is None:
                print(f"{ticker}의 현재 가격을 가져올 수 없습니다.")
                continue
            return price
        except Exception as e:
            print(f"{ticker}의 현재 가격 조회 중 오류 발생: {e}")
            time.sleep(1)
    return None

# 매수 로직 (최소 거래 금액을 확인 및 잔고 부족 처리)
def buy_crypto(ticker, amount):
    if amount < MIN_TRADE_AMOUNT:
        print(f"{ticker} 매수 금액 {amount} KRW는 최소 거래 금액 {MIN_TRADE_AMOUNT} KRW 이하입니다.")
        return None
    
    krw_balance = get_cached_balance("KRW")  # 최신 KRW 잔고 확인
    if krw_balance < amount:
        print(f"KRW 잔고 부족: 요청된 매수 금액 {amount} KRW, 실제 잔고 {krw_balance} KRW")
        return None
    
    time.sleep(1/8)
    try:
        return upbit.buy_market_order(ticker, amount)
    except Exception as e:
        print(f"{ticker} 매수 오류 발생: {e}")
        return None

# 매도 로직 - 시장가 매도
def sell_crypto(ticker, amount):
    total_amount = amount * pyupbit.get_current_price(ticker)
    if total_amount < MIN_TRADE_AMOUNT:
        print(f"{ticker} 매도 금액 {total_amount} KRW는 최소 거래 금액 {MIN_TRADE_AMOUNT} KRW 이하입니다.")
        return None
    time.sleep(1/8)
    try:
        return upbit.sell_market_order(ticker, amount)  # 시장가 매도
    except Exception as e:
        print(f"{ticker} 시장가 매도 오류 발생: {e}")
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

# 가중 평균 매수가 업데이트 함수
def update_weighted_avg_price(ticker, new_buy_price, new_buy_amount):
    # 현재 총 매수량 및 매수 금액
    total_amount = trade_state[ticker]['total_amount']
    total_cost = trade_state[ticker]['total_cost']

    # 새로운 매수 금액을 추가하여 가중 평균 가격을 계산
    total_new_cost = total_cost + (new_buy_price * new_buy_amount)
    total_new_amount = total_amount + new_buy_amount

    # 가중 평균 매수 가격 업데이트
    new_avg_buy_price = total_new_cost / total_new_amount

    # 업데이트된 값 반영
    trade_state[ticker]['total_cost'] = total_new_cost
    trade_state[ticker]['total_amount'] = total_new_amount
    trade_state[ticker]['buy1_price'] = new_avg_buy_price

    print(f"{ticker}: 가중 평균 매수가 업데이트 - 새로운 평균 매수가: {new_avg_buy_price:.2f} KRW, 총 매수량: {total_new_amount} 개")

# 추가 매수 로직 (손실률에 따른 추가 매수 - 잔고의 2배 매수)
def execute_additional_buy(ticker, current_price, profit_rate):
    print(f"{ticker}: 추가 매수 확인 중 - 현재 손익률: {profit_rate:.2f}%")

    # 추가 매수 조건 확인
    for condition in additional_buy_conditions:
        if profit_rate <= condition['trigger_loss']:
            # 현재 암호화폐 잔고의 2배를 추가 매수
            crypto_balance = get_cached_balance(ticker.split("-")[1])
            buy_amount = crypto_balance * current_price * 2  # 잔고 2배에 해당하는 KRW 금액 계산
            
            if buy_amount < MIN_TRADE_AMOUNT:
                print(f"{ticker}: 추가 매수 금액이 최소 거래 금액 {MIN_TRADE_AMOUNT} KRW 미만이므로 매수 중단.")
                return False
            print(f"{ticker}: 추가 매수 진행 중 - 매수 금액: {buy_amount:.2f} KRW")

            # 매수 실행
            buy_order = buy_crypto(ticker, buy_amount)
            if buy_order is not None:
                # 현재 가격으로 가중 평균 매수가 업데이트
                update_weighted_avg_price(ticker, current_price, buy_amount / current_price)
            return True
    print(f"{ticker}: 추가 매수 조건 미충족 - 추가 매수 중단")
    return False

# 단일 티커에 대한 매수/매도 로직 실행 함수
def trade_single_ticker(ticker):
    current_price = safe_get_current_price(ticker)
    if current_price is None:
        print(f"{ticker}: 현재 가격을 가져올 수 없어 매매 중단")
        return

    # 실제 잔고 확인
    crypto_balance = get_cached_balance(ticker.split("-")[1])

    # 내부 상태와 실제 잔고 불일치 감지
    if crypto_balance == 0 and trade_state[ticker]['total_amount'] > 0:
        print(f"{ticker}: 실제 잔고가 0입니다. 외부에서 매도된 것으로 추정됩니다. 상태 초기화")
        reset_trade_state(ticker)
        return

    avg_buy_price = trade_state[ticker]['buy1_price']
    if avg_buy_price is None:
        # 매수 신호 감지 후 첫 매수 시도
        print(f"{ticker}: 첫 매수 시도 중")
        execute_buy_on_rsi_signal(ticker)
        return

    # 손익률 계산 및 로그 출력
    profit_rate = ((current_price - avg_buy_price) / avg_buy_price) * 100
    print(f"{ticker}: 손익률 계산 중 - 현재 가격: {current_price} KRW, 손익률: {profit_rate:.2f}%")

    # 추가 매수 로직
    execute_additional_buy(ticker, current_price, profit_rate)

    # 매도 조건 확인
    should_sell_now, reason = should_sell(ticker, current_price)
    if should_sell_now:
        print(f"{ticker}: 매도 조건 충족 - 이유: {reason}")
        if crypto_balance > 0:
            sell_order = sell_crypto(ticker, crypto_balance)  # 시장가 매도
            if sell_order:
                reset_trade_state(ticker)
                print(f"{ticker}: 매도 완료 - 가격 {current_price} KRW")
    else:
        print(f"{ticker}: 매도 조건 미충족 - 매도 중단")

# 매도 조건 확인 함수 (손익률 및 매도 조건 확인)
def should_sell(ticker, current_price):
    avg_buy_price = trade_state[ticker]['buy1_price']
    if avg_buy_price is None:
        print(f"{ticker}: 평균 매수 가격이 설정되지 않음. 매도 불가.")
        return False, None
    
    # 손익률 계산
    profit_rate = ((current_price - avg_buy_price) / avg_buy_price) * 100
    print(f"{ticker}: 손익률 계산 - 현재 가격: {current_price}, 평균 매수 가격: {avg_buy_price}, 손익률: {profit_rate:.2f}%")

    # 매도 조건 확인
    if profit_rate >= profit_threshold:
        print(f"{ticker}: 수익률 {profit_rate:.2f}%가 {profit_threshold}% 이상이므로 매도 조건 충족.")
        return True, "profit"

    # 추가 매수가 7차례 이상 이루어졌고 손실률이 -2.5% 이하일 때
    if trade_state[ticker]['buy_executed_count'] >= 7 and profit_rate <= loss_threshold_after_final_buy:
        print(f"{ticker}: 손익률 {profit_rate:.2f}%, 손실 한도 {-loss_threshold_after_final_buy}% 이하이므로 손실 매도 조건 충족.")
        return True, "loss"
    
    print(f"{ticker}: 매도 조건 충족 안됨.")
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

# RSI 매수 신호 감지
def detect_rsi_buy_signal(ticker, rsi_threshold=rsi_threshold, interval=interval):
    rsi = calculate_rsi(ticker, interval=interval)
    if rsi is None:
        return False

    print(f"{ticker}: 현재 RSI 값 - {rsi:.2f}")

    if rsi <= rsi_threshold:
        print(f"{ticker}: RSI 매수 신호 감지 - RSI 값: {rsi:.2f}")
        return True
    else:
        print(f"{ticker}: RSI 매수 신호 미충족 - RSI 값: {rsi:.2f}")
        return False

# 초기 매수 실행 함수
def execute_buy_on_rsi_signal(ticker):
    if trade_state[ticker]['buy_executed_count'] > 0:
        print(f"{ticker}: 이미 첫 매수가 완료된 종목입니다.")
        return False

    if detect_rsi_buy_signal(ticker):
        krw_balance = get_cached_balance("KRW")
        buy_amount = krw_balance * initial_buy_percent
        if buy_amount < MIN_TRADE_AMOUNT:
            print(f"{ticker}: 잔고 부족으로 매수 불가 - 매수 금액 {buy_amount} KRW, 최소 거래 금액 {MIN_TRADE_AMOUNT} KRW")
            return False
        
        print(f"{ticker}: RSI 매수 신호 감지 - 매수 금액 {buy_amount} KRW")
        buy_order = buy_crypto(ticker, buy_amount)
        if buy_order is not None:
            current_price = safe_get_current_price(ticker)
            if current_price:
                trade_state[ticker]['buy1_price'] = current_price
                trade_state[ticker]['buy_executed_count'] += 1
                print(f"{ticker}: 매수 완료 - 가격 {current_price} KRW, 매수 횟수: {trade_state[ticker]['buy_executed_count']}")
        time.sleep(3)
        return True
    else:
        print(f"{ticker}: RSI 조건을 충족하지 못해 매수 시도 안 함")
    return False

# 업비트 로그인
upbit = pyupbit.Upbit(access, secret)
print("자동 거래 시작")

# Step 1: 시가총액 상위 40개 티커 업데이트
update_top_40_tickers()
initialize_trade_state()

# Step 2: 각 티커에 대해 1분마다 거래 실행
last_ticker_update = datetime.now()

while True:
    start_time = time.time()
    
    # 매일 한 번 상위 40개 티커 업데이트
    if datetime.now() - last_ticker_update > timedelta(days=1):
        update_top_40_tickers()
        initialize_trade_state()  # 티커 변경에 따라 상태 초기화
        last_ticker_update = datetime.now()

    for ticker in tickers:
        print(f"{ticker}의 매수 상태 확인 중...")
        trade_single_ticker(ticker)  # 매도 및 추가 매수는 손실률 기준으로 실행
        time.sleep(1)

    elapsed_time = time.time() - start_time
    if elapsed_time < 60:
        time.sleep(60 - elapsed_time)
