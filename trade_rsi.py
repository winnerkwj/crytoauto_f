import time
import pyupbit
import threading
import os
import pickle

# 최소 거래 금액 설정
MIN_TRADE_AMOUNT = 500  # 최소 거래 금액 500 KRW

# API 키 설정 파일 경로 (사용자 환경에 맞게 변경하세요)
api_key_file_path = r'C:\Users\winne\OneDrive\바탕 화면\upbit_key.txt'

# 거래 상태 저장 파일 경로
trade_state_file_path = 'trade_state.pkl'  # 원하는 경로로 변경 가능

# API 키 읽기
with open(api_key_file_path, 'r') as file:
    access_key = file.readline().strip()
    secret_key = file.readline().strip()

# 업비트 로그인
upbit = pyupbit.Upbit(access_key, secret_key)
print("자동 거래 시작")

# 상위 거래량 30종목 가져오기
def get_top_volume_tickers(n=30):
    tickers = pyupbit.get_tickers(fiat="KRW")
    ticker_volumes = {}
    for ticker in tickers:
        time.sleep(0.1)  # API 호출 제한을 피하기 위한 딜레이
        ohlcv = pyupbit.get_ohlcv(ticker, interval="day", count=1)
        if ohlcv is not None and not ohlcv.empty:
            volume = ohlcv['volume'].iloc[-1]
            ticker_volumes[ticker] = volume
    sorted_tickers = sorted(ticker_volumes.items(), key=lambda x: x[1], reverse=True)
    top_tickers = [ticker for ticker, volume in sorted_tickers[:n]]
    return top_tickers

# 거래할 암호화폐 목록 설정
crypto_tickers = get_top_volume_tickers(30)

# 사용자 설정 변수들
interval = "minute1"         # RSI 계산에 사용할 캔들 간격 (1분봉)
rsi_period = 14              # RSI 계산 기간
rsi_threshold = 27           # RSI가 27 이하일 때 매수
initial_buy_ratio = 0.01     # 초기 매수 시 사용할 KRW 잔고 비율 (1%)

profit_target = 0.45         # 수익률이 0.45% 이상일 때 매도
stop_loss_limit = -2.5       # 손실률이 -2.5% 이하일 때 손절 (7차 매수 이후)

# 종목별 상태 관리 딕셔너리 초기화
trade_state = {}
rsi_values = {}  # 종목별 RSI 값을 저장하는 딕셔너리

# 공유 자원에 대한 락 생성
trade_state_lock = threading.RLock()

# 거래 상태를 파일에 저장하는 함수
def save_trade_state():
    try:
        with open(trade_state_file_path, 'wb') as f:
            pickle.dump(trade_state, f)
    except Exception as e:
        print(f"거래 상태 저장 중 오류 발생: {e}")

# 거래 상태를 파일에서 불러오는 함수
def load_trade_state():
    global trade_state
    if os.path.exists(trade_state_file_path):
        try:
            with open(trade_state_file_path, 'rb') as f:
                trade_state = pickle.load(f)
            print("거래 상태를 파일에서 불러왔습니다.")
        except Exception as e:
            print(f"거래 상태 불러오기 중 오류 발생: {e}")
            trade_state = {}
    else:
        trade_state = {}

# 현재 KRW 잔고 조회 함수 (출금 가능 금액)
def get_krw_balance():
    try:
        balances = upbit.get_balances()
        for b in balances:
            if b['currency'] == 'KRW':
                if b['balance'] is not None and b['locked'] is not None:
                    # 사용 가능한 KRW 잔고 계산
                    available_krw = float(b['balance']) - float(b['locked'])
                    return available_krw
        return 0.0
    except Exception as e:
        print(f"KRW 잔고 조회 중 오류 발생: {e}")
        return 0.0

# 현재 암호화폐 보유 조회 함수
def get_crypto_balance(ticker):
    currency = ticker.split('-')[1]
    try:
        balances = upbit.get_balances()
        for b in balances:
            if b['currency'] == currency:
                if b['balance'] is not None:
                    return float(b['balance'])
        return 0.0
    except Exception as e:
        print(f"{ticker} 잔고 조회 중 오류 발생: {e}")
        return 0.0

# 1분봉 가져오는 함수
def get_ohlcv(ticker):
    try:
        df = pyupbit.get_ohlcv(ticker, interval=interval, count=rsi_period+1)
        return df
    except Exception as e:
        print(f"{ticker}의 OHLCV 데이터 가져오기 오류 발생: {e}")
        return None

# RSI 계산 함수
def calculate_rsi(df):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=rsi_period, min_periods=1).mean()
    avg_loss = loss.rolling(window=rsi_period, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

# 실시간 1분봉 RSI 조회 함수
def get_current_rsi(ticker):
    df = get_ohlcv(ticker)
    if df is not None and not df.empty:
        rsi = calculate_rsi(df)
        return rsi
    else:
        return None

# 매수 함수
def buy_crypto_currency(ticker, amount):
    try:
        print(f"{ticker} 매수 주문 제출: 금액 {amount} KRW")
        return upbit.buy_market_order(ticker, amount)
    except Exception as e:
        print(f"{ticker} 매수 오류 발생: {e}")
        return None

# 매도 함수
def sell_crypto_currency(ticker, amount):
    try:
        print(f"{ticker} 매도 주문 제출: 수량 {amount}")
        return upbit.sell_market_order(ticker, amount)
    except Exception as e:
        print(f"{ticker} 매도 오류 발생: {e}")
        return None

# 수익률 계산 함수
def calculate_profit_rate(ticker, current_price):
    with trade_state_lock:
        total_cost = trade_state[ticker]['total_cost']
        total_amount = trade_state[ticker]['total_amount']
    avg_buy_price = total_cost / total_amount
    profit_rate = ((current_price - avg_buy_price) / avg_buy_price) * 100
    return profit_rate

# 매수 로직
def buy_logic(ticker):
    rsi = get_current_rsi(ticker)
    if rsi is None:
        return
    rsi_values[ticker] = rsi  # RSI 값 저장
    print(f"{ticker} 현재 RSI: {rsi:.2f}")

    if rsi <= rsi_threshold:
        krw_balance = get_krw_balance()
        amount_to_buy = krw_balance * initial_buy_ratio
        if amount_to_buy < MIN_TRADE_AMOUNT:
            print(f"{ticker} 매수 불가: 매수 금액이 최소 거래 금액보다 적습니다.")
            return
        buy_order = buy_crypto_currency(ticker, amount_to_buy)
        if buy_order is not None:
            current_price = pyupbit.get_current_price(ticker)
            with trade_state_lock:
                trade_state[ticker] = {
                    'buy_count': 1,
                    'total_amount': amount_to_buy / current_price,
                    'total_cost': amount_to_buy,
                }
            print(f"{ticker} 초기 매수 완료: 매수 금액 {amount_to_buy} KRW, 가격 {current_price} KRW")
            save_trade_state()

# 추가 매수 로직
def additional_buy_logic(ticker):
    with trade_state_lock:
        buy_count = trade_state[ticker]['buy_count']
    current_price = pyupbit.get_current_price(ticker)
    profit_rate = calculate_profit_rate(ticker, current_price)
    thresholds = {
        1: -1.0,
        2: -1.2,
        3: -1.3,
        4: -1.4,
        5: -1.5,
        6: -1.7,
        7: -2.0,
    }
    if buy_count >= 7:
        return  # 최대 7차 매수까지 진행
    next_buy_count = buy_count + 1
    threshold = thresholds.get(buy_count, None)
    if threshold is not None and profit_rate <= threshold:
        crypto_balance = get_crypto_balance(ticker)
        current_amount = crypto_balance  # 현재 보유 수량
        amount_to_buy = current_amount * current_price  # 현재 보유 금액과 동일한 금액
        krw_balance = get_krw_balance()
        if krw_balance < amount_to_buy or amount_to_buy < MIN_TRADE_AMOUNT:
            print(f"{ticker} 추가 매수 불가: 잔고 부족 또는 최소 거래 금액 미만")
            return
        buy_order = buy_crypto_currency(ticker, amount_to_buy)
        if buy_order is not None:
            with trade_state_lock:
                trade_state[ticker]['buy_count'] = next_buy_count
                trade_state[ticker]['total_amount'] += amount_to_buy / current_price
                trade_state[ticker]['total_cost'] += amount_to_buy
            print(f"{ticker} {next_buy_count}차 추가 매수 완료: 매수 금액 {amount_to_buy} KRW")
            save_trade_state()

# 매도 로직
def sell_logic(ticker):
    current_price = pyupbit.get_current_price(ticker)
    profit_rate = calculate_profit_rate(ticker, current_price)
    crypto_balance = get_crypto_balance(ticker)
    if profit_rate >= profit_target:
        sell_order = sell_crypto_currency(ticker, crypto_balance)
        if sell_order is not None:
            with trade_state_lock:
                del trade_state[ticker]
            print(f"{ticker} 목표 수익 달성 매도 완료: 수익률 {profit_rate:.2f}%")
            save_trade_state()
    else:
        with trade_state_lock:
            buy_count = trade_state[ticker]['buy_count']
        if buy_count >= 7 and profit_rate <= stop_loss_limit:
            krw_balance = get_krw_balance()
            if krw_balance < MIN_TRADE_AMOUNT:
                sell_order = sell_crypto_currency(ticker, crypto_balance)
                if sell_order is not None:
                    with trade_state_lock:
                        del trade_state[ticker]
                    print(f"{ticker} 손절 매도 완료: 수익률 {profit_rate:.2f}%")
                    save_trade_state()

# 메인 매매 로직
def trade(ticker):
    crypto_balance = get_crypto_balance(ticker)
    if ticker not in trade_state and crypto_balance == 0:
        # 매수된 암호화폐가 없을 때
        buy_logic(ticker)
    else:
        if ticker in trade_state and crypto_balance == 0:
            # 외부에서 암호화폐를 매도한 것으로 간주하고 trade_state 초기화
            with trade_state_lock:
                del trade_state[ticker]
            print(f"{ticker}: 외부에서 암호화폐를 매도하여 거래 상태를 초기화합니다.")
            save_trade_state()
            return
        elif ticker not in trade_state and crypto_balance > 0:
            # 외부에서 암호화폐를 매수한 것으로 간주하고 trade_state 업데이트
            current_price = pyupbit.get_current_price(ticker)
            total_amount = crypto_balance
            total_cost = current_price * total_amount
            with trade_state_lock:
                trade_state[ticker] = {
                    'buy_count': 1,
                    'total_amount': total_amount,
                    'total_cost': total_cost,
                }
            print(f"{ticker}: 외부에서 암호화폐를 매수하여 거래 상태를 업데이트합니다.")
            save_trade_state()
        # 매도 로직 실행
        sell_logic(ticker)
        # 추가 매수 로직 실행
        additional_buy_logic(ticker)

# 거래 대상이 아닌 종목 보유 여부 확인 함수
def check_non_target_balances():
    balances = upbit.get_balances()
    non_target_tickers = []
    for b in balances:
        currency = b['currency']
        if currency != 'KRW' and float(b['balance']) > 0:
            ticker = 'KRW-' + currency
            if ticker not in crypto_tickers:
                non_target_tickers.append(ticker)
    if non_target_tickers:
        print(f"거래 대상이 아닌 종목을 보유하고 있습니다: {non_target_tickers}")

# 티커별로 매매 로직을 실행하는 함수 (스레드에서 실행될 함수)
def trade_crypto_thread(ticker):
    while True:
        try:
            trade(ticker)
        except Exception as e:
            print(f"{ticker} 거래 중 오류 발생: {e}")
        time.sleep(1)  # 1초마다 실행

# 메인 코드 실행
if __name__ == "__main__":
    # 거래 상태 로드 또는 초기화
    load_trade_state()

    # 거래 대상이 아닌 종목 보유 여부 확인
    check_non_target_balances()

    # 각 티커에 대해 스레드를 생성하여 매매 로직 실행
    threads = []
    for ticker in crypto_tickers:
        thread = threading.Thread(target=trade_crypto_thread, args=(ticker,))
        thread.daemon = True  # 메인 스레드 종료 시 함께 종료되도록 설정
        thread.start()
        threads.append(thread)
        time.sleep(0.1)  # 스레드 생성 간 딜레이 추가

    # 메인 스레드는 계속 실행되도록 유지
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("프로그램이 종료되었습니다.")
