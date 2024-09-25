import time
import pyupbit

# API 키 설정 (실제 API 키를 여기에 넣어야 합니다)
key_file_path = r'C:\Users\winne\OneDrive\바탕 화면\upbit_key.txt'  # <사용자 이름>을 실제 사용자 이름으로 변경하세요

# API 키 읽기
with open(key_file_path, 'r') as file:
    access = file.readline().strip()  # 첫 번째 줄에서 Access Key 읽기
    secret = file.readline().strip()  # 두 번째 줄에서 Secret Key 읽기

# 거래할 암호화폐 종목 설정 (여러 종목)
tickers = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-EOS", "KRW-ADA", "KRW-DOGE"]

# 각 종목별로 거래 상태를 저장할 딕셔너리 초기화
trade_state = {ticker: {"buy1_price": None, "buy2_executed": False, "buy3_executed": False,
                        "buy4_executed": False, "buy5_executed": False, "buy6_executed": False,
                        "buy7_executed": False, "sell_executed": False} for ticker in tickers}

# 사용자가 관리할 변수 설정

# 분봉 데이터 간격 설정
interval = "minute3"  # "minute3"는 3분봉 데이터를 의미합니다. "minute1", "minute5" 등으로 변경 가능

# 매수 조건 설정
k_value = 0.7  # 변동성 돌파 전략에서 사용할 k 값
initial_buy_percent = 0.01  # 초기 매수 시 원화 잔고의 몇 퍼센트를 사용할지 설정

# 매도 조건 설정
profit_threshold = 0.1  # 매도 조건 0.1: 수익률이 이 값 이상일 때 매도
loss_threshold_after_final_buy = -3  # 매도 조건 2: 최종 매수 단계 이후 손실률이 이 값 이하일 때 매도

# 추가 매수 조건 설정 (각 단계별 관리)
additional_buy_conditions = [
    {"trigger_loss": -1, "buy_ratio": 0.01},  # 2단계: 손실률이 -1% 이하일 때 1% 매수
    {"trigger_loss": -2, "buy_ratio": 0.015},  # 3단계: 손실률이 -2% 이하일 때 1.5% 매수
    {"trigger_loss": -3, "buy_ratio": 0.02},  # 4단계: 손실률이 -3% 이하일 때 2% 매수
    {"trigger_loss": -4, "buy_ratio": 0.025},  # 5단계: 손실률이 -4% 이하일 때 2.5% 매수
    {"trigger_loss": -5, "buy_ratio": 0.03},  # 6단계: 손실률이 -5% 이하일 때 3% 매수
    {"trigger_loss": -6, "buy_ratio": 0.035},  # 7단계: 손실률이 -6% 이하일 때 3.5% 매수
    {"trigger_loss": -7, "buy_ratio": 0.04},  # 8단계: 손실률이 -7% 이하일 때 4% 매수
]

def get_balance(currency):
    """잔고 조회 함수 (원화 또는 암호화폐 잔고 조회)"""
    balances = upbit.get_balances()  # 계좌 잔고 정보를 가져옵니다
    for b in balances:  # 잔고 정보에서 각각의 통화를 확인
        if b['currency'] == currency:  # 요청한 통화와 일치하는 잔고를 찾습니다
            if b['balance'] is not None:
                return float(b['balance'])  # 잔고가 있으면 잔고를 반환
            else:
                return 0  # 잔고가 없으면 0을 반환
    return 0  # 해당 통화에 대한 정보가 없을 경우 0 반환

def buy_crypto(ticker, amount):
    """암호화폐 구매 함수 (시장가 주문)"""
    try:
        return upbit.buy_market_order(ticker, amount)  # 지정한 암호화폐를 시장가로 구매합니다
    except Exception as e:
        print(f"{ticker} 주문 중 오류 발생: {e}")
        return None

def sell_crypto(ticker, amount):
    """암호화폐 매도 함수 (시장가 주문)"""
    try:
        return upbit.sell_market_order(ticker, amount)  # 지정한 암호화폐를 시장가로 판매합니다
    except Exception as e:
        print(f"{ticker} 매도 중 오류 발생: {e}")
        return None

def safe_get_current_price(ticker):
    """현재 암호화폐의 시장 가격을 안전하게 조회하는 함수"""
    try:
        price = pyupbit.get_current_price(ticker)  # 현재 암호화폐 가격을 가져옵니다
        if price is None:
            print(f"{ticker}의 현재 가격을 가져올 수 없습니다.")
            return None
        return price
    except Exception as e:
        print(f"{ticker}의 현재 가격을 가져오는 중 오류 발생: {e}")
        return None

def get_volatility_breakout_price(ticker, k=k_value):
    """변동성 돌파 전략을 위한 분봉 매수 가격 계산 함수"""
    try:
        # 설정한 interval(분봉 간격) 데이터를 가져옵니다.
        df = pyupbit.get_ohlcv(ticker, interval=interval, count=2)
        if df is None or df.empty:
            print(f"{ticker}의 {interval} 데이터를 가져올 수 없습니다.")
            return None

        previous_candle = df.iloc[-2]  # 이전 분봉 데이터
        current_open = df.iloc[-1]['open']  # 현재 분봉의 시가
        volatility = previous_candle['high'] - previous_candle['low']  # 이전 분봉의 변동성 계산
        target_price = current_open + (volatility * k)  # 매수 목표가 계산

        return target_price
    except Exception as e:
        print(f"{ticker} {interval} 변동성 돌파 가격을 계산하는 중 오류 발생: {e}")
        return None

# 업비트 로그인
upbit = pyupbit.Upbit(access, secret)
print("자동 거래 시작")

# Step 1: 각 종목에 대해 잔고 조회 및 초기 구매 결정
krw_balance = get_balance("KRW")  # 원화 잔고 조회
jan1_percent = krw_balance * initial_buy_percent  # 초기 매수 금액 설정

# 각 암호화폐에 대해 초기 구매 진행
for ticker in tickers:
    crypto_symbol = ticker.split("-")[1]  # 티커에서 암호화폐 심볼 추출
    crypto_balance = get_balance(crypto_symbol)  # 해당 암호화폐 잔고 조회
    current_price = safe_get_current_price(ticker)  # 현재 암호화폐의 시장 가격 조회
    target_price = get_volatility_breakout_price(ticker)  # 분봉 설정에 따른 매수 목표가 계산

    if current_price is None or target_price is None:
        continue  # 가격 정보를 가져오지 못했으면 다음 티커로 넘어감

    if current_price >= target_price:  # 현재 가격이 매수 목표가를 돌파했을 때
        buy1_amount = jan1_percent  # 잔고의 설정된 비율을 초기 구매에 사용
        print(f"{ticker} BUY1 진행 ({interval} 변동성 돌파 전략 사용): 금액: {buy1_amount} KRW")

        # 초기 매수 수행
        buy1_order = buy_crypto(ticker, buy1_amount)
        if buy1_order is not None:
            trade_state[ticker]['buy1_price'] = safe_get_current_price(ticker)  # 첫 매수 시점의 가격 기록
            print(f"{ticker} BUY1 구매 완료: 가격: {trade_state[ticker]['buy1_price']} KRW")

    time.sleep(3)  # 다음 종목을 처리하기 전 3초 대기

# Step 3: 수익률 모니터링 및 동적 추가 구매/매도 실행
while True:
    for ticker in tickers:
        current_price = safe_get_current_price(ticker)  # 현재 암호화폐 가격을 안전하게 가져오기
        if current_price is None:
            continue  # 가격을 가져오지 못했을 경우 다음 티커로 넘어감

        buy1_price = trade_state[ticker]['buy1_price']  # 첫 매수 가격 가져오기
        if buy1_price is not None:
            # 현재 가격과 첫 매수 가격을 비교하여 수익률 계산
            profit_rate = ((current_price - buy1_price) / buy1_price) * 100

            # 매도 조건 1: 수익률이 설정된 값 이상일 때 전체 매도
            if profit_rate >= profit_threshold and not trade_state[ticker]['sell_executed']:
                trade_state[ticker]['sell_executed'] = True
                crypto_balance = get_balance(ticker.split("-")[1])  # 해당 암호화폐 잔고 조회
                sell_order = sell_crypto(ticker, crypto_balance)  # 해당 암호화폐 전량 매도
                if sell_order is not None:
                    print(f"{ticker} 전액 매도 완료: 수익률 {profit_rate:.2f}%")
                    trade_state[ticker]['sell_executed'] = False  # 매수 조건 충족 시 다시 매수 가능하도록 플래그 초기화
                time.sleep(3)  # 매도 후 대기 시간

            # 매도 조건 2: 최종 매수 단계 이후 손실률이 설정된 값 이하일 때 전체 매도
            if trade_state[ticker]['buy7_executed'] and profit_rate <= loss_threshold_after_final_buy and not trade_state[ticker]['sell_executed']:
                trade_state[ticker]['sell_executed'] = True
                crypto_balance = get_balance(ticker.split("-")[1])  # 해당 암호화폐 잔고 조회
                sell_order = sell_crypto(ticker, crypto_balance)  # 해당 암호화폐 전량 매도
                if sell_order is not None:
                    print(f"{ticker} 전액 매도 완료: 손실률 {profit_rate:.2f}% (최종 매수 단계 이후)")
                    trade_state[ticker]['sell_executed'] = False  # 매도 후 플래그 초기화
                time.sleep(3)  # 매도 후 대기 시간

            # 추가 매수 및 매도 단계별 조건 평가
            for i, condition in enumerate(additional_buy_conditions):
                # 추가 매수 조건: 손실률이 설정된 값 이하이고, 현재 가격이 변동성 돌파 목표가 이상일 때
                additional_target_price = get_volatility_breakout_price(ticker)  # 추가 매수를 위한 변동성 돌파 목표가 재계산
                if (profit_rate <= condition['trigger_loss']) and (current_price >= additional_target_price) and not trade_state[f'buy{i+2}_executed'] and i < 7:
                    trade_state[f'buy{i+2}_executed'] = True
                    # 추가 매수 금액을 설정된 비율로 계산
                    buy_amount = (krw_balance * condition['buy_ratio'])
                    buy_order = buy_crypto(ticker, buy_amount)
                    if buy_order is not None:
                        krw_balance -= buy_amount  # 매수 후 잔고 차감
                        print(f"{ticker} BUY{i+2} 추가 구매 완료: 금액: {buy_amount} KRW, 가격: {current_price} KRW")
                    time.sleep(3)

        time.sleep(10)  # 각 티커의 가격을 10초마다 확인
