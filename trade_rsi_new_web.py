import time
import pyupbit
import websockets
import asyncio
import json

key_file_path = r'C:\Users\winne\OneDrive\바탕 화면\upbit_key.txt'

# 요청 제한 관리용 변수
request_count = 0
max_request_per_minute = 100  # 분당 최대 요청 수
max_request_per_second = 1     # 초당 최대 요청 수

# 1. 로그인
with open(key_file_path, 'r') as file:
    access = file.readline().strip()
    secret = file.readline().strip()

upbit = pyupbit.Upbit(access, secret)

# 2. 종목 리스트 (상위 시가 총액 33종목)
tickers = [
    "KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-EOS", "KRW-ADA", 
    "KRW-DOGE", "KRW-LOOM", "KRW-SHIB", "KRW-NEO", "KRW-UXLINK", "KRW-HBAR", "KRW-STPT", "KRW-SEI",
    "KRW-ZRO", "KRW-HIVE", "KRW-SOL", "KRW-HIFI", "KRW-TFUEL", 
    "KRW-WAVES", "KRW-CVC", "KRW-W", "KRW-ARK", "KRW-STX", 
    "KRW-UPP", "KRW-CHZ", "KRW-SNT", "KRW-BLUR", "KRW-APT", 
    "KRW-DKA", "KRW-ATH", "KRW-NEAR", "KRW-ONG","KRW-SUI","KRW-ORBS",
]

# 3. 변수 설정
rsi_period = 14
rsi_threshold = 20  # RSI 21 이하일 때 매수
initial_invest_ratio = 0.005  # 잔고의 0.5%
target_profit_rate = 0.0045  # 목표 수익률 0.45%
stop_loss_rate = -0.025  # 손절매 -2.5%
maintain_profit_rate = -0.005  # 추가 매수를 위한 수익률 조건

# 요청 제한 체크 함수
async def rate_limit_check():
    global request_count
    if request_count >= max_request_per_minute:
        print("분당 요청 제한 도달, 10초 대기 중...")
        await asyncio.sleep(10)
        request_count = 0
    elif request_count % max_request_per_second == 0:
        print("초당 요청 제한 도달, 1초 대기 중...")
        await asyncio.sleep(0.2)

# 4. RSI 계산 함수
def get_rsi(ticker):
    df = pyupbit.get_ohlcv(ticker, interval="minute1", count=rsi_period + 1)
    close_prices = df['close']  # 종가 데이터만 사용

    # RSI 계산
    delta = close_prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=rsi_period, min_periods=1).mean()
    avg_loss = loss.rolling(window=rsi_period, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]  # 가장 최근 RSI 값

# 5. 현재가 매수 후 미체결 시 재주문 (최대 3회 재시도)
def place_buy_order(ticker, krw_balance, invest_amount, attempt=1):
    current_price = pyupbit.get_current_price(ticker)  # 현재가 확인
    try:
        # 현재가 매수
        order = upbit.buy_limit_order(ticker, current_price, invest_amount / current_price)
        print(f"{ticker} 현재가 매수 주문 - 가격: {current_price}, 금액: {invest_amount} KRW, 시도 횟수: {attempt}")

        time.sleep(10)  # 10초 대기 후 체결 확인
        order_info = upbit.get_order(order['uuid'])  # 주문 정보 가져오기

        # 미체결 시 주문 취소 및 재주문 (최대 3회 재시도)
        if order_info['state'] != 'done':  # 주문이 체결되지 않음
            print(f"{ticker} 매수 주문 미체결 - 주문 취소 후 재시도")
            upbit.cancel_order(order['uuid'])
            time.sleep(1)
            if attempt < 3:
                place_buy_order(ticker, krw_balance, invest_amount, attempt + 1)  # 재주문
            else:
                print(f"{ticker} 매수 주문 3회 시도 후 미체결, 다음으로 넘어갑니다.")
        else:
            print(f"{ticker} 매수 주문 체결 완료 - 가격: {current_price}")
    except Exception as e:
        print(f"{ticker} 매수 주문 실패: {e}")

# 6. 현재가 매도 후 미체결 시 재주문 (최대 3회 재시도)
def place_sell_order(ticker, balance, attempt=1):
    current_price = pyupbit.get_current_price(ticker)  # 현재가 확인
    try:
        # 현재가 매도
        order = upbit.sell_limit_order(ticker, current_price, balance)
        print(f"{ticker} 현재가 매도 주문 - 가격: {current_price}, 수량: {balance}, 시도 횟수: {attempt}")

        time.sleep(10)  # 10초 대기 후 체결 확인
        order_info = upbit.get_order(order['uuid'])  # 주문 정보 가져오기

        # 미체결 시 주문 취소 및 재주문 (최대 3회 재시도)
        if order_info['state'] != 'done':  # 주문이 체결되지 않음
            print(f"{ticker} 매도 주문 미체결 - 주문 취소 후 재시도")
            upbit.cancel_order(order['uuid'])
            time.sleep(1)
            if attempt < 3:
                place_sell_order(ticker, balance, attempt + 1)  # 재주문
            else:
                print(f"{ticker} 매도 주문 3회 시도 후 미체결, 다음으로 넘어갑니다.")
        else:
            print(f"{ticker} 매도 주문 체결 완료 - 가격: {current_price}")
    except Exception as e:
        print(f"{ticker} 매도 주문 실패: {e}")

# 7. 실시간 가격 모니터링 함수
async def watch_price(tickers):
    global request_count
    while True:  # 끊어지면 다시 연결 시도
        try:
            url = "wss://api.upbit.com/websocket/v1"
            async with websockets.connect(url) as websocket:
                subscribe_data = [
                    {"ticket": "test"},
                    {"type": "ticker", "codes": tickers, "isOnlyRealtime": True},
                    {"format": "SIMPLE"}
                ]
                await websocket.send(json.dumps(subscribe_data))
                request_count += 1  # 요청 수 증가

                while True:
                    await rate_limit_check()  # 요청 제한 체크
                    data = await websocket.recv()
                    request_count += 1  # 요청 수 증가
                    data = json.loads(data)

                    # 'cd'는 종목 코드, 'tp'는 거래 가격
                    if 'cd' in data and 'tp' in data:
                        ticker = data['cd']
                        current_price = data['tp']
                        print(f"{ticker} 실시간 가격: {current_price}")

                        # 현재 보유 수량 및 평균 매수가
                        balance = upbit.get_balance(ticker)
                        avg_buy_price = upbit.get_avg_buy_price(ticker)

                        # 잔고가 있을 때는 수익률 관리 및 손절매만 수행
                        if balance > 0:
                            profit_rate = (current_price - avg_buy_price) / avg_buy_price
                            print(f"{ticker} 보유 수량 {balance} | 수익률: {profit_rate*100:.2f}%")

                            # 목표 수익률 도달 시 매도
                            if profit_rate >= target_profit_rate:
                                place_sell_order(ticker, balance)  # 현재가 매도 실행
                                continue

                            # 손절매 도달 시 매도
                            if profit_rate <= stop_loss_rate:
                                place_sell_order(ticker, balance)  # 현재가 매도 실행
                                continue
                            
                            # **추가 매수 조건**: 수익률이 -0.5% 이하이고, RSI가 21 이하일 때 추가 매수
                            if profit_rate <= maintain_profit_rate and get_rsi(ticker) < 21:
                                krw_balance = upbit.get_balance("KRW")
                                invest_amount = krw_balance * initial_invest_ratio * 2  # 2배로 추가 매수
                                fee = invest_amount * 0.0005  # 수수료 계산
                                total_invest_amount = invest_amount + fee  # 수수료 포함한 총 금액

                                if total_invest_amount > 5000:  # 최소 주문 금액 확인
                                    if krw_balance >= total_invest_amount:
                                        place_buy_order(ticker, krw_balance, invest_amount)  # 추가 매수 실행
                                    else:
                                        print(f"{ticker} 추가 매수 실패 - 잔고 부족. 잔액: {krw_balance}, 필요 금액: {total_invest_amount}")
                                else:
                                    print(f"{ticker} 추가 매수 실패 - 최소 주문 금액 미만. 금액: {total_invest_amount}")
                                await rate_limit_check()  # 요청 제한 체크
                      
                        # 잔고가 없을 때만 매수 수행
                        elif balance == 0:
                            rsi = get_rsi(ticker)
                            print(f"{ticker} RSI: {rsi:.2f}")
                            if rsi < rsi_threshold:
                                krw_balance = upbit.get_balance("KRW")
                                invest_amount = krw_balance * initial_invest_ratio
                                if invest_amount > 5000:  # 최소 주문 금액 체크
                                    place_buy_order(ticker, krw_balance, invest_amount)  # 현재가 매수 실행
                                else:
                                    print(f"{ticker} 매수 실패 - 잔액 부족")
                                await rate_limit_check()  # 요청 제한 체크
                                time.sleep(0.1)

        except websockets.exceptions.ConnectionClosedError as e:
            print(f"웹소켓 연결 끊김, 재연결 시도 중: {e}")
            await asyncio.sleep(5)  # 5초 대기 후 재연결 시도
        except Exception as e:
            print(f"예기치 못한 오류 발생: {e}")
            await asyncio.sleep(5)  # 5초 대기 후 재연결 시도

# 8. asyncio를 사용하여 웹소켓 시작
async def main():
    await watch_price(tickers)

asyncio.run(main())