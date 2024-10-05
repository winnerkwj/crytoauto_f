import time
import pyupbit
import websockets
import asyncio
import json

key_file_path = r'C:\Users\winne\OneDrive\바탕 화면\upbit_key.txt'

# 1. 로그인
with open(key_file_path, 'r') as file:
    access = file.readline().strip()
    secret = file.readline().strip()

upbit = pyupbit.Upbit(access, secret)

# 2. 종목 리스트 (상위 시가 총액 20종목)
tickers = [
    "KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-EOS", "KRW-ADA", 
    "KRW-DOGE", "KRW-LOOM", "KRW-SHIB", "KRW-NEO", "KRW-UXLINK", 
    "KRW-ARDR", "KRW-GAS", "KRW-HBAR", "KRW-STPT", "KRW-SEI",
    "KRW-ZRO", "KRW-HIVE", "KRW-SOL", "KRW-HIFI", "KRW-TFUEL", 
    "KRW-WAVES", "KRW-CVC", "KRW-W"
]

# 3. 변수 설정
rsi_period = 14
rsi_threshold = 21  # RSI 23 이하일 때 매수
initial_invest_ratio = 0.01  # 잔고의 1%
target_profit_rate = 0.0045  # 목표 수익률 0.45%
stop_loss_rate = -0.025  # 손절매 -2.5%
maintain_profit_rate = -0.005  # 추가 매수를 위한 수익률 조건

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

# 5. 실시간 가격 모니터링 함수
async def watch_price(tickers):
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

                while True:
                    data = await websocket.recv()
                    data = json.loads(data)

                    # 'cd'는 종목 코드, 'tp'는 거래 가격
                    if 'cd' in data and 'tp' in data:
                        ticker = data['cd']
                        current_price = data['tp']
                        print(f"{ticker} 실시간 가격: {current_price}")

                        # 현재 보유 수량 및 평균 매수가
                        balance = upbit.get_balance(ticker)
                        avg_buy_price = upbit.get_avg_buy_price(ticker)

                        # RSI 계산 및 매매 조건 확인
                        rsi = get_rsi(ticker)
                        print(f"{ticker} RSI: {rsi:.2f}")

                        # 5.1 초기 매수 (RSI 30 이하일 때 매수)
                        if balance == 0 and rsi < rsi_threshold:
                            krw_balance = upbit.get_balance("KRW")
                            invest_amount = krw_balance * initial_invest_ratio
                            if invest_amount > 5000:  # 최소 주문 금액 체크
                                order = upbit.buy_market_order(ticker, invest_amount)
                                print(f"{ticker} 매수 주문 완료 - 금액: {invest_amount} KRW")
                            else:
                                print(f"{ticker} 매수 실패 - 잔액 부족")
                            time.sleep(0.1)

                        # 5.2 추가 매수 및 수익률 관리
                        elif balance > 0:
                            profit_rate = (current_price - avg_buy_price) / avg_buy_price
                            print(f"{ticker} 수익률: {profit_rate*100:.2f}%")

                            # 목표 수익률 도달 시 매도
                            if profit_rate >= target_profit_rate:
                                order = upbit.sell_market_order(ticker, balance)
                                print(f"{ticker} 매도 주문 완료 - 수익 실현")
                                time.sleep(0.1)
                                continue

                            # 손절매 도달 시 매도
                            if profit_rate <= stop_loss_rate:
                                order = upbit.sell_market_order(ticker, balance)
                                print(f"{ticker} 매도 주문 완료 - 손절매 실행")
                                time.sleep(0.1)
                                continue

                            # 추가 매수: 수익률이 -0.5% 이하이고 RSI가 30 이하일 때
                            if profit_rate <= maintain_profit_rate and rsi < rsi_threshold:
                                krw_balance = upbit.get_balance("KRW")
                                invest_amount = krw_balance * initial_invest_ratio * 2  # 2배로 추가 매수
                                fee = invest_amount * 0.0005  # 수수료 계산 (약 0.05%)
                                total_invest_amount = invest_amount + fee  # 수수료 포함한 총 금액

                                if total_invest_amount > 5000:  # 최소 주문 금액 확인
                                    if krw_balance >= total_invest_amount:
                                        order = upbit.buy_market_order(ticker, invest_amount)
                                        print(f"{ticker} 추가 매수 완료 - 금액: {invest_amount} KRW")
                                    else:
                                        print(f"{ticker} 추가 매수 실패 - 잔고 부족. 잔액: {krw_balance}, 필요 금액: {total_invest_amount}")
                                else:
                                    print(f"{ticker} 추가 매수 실패 - 최소 주문 금액 미만. 금액: {total_invest_amount}")
                                time.sleep(0.1)

        except websockets.exceptions.ConnectionClosedError as e:
            print(f"웹소켓 연결 끊김, 재연결 시도 중: {e}")
            await asyncio.sleep(5)  # 5초 대기 후 재연결 시도
        except Exception as e:
            print(f"예기치 못한 오류 발생: {e}")
            await asyncio.sleep(5)  # 5초 대기 후 재연결 시도

# 6. asyncio를 사용하여 웹소켓 시작
async def main():
    await watch_price(tickers)

asyncio.run(main())
