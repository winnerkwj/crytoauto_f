import asyncio
import time
import pyupbit
import json
import pandas as pd
import websockets
import aiohttp
import logging
from collections import defaultdict

# 업비트 API 키 파일 경로 설정
key_file_path = r'C:\Users\winne\OneDrive\바탕 화면\upbit_key.txt'

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 요청 제한을 관리하는 클래스 정의
class RateLimiter:
    def __init__(self, max_calls, period=1.0):
        self.max_calls = max_calls      # 최대 호출 수
        self.period = period            # 시간 간격 (초 단위)
        self.calls = []
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        async with self.lock:
            now = asyncio.get_event_loop().time()
            # 기간 내의 호출 기록만 유지
            self.calls = [call for call in self.calls if call > now - self.period]
            if len(self.calls) >= self.max_calls:
                sleep_time = self.calls[0] + self.period - now
                await asyncio.sleep(sleep_time)
            self.calls.append(now)
    
    async def __aenter__(self):
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        pass

# 글로벌 락 선언
order_lock = asyncio.Lock()

# 1. 로그인
with open(key_file_path, 'r') as file:
    access = file.readline().strip()  # 액세스 키 읽기
    secret = file.readline().strip()  # 시크릿 키 읽기

upbit = pyupbit.Upbit(access, secret)  # Upbit 객체 생성

# 요청 제한 관리 객체 생성
# EXCHANGE API 요청 제한
order_request_limiter = RateLimiter(max_calls=8, period=1.0)       # 주문 요청 (POST /v1/orders)
non_order_request_limiter = RateLimiter(max_calls=30, period=1.0)  # 주문 외 요청

# QUOTATION API 요청 제한 (엔드포인트별로 관리)
public_api_limiters = defaultdict(lambda: RateLimiter(max_calls=10, period=1.0))  # 엔드포인트별로 요청 제한 관리

# 2. 동적 종목 리스트 생성 (상위 거래량 15종목)
async def get_top_volume_tickers(limit=30):
    endpoint = 'get_tickers'
    async with public_api_limiters[endpoint]:
        tickers = pyupbit.get_tickers(fiat="KRW")  # 원화로 거래되는 모든 종목 코드 가져오기
    
    url = "https://api.upbit.com/v1/ticker"  # 종목 정보 조회를 위한 API 엔드포인트
    params = {'markets': ','.join(tickers)}   # 모든 종목 코드를 콤마로 구분하여 요청 파라미터로 설정

    # 비동기 HTTP 세션을 열어서 API 요청
    endpoint = 'ticker'
    async with public_api_limiters[endpoint]:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()  # API 응답을 JSON 형태로 파싱

    # 각 종목의 24시간 누적 거래대금을 기준으로 내림차순 정렬
    data.sort(key=lambda x: x['acc_trade_price_24h'], reverse=True)
    # 상위 limit 개수만큼의 종목 코드를 리스트로 추출
    top_tickers = [x['market'] for x in data[:limit]]
    return top_tickers  # 상위 거래량 종목 리스트 반환

# 3. 변수 설정
rsi_period = 14             # RSI 계산에 사용할 기간 (14분)
rsi_threshold = 18          # RSI가 21 이하일 때 매수
rsi_threshold_additional = 35  # 추가 매수를 위한 RSI 임계값 (29 이하)
initial_invest_ratio = 0.005# 초기 투자 비율 (잔고의 0.5%)
target_profit_rate = 0.0035   # 목표 수익률 (0.35%)
stop_loss_rate = -0.028       # 손절매 기준 (-2.5%)
maintain_profit_rate = -0.005 # 추가 매수 기준 수익률 (-0.5%)

# RSI 계산 주기 (초 단위)
rsi_calculation_interval = 5  # 5초마다 RSI 계산

# 추가 매수를 위한 최소 보유 시간 (초 단위)
min_hold_time_for_additional_buy = 3  # 3초

# 종목별 보유 시작 시간 저장 딕셔너리
hold_start_time = {}

# 종목별 추가 매수 횟수 저장 딕셔너리
additional_buy_count = defaultdict(int)
max_additional_buys = 100  # 종목별 최대 추가 매수 횟수

# 보유 종목 리스트 관리 딕셔너리
holding_tickers = {}  # 종목별 보유 수량 저장

# 4. RSI 값 캐싱을 위한 딕셔너리 및 타임스탬프
rsi_cache = {}
rsi_timestamp = {}

# 5. RSI 계산 함수 (Wilder's Moving Average 사용)
async def get_rsi(ticker):
    now = time.time()
    # 이전에 계산한 RSI가 있고, 계산한 지 일정 시간이 지나지 않았다면 캐시된 값 사용
    if ticker in rsi_cache and now - rsi_timestamp.get(ticker, 0) < rsi_calculation_interval:
        return rsi_cache[ticker]
    try:
        endpoint = 'ohlcv'
        async with public_api_limiters[endpoint]:
            # 지정한 종목의 과거 가격 데이터를 가져옴
            df = pyupbit.get_ohlcv(ticker, interval="minute1", count=rsi_period * 2)
        if df is None or df.empty:
            logging.warning(f"{ticker} 데이터 수집 실패 또는 데이터 없음")
            return None  # 데이터가 없을 경우 None 반환
        close = df['close']  # 종가 데이터만 추출
        delta = close.diff().dropna()  # 종가의 변화량 계산
        gain = delta.clip(lower=0)     # 상승분만 추출
        loss = -delta.clip(upper=0)    # 하락분만 추출 (양수로 변환)
        # Wilder's Moving Average 계산
        avg_gain = gain.ewm(alpha=1/rsi_period, min_periods=rsi_period).mean()
        avg_loss = loss.ewm(alpha=1/rsi_period, min_periods=rsi_period).mean()
        rs = avg_gain / avg_loss       # RS 계산
        rsi = 100 - (100 / (1 + rs))   # RSI 계산
        rsi_value = rsi.iloc[-1]       # 가장 최근의 RSI 값
        # 캐시에 저장
        rsi_cache[ticker] = rsi_value
        rsi_timestamp[ticker] = now
        return rsi_value            # RSI 값 반환
    except Exception as e:
        logging.error(f"{ticker} RSI 계산 중 오류 발생: {e}")
        return None  # 오류 발생 시 None 반환

# 6. 현재가 매수 후 미체결 시 재주문 (최대 1회 재시도)
async def place_buy_order(ticker, krw_balance, invest_amount):
    max_attempts = 1  # 최대 시도 횟수 설정
    for attempt in range(1, max_attempts + 1):
        endpoint = 'current_price'
        async with public_api_limiters[endpoint]:
            current_price = pyupbit.get_current_price(ticker)  # 현재 가격 조회
        try:
            async with order_request_limiter:
                # 현재가로 지정가 매수 주문
                order = upbit.buy_limit_order(ticker, current_price, invest_amount / current_price)
            logging.info(f"{ticker} 매수 주문 시도 {attempt}회차 - 가격: {current_price}, 금액: {invest_amount} KRW")
            await asyncio.sleep(10)  # 주문 체결 대기 시간
            async with non_order_request_limiter:
                order_info = upbit.get_order(order['uuid'])  # 주문 정보 조회
            if order_info and order_info.get('state') == 'done':
                logging.info(f"{ticker} 매수 주문 체결 완료")
                # 보유 종목 리스트에 추가
                holding_tickers[ticker] = upbit.get_balance(ticker)
                hold_start_time[ticker] = time.time()  # 보유 시작 시간 저장
                return  # 주문이 체결되면 함수 종료
            else:
                logging.info(f"{ticker} 매수 주문 미체결 - 주문 취소 후 재시도")
                async with non_order_request_limiter:
                    upbit.cancel_order(order['uuid'])  # 미체결 주문 취소
                await asyncio.sleep(1)  # 잠시 대기 후 재시도
        except Exception as e:
            logging.error(f"{ticker} 매수 주문 실패: {e}")
            await asyncio.sleep(1)  # 오류 발생 시 잠시 대기 후 재시도
    logging.error(f"{ticker} 매수 주문 실패 - 최대 시도 횟수 초과")  # 최대 시도 횟수를 초과하면 실패 메시지 출력

# 7. 현재가 매도 후 미체결 시 재주문 (최대 3회 재시도)
async def place_sell_order(ticker, balance):
    max_attempts = 1  # 최대 시도 횟수 설정
    for attempt in range(1, max_attempts + 1):
        endpoint = 'current_price'
        async with public_api_limiters[endpoint]:
            current_price = pyupbit.get_current_price(ticker)  # 현재 가격 조회
        try:
            async with order_request_limiter:
                # 현재가로 지정가 매도 주문
                order = upbit.sell_limit_order(ticker, current_price, balance)
            logging.info(f"{ticker} 매도 주문 시도 {attempt}회차 - 가격: {current_price}, 수량: {balance}")
            await asyncio.sleep(10)  # 주문 체결 대기 시간
            async with non_order_request_limiter:
                order_info = upbit.get_order(order['uuid'])  # 주문 정보 조회
            if order_info and order_info.get('state') == 'done':
                logging.info(f"{ticker} 매도 주문 체결 완료")
                # 보유 종목 리스트에서 제거
                holding_tickers.pop(ticker, None)
                # 추가 매수 횟수 초기화
                additional_buy_count.pop(ticker, None)
                # 보유 시작 시간 초기화
                hold_start_time.pop(ticker, None)
                return  # 주문이 체결되면 함수 종료
            else:
                logging.info(f"{ticker} 매도 주문 미체결 - 주문 취소 후 재시도")
                async with non_order_request_limiter:
                    upbit.cancel_order(order['uuid'])  # 미체결 주문 취소
                await asyncio.sleep(1)  # 잠시 대기 후 재시도
        except Exception as e:
            logging.error(f"{ticker} 매도 주문 실패: {e}")
            await asyncio.sleep(1)  # 오류 발생 시 잠시 대기 후 재시도
    logging.error(f"{ticker} 매도 주문 실패 - 최대 시도 횟수 초과")  # 최대 시도 횟수를 초과하면 실패 메시지 출력


# 8. 실시간 가격 모니터링 함수 (수정된 부분)
async def watch_price():
    url = "wss://api.upbit.com/websocket/v1"
    previous_prices = {}
    previous_profit_rates = {}
    last_update = 0  # 초기값을 0으로 설정하여 즉시 갱신되도록 함
    update_interval = 3600  # 1시간 (3600초)

    while True:
        # 종목 리스트 갱신
        if time.time() - last_update >= update_interval:
            tickers = await get_top_volume_tickers()
            last_update = time.time()
            logging.info("상위 거래량 종목 리스트 갱신")

        # 보유 종목 리스트와 합치기
        all_tickers = list(set(tickers + list(holding_tickers.keys())))

        try:
            async with websockets.connect(url, ping_interval=60, ping_timeout=10) as websocket:
                subscribe_data = [
                    {"ticket": "test"},
                    {"type": "ticker", "codes": all_tickers, "isOnlyRealtime": True},
                    {"format": "SIMPLE"}
                ]
                await websocket.send(json.dumps(subscribe_data))

                while True:
                    data = await websocket.recv()
                    data = json.loads(data)
                    if 'cd' in data and 'tp' in data:
                        ticker = data['cd']
                        current_price = data['tp']

                        # RSI 값 가져오기
                        rsi = await get_rsi(ticker)
                        if rsi is not None:
                            rsi_str = f"{rsi:.2f}"
                        else:
                            rsi_str = "N/A"

                        # 가격 변동이 있을 때만 출력
                        if ticker not in previous_prices or previous_prices[ticker] != current_price:
                            logging.info(f"{ticker} 실시간 가격: {current_price}, RSI: {rsi_str}")
                            previous_prices[ticker] = current_price

                        # 현재 보유 수량 및 평균 매수가 조회
                        async with non_order_request_limiter:
                            balance = upbit.get_balance(ticker)
                            avg_buy_price = upbit.get_avg_buy_price(ticker)

                        if balance > 0:
                            # 수익률 계산 (수수료 미고려)
                            profit_rate = (current_price - float(avg_buy_price)) / float(avg_buy_price)

                            # 수익률 변동이 있을 때만 출력
                            if ticker not in previous_profit_rates or abs(previous_profit_rates[ticker] - profit_rate) >= 0.0001:
                                logging.info(f"{ticker} 보유 수량: {balance}, 수익률: {profit_rate*100:.2f}%")
                                previous_profit_rates[ticker] = profit_rate

                            # 수수료를 고려한 목표 수익률 및 손절매 수익률
                            adjusted_target_profit_rate = target_profit_rate + 0.001  # 0.1% 추가
                            adjusted_stop_loss_rate = stop_loss_rate - 0.001          # 0.1% 추가

                            if profit_rate >= adjusted_target_profit_rate:
                                await place_sell_order(ticker, balance)
                                continue
                            if profit_rate <= adjusted_stop_loss_rate:
                                await place_sell_order(ticker, balance)
                                continue
                            if profit_rate <= maintain_profit_rate:
                                # 최소 보유 시간이 지났는지 확인
                                if ticker in hold_start_time and time.time() - hold_start_time[ticker] >= min_hold_time_for_additional_buy:
                                    # 추가 매수 횟수 확인
                                    if additional_buy_count[ticker] < max_additional_buys:
                                        if rsi is not None and rsi < rsi_threshold_additional:
                                            logging.info(f"{ticker} RSI: {rsi:.2f}")
                                            async with non_order_request_limiter:
                                                krw_balance = upbit.get_balance("KRW")
                                            invest_amount = krw_balance * initial_invest_ratio
                                            fee = invest_amount * 0.0005
                                            total_invest_amount = invest_amount + fee
                                            if total_invest_amount > 5000 and krw_balance >= total_invest_amount:
                                                await place_buy_order(ticker, krw_balance, invest_amount)
                                                additional_buy_count[ticker] += 1  # 추가 매수 횟수 증가
                                            else:
                                                logging.info(f"{ticker} 추가 매수 실패 - 잔고 부족 또는 최소 금액 미만")
                        else:
                            if rsi is not None and rsi < rsi_threshold:
                                async with order_lock:
                                    # 락 안에서 잔고 재확인
                                    async with non_order_request_limiter:
                                        balance = upbit.get_balance(ticker)
                                    if balance == 0:
                                        logging.info(f"{ticker} RSI: {rsi:.2f}")
                                        async with non_order_request_limiter:
                                            krw_balance = upbit.get_balance("KRW")
                                        invest_amount = krw_balance * initial_invest_ratio
                                        if invest_amount > 5000:
                                            await place_buy_order(ticker, krw_balance, invest_amount)
                                        else:
                                            logging.info(f"{ticker} 매수 실패 - 잔고 부족")
                        await asyncio.sleep(0.1)  # 대기 시간 최소화

                    # 종목 리스트 갱신 체크
                    if time.time() - last_update >= update_interval:
                        logging.info("종목 리스트 갱신을 위해 웹소켓 연결을 종료합니다.")
                        break  # 내부 루프 종료하여 웹소켓 재연결

        except websockets.exceptions.ConnectionClosedError as e:
            logging.warning(f"웹소켓 연결 끊김, 재연결 시도 중: {e}")
            await asyncio.sleep(2)
        except Exception as e:
            logging.error(f"예기치 못한 오류 발생: {e}")
            await asyncio.sleep(2)

# 9. 메인 함수
async def main():
    await watch_price()

# 프로그램 시작
if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("프로그램이 사용자에 의해 중단되었습니다.")
