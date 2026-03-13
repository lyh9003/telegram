import os
import json
import re
import time
import logging
import pandas as pd
from collections import Counter
from datetime import datetime, timezone, timedelta

KST = timezone(timedelta(hours=9))
from telethon.sync import TelegramClient
from openai import OpenAI

# ====== 로깅 설정 ======
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ====== 반도체 키워드 사전 정의 ======
COMPANY = [
    r'\b(TSMC|TSM)\b', r'\bASML\b', r'\bMicron\b', r'\bMU\b', r'\bNVIDIA|NVDA\b', r'\bAMD\b',
    r'삼성전자', r'하이닉스', r'삼전', r'엔비디아', r'인텔', r'Intel', r'퀄컴', r'Qualcomm'
]
MEMORY = [
    r'\bHBM\b', r'\bDDR\b', r'\bLPDDR\b',
    r'DRAM', r'NAND', r'플래시', r'메모리\b', r'SSD\b', r'솔리드스테이트'
]
PRODUCT = [
    r'파운드리', r'\bfoundry\b', r'GPU', r'CPU', r'ASIC', r'NPU', r'SoC', r'\bFPGA\b', r'칩렛|chiplet',
    r'AI칩', r'반도체', r'semiconductor'
]
TECH = [
    r'EUV', r'DUV', r'포토|리소그래피|lithography', r'식각|에칭|etch', r'증착|CVD',
    r'GAA|FinFET', r'nm\b|나노미터', r'노광', r'수율|yield', r'웨이퍼|wafer', r'칩|die\b',
]

CSV_COLUMNS = [
    'channel', 'sender_id', 'date_utc', 'date_local', 'labels',
    'message', 'normalized_text', 'message_length',
    'forward_count', 'forward_channels',
    'summary', 'keywords', 'sentiment'
]

# ====== 채널 로드 ======
def load_channels(file_name='channels.txt'):
    """channels.txt에서 채널 목록 로드 (# 주석 지원)"""
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            channels = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        logger.info(f"채널 {len(channels)}개 로드 완료: {file_name}")
        return channels
    except FileNotFoundError:
        logger.error(f"채널 파일을 찾을 수 없습니다: {file_name}")
        return []

# ====== 파일 관리 함수들 ======
def load_existing_data(file_name):
    """기존 CSV 파일 로드"""
    logger.info(f"기존 데이터 로드 시도: {file_name}")
    if os.path.exists(file_name):
        try:
            existing_df = pd.read_csv(file_name, encoding='utf-8-sig')
            logger.info(f"기존 파일 로드 완료: {len(existing_df)}개")
            return existing_df
        except Exception as e:
            logger.error(f"기존 파일 로드 실패: {e}")
            return pd.DataFrame()
    else:
        logger.info("기존 파일이 없습니다. 새로운 데이터를 생성합니다.")
        return pd.DataFrame()

def merge_and_remove_duplicates(existing_df, new_df):
    """기존 데이터와 새 데이터 병합 및 중복 제거 (forward_count는 최신값 우선)"""
    logger.info(f"데이터 병합 시작 - 기존: {len(existing_df)}개, 새로운: {len(new_df)}개")

    if not existing_df.empty and not new_df.empty:
        combined_df = pd.concat([new_df, existing_df], ignore_index=True)  # 새 데이터 먼저 → keep='first'로 최신값 유지
    elif not new_df.empty:
        logger.info(f"새 데이터만 사용합니다. 데이터 개수: {len(new_df)}개")
        return new_df
    else:
        logger.info("새 데이터가 없습니다.")
        return existing_df

    before_dedup = len(combined_df)
    deduped = combined_df.drop_duplicates(subset='normalized_text', keep='first').reset_index(drop=True)
    after_dedup = len(deduped)
    logger.info(f"중복 제거 완료: {before_dedup}개 → {after_dedup}개 (제거: {before_dedup - after_dedup}개)")
    return deduped

def save_updated_data(data, file_name):
    """데이터를 CSV 파일로 저장"""
    logger.info(f"데이터 저장 시도: {file_name} ({len(data)}개 행)")
    try:
        if data.empty:
            pd.DataFrame(columns=CSV_COLUMNS).to_csv(file_name, encoding='utf-8-sig', index=False)
            logger.info(f"빈 CSV 파일이 생성되었습니다: {file_name}")
        else:
            data.to_csv(file_name, encoding='utf-8-sig', index=False)
            logger.info(f"저장 완료: {file_name} ({os.path.getsize(file_name)} bytes)")
    except Exception as e:
        logger.error(f"파일 저장 실패: {e}")

# ====== 텍스트 처리 함수들 ======
def compile_any(patterns):
    """정규식 패턴들을 하나로 컴파일"""
    try:
        return re.compile(r'(' + '|'.join(patterns) + r')', re.IGNORECASE)
    except Exception as e:
        logger.error(f"정규식 컴파일 오류: {e}")
        return re.compile(r'반도체|semiconductor', re.IGNORECASE)

def normalize_text(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()

def extract_labels(text: str, rx_dict):
    return [k for k, rx in rx_dict.items() if rx.search(text)]

def is_semiconductor_related(text: str, fast_rx) -> bool:
    return bool(fast_rx.search(text))

# ====== GPT 분석 함수 (요약 + 키워드 + 감성 통합) ======
def analyze_with_gpt(text: str, openai_client: OpenAI):
    """GPT로 요약, 키워드, 감성을 한 번에 분석"""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "당신은 반도체 및 기술 분야 텍스트를 분석하는 전문가입니다. 반드시 JSON 형식으로만 답변하세요."
                },
                {
                    "role": "user",
                    "content": f"""다음 텍스트를 분석하세요.

요약 조건:
- 반드시 "1. 내용\n2. 내용\n3. 내용" 형식으로 번호를 붙여 작성 (번호 생략 절대 금지)
- 짧은 글: 1~2개 항목, 긴 글: 최대 5개 항목
- ~했음, ~함, ~였음, ~임, ~있음 식 간결체 사용
- 각 항목 사이 줄바꿈(\n) 포함

감성 분류 기준 (삼성전자 메모리 입장 기준):
- 긍정적: AI/반도체 수요 증가, 설비 투자 확대, 실적 개선, 공급 부족(수요 강세), 메모리 판가 상승, GPU 수요 급증, 데이터센터 투자 확대, 메모리 경쟁우위
- 부정적: 수요 감소, 재고 증가, 규제 강화, 경쟁 심화, 실적 악화, 메모리 판가 하락, 감산, 수출 제한
- 중립적: 단순 현황 보고, 인사/조직 변경, 방향 불명확한 내용
※ 서비스 불안정·과부하 등 간접 신호도 수요 강세면 긍정적으로 분류

텍스트: {text}

아래 JSON 형식으로만 답변:
{{"summary": "요약 내용", "keywords": ["키워드1", "키워드2", ...], "sentiment": "긍정적|부정적|중립적"}}"""
                }
            ],
            max_tokens=700,
            temperature=0.2
        )

        result = json.loads(response.choices[0].message.content)
        summary = result.get('summary', '요약 없음')
        keywords = ', '.join(result.get('keywords', []))
        sentiment = result.get('sentiment', '중립적')

        logger.debug(f"GPT 분석 완료 - 감성: {sentiment}, 키워드: {keywords}")
        return summary, keywords, sentiment

    except Exception as e:
        logger.error(f"GPT 분석 오류: {e}")
        return "분석 실패", "분석 실패", "분석 실패"

# ====== 메인 크롤링 함수 ======
def crawl_telegram_messages(channel_usernames, api_id, api_hash, limit_per_channel=30):
    """텔레그램 메시지 크롤링"""
    logger.info("텔레그램 메시지 크롤링 시작...")

    RX = {
        'Company': compile_any(COMPANY),
        'Memory': compile_any(MEMORY),
        'Product': compile_any(PRODUCT),
        'Tech': compile_any(TECH)
    }
    FAST_ANY = compile_any(COMPANY + MEMORY + PRODUCT + TECH)

    messages_data = []

    try:
        with TelegramClient('my_session', api_id, api_hash) as telegram_client:
            logger.info("텔레그램 클라이언트 연결 성공")

            for channel_idx, username in enumerate(channel_usernames):
                logger.info(f"채널 '{username}' 크롤링 중... ({channel_idx + 1}/{len(channel_usernames)})")
                channel_message_count = 0

                try:
                    entity = telegram_client.get_entity(username)
                    logger.debug(f"채널 '{username}' 접근 성공 - ID: {entity.id}")

                    for message in telegram_client.iter_messages(username, limit=limit_per_channel):
                        try:
                            if not message.text:
                                continue

                            raw_text = message.text
                            normalized = normalize_text(raw_text)

                            if len(normalized) < 200:
                                continue

                            if not is_semiconductor_related(normalized, FAST_ANY):
                                continue

                            labels = extract_labels(normalized, RX) or ["Uncategorized"]
                            channel_message_count += 1

                            messages_data.append({
                                'channel': username,
                                'sender_id': str(message.sender_id).replace("-", "") if message.sender_id else "unknown",
                                'date_utc': message.date.astimezone(timezone.utc).isoformat() if message.date else datetime.now(timezone.utc).isoformat(),
                                'date_local': message.date.astimezone(KST).strftime('%Y-%m-%d %H:%M:%S') if message.date else datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S'),
                                'labels': ";".join(sorted(set(labels))),
                                'message': raw_text,
                                'normalized_text': normalized,
                                'message_length': len(raw_text),
                                'forward_count': round((message.forwards / message.views) * 10000) if message.forwards and message.views else 0,
                                'forward_channels': username
                            })

                        except Exception as e:
                            logger.error(f"메시지 처리 중 오류: {e}")
                            continue

                    logger.info(f"채널 '{username}'에서 {channel_message_count}개 메시지 수집")

                except Exception as e:
                    logger.error(f"채널 '{username}' 크롤링 중 오류: {e}")
                    continue

    except Exception as e:
        logger.error(f"텔레그램 클라이언트 연결 실패: {e}")
        return pd.DataFrame()

    logger.info(f"총 {len(messages_data)}개의 메시지 수집 완료")
    return pd.DataFrame(messages_data)

def process_with_gpt(df, openai_client):
    """GPT를 사용하여 메시지 분석"""
    logger.info(f"GPT 분석 시작 - 처리할 메시지: {len(df)}개")

    if df.empty:
        df['summary'] = '데이터 없음'
        df['keywords'] = '데이터 없음'
        df['sentiment'] = '데이터 없음'
        return df

    summaries, keywords_list, sentiments = [], [], []

    for idx, row in df.iterrows():
        if idx % 10 == 0:
            logger.info(f"GPT 분석 진행: {idx + 1}/{len(df)}")

        summary, keywords, sentiment = analyze_with_gpt(row['message'], openai_client)
        summaries.append(summary)
        keywords_list.append(keywords)
        sentiments.append(sentiment)

        time.sleep(0.5)

    df['summary'] = summaries
    df['keywords'] = keywords_list
    df['sentiment'] = sentiments

    logger.info("GPT 분석 완료")
    return df

# ====== 메인 실행 부분 ======
def main():
    logger.info("프로그램 시작")

    # 환경변수 검증
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY가 설정되지 않았습니다.")
        return

    api_id_str = os.getenv("API_ID")
    api_hash = os.getenv("API_HASH")
    if not api_id_str or not api_hash:
        logger.error(f"텔레그램 API 정보가 누락되었습니다. API_ID: {'있음' if api_id_str else '없음'}, API_HASH: {'있음' if api_hash else '없음'}")
        return

    try:
        api_id = int(api_id_str)
    except ValueError:
        logger.error("API_ID를 정수로 변환할 수 없습니다.")
        return

    openai_client = OpenAI(api_key=api_key)
    logger.info(f"현재 작업 디렉토리: {os.getcwd()}")

    # 채널 로드
    channel_usernames = load_channels('channels.txt')
    if not channel_usernames:
        logger.error("크롤링할 채널이 없습니다.")
        return

    csv_filename = "telegram_semiconductor_messages.csv"
    existing_data = load_existing_data(csv_filename)
    new_messages = crawl_telegram_messages(channel_usernames, api_id, api_hash, limit_per_channel=30)

    if not new_messages.empty:
        logger.info(f"새 메시지 수집 성공: {len(new_messages)}개")

        # 이미 처리된 메시지는 GPT 스킵
        if not existing_data.empty and 'normalized_text' in existing_data.columns:
            existing_texts = set(existing_data['normalized_text'].dropna())
            truly_new = new_messages[~new_messages['normalized_text'].isin(existing_texts)].reset_index(drop=True)
            already_exists = new_messages[new_messages['normalized_text'].isin(existing_texts)].reset_index(drop=True)
            logger.info(f"신규: {len(truly_new)}개, 기존(GPT 스킵): {len(already_exists)}개")
        else:
            truly_new = new_messages
            already_exists = pd.DataFrame()

        processed_new = process_with_gpt(truly_new, openai_client)

        # forward_count 업데이트를 위해 기존 메시지도 병합에 포함
        if not already_exists.empty:
            # 기존 데이터에서 summary/keywords/sentiment 가져와 채움
            existing_cols = existing_data[['normalized_text', 'summary', 'keywords', 'sentiment']].drop_duplicates('normalized_text')
            already_exists = already_exists.merge(existing_cols, on='normalized_text', how='left', suffixes=('', '_old'))
            for col in ['summary', 'keywords', 'sentiment']:
                if f'{col}_old' in already_exists.columns:
                    already_exists[col] = already_exists[f'{col}_old']
                    already_exists = already_exists.drop(columns=[f'{col}_old'])
            processed_messages = pd.concat([processed_new, already_exists], ignore_index=True)
        else:
            processed_messages = processed_new

        updated_data = merge_and_remove_duplicates(existing_data, processed_messages)

        if 'date_utc' in updated_data.columns and not updated_data.empty:
            try:
                updated_data['date_utc'] = pd.to_datetime(updated_data['date_utc'])
                updated_data = updated_data.sort_values('date_utc', ascending=False)
                updated_data['date_utc'] = updated_data['date_utc'].dt.strftime('%Y-%m-%dT%H:%M:%S+00:00')
            except Exception as e:
                logger.error(f"날짜 정렬 중 오류: {e}")

        save_updated_data(updated_data, csv_filename)
        logger.info(f"=== 크롤링 완료 === 새로 수집: {len(new_messages)}개 / 최종 총계: {len(updated_data)}개")

        if 'labels' in updated_data.columns:
            all_labels = [
                label
                for labels_str in updated_data['labels'].dropna()
                for label in labels_str.split(';')
            ]
            if all_labels:
                logger.info("=== 라벨별 통계 ===")
                for label, count in Counter(all_labels).most_common():
                    logger.info(f"  {label}: {count}개")

    else:
        logger.info("새로 수집된 메시지가 없습니다.")
        save_updated_data(pd.DataFrame(columns=CSV_COLUMNS), csv_filename)

    logger.info("프로그램 종료")


if __name__ == "__main__":
    main()
