


import os
import csv
import re
import time
import pandas as pd
from datetime import datetime, timezone
from telethon.sync import TelegramClient
from openai import OpenAI

# ====== OpenAI 설정 ======
api_key = os.getenv(OPENAI_API_KEY)
client = OpenAI(api_key=api_key)

# ====== 텔레그램 설정 ======
api_id = int(os.getenv("API_ID"))
api_hash = os.getenv("API_HASH")

channel_usernames = [
    'merITz_Tech', 'beluga_investment', 'cahier_de_market', 'aetherjapanresearch', 'kimcharger',
    'KISemicon', 'anakinvest', 'toberichplz', 'Ryu일무이', 'jake8lee', 'bornlupin', 'GlobalTechMoon'
]

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
    r'EUV', r'DUV', r'포토|리소그래피|lithography', r'식각|에칭|etch', r'증착|CVD|ALD',
    r'GAA|FinFET', r'공정|nm\b|나노미터', r'노광', r'수율|yield', r'웨이퍼|wafer', r'칩|die\b',
    r'3nm', r'5nm', r'7nm', r'14nm'
]

# ====== 파일 관리 함수들 ======
def load_existing_data(file_name):
    """기존 CSV 파일 로드"""
    if os.path.exists(file_name):
        existing_df = pd.read_csv(file_name, encoding='utf-8-sig')
        print(f"기존 파일 로드 완료: {file_name}, 데이터 수: {len(existing_df)}개")
        return existing_df
    else:
        print("기존 파일이 없습니다. 새로운 데이터를 생성합니다.")
        return pd.DataFrame()

def merge_and_remove_duplicates(existing_df, new_df):
    """기존 데이터와 새 데이터 병합 및 중복 제거"""
    if not existing_df.empty and not new_df.empty:
        # 'normalized_text' 컬럼을 기준으로 중복 제거
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset='normalized_text', keep='first').reset_index(drop=True)
        print(f"기존 데이터와 새 데이터를 병합했습니다. 최종 데이터 개수: {len(combined_df)}")
    elif not new_df.empty:
        combined_df = new_df
        print(f"새 데이터만 사용합니다. 데이터 개수: {len(combined_df)}")
    else:
        combined_df = existing_df
        print("새 데이터가 없습니다.")
    return combined_df

def save_updated_data(data, file_name):
    """데이터를 CSV 파일로 저장"""
    data.to_csv(file_name, encoding='utf-8-sig', index=False)
    print(f"업데이트된 데이터를 저장했습니다: {file_name}")

# ====== 텍스트 처리 함수들 ======
def compile_any(patterns):
    """정규식 패턴들을 하나로 컴파일"""
    return re.compile(r'(' + '|'.join(patterns) + r')', re.IGNORECASE)

def normalize_text(s: str) -> str:
    """텍스트 정규화 (공백 정리)"""
    return re.sub(r'\s+', ' ', s).strip()

def extract_labels(text: str, rx_dict):
    """텍스트에서 카테고리 라벨 추출"""
    labels = []
    for k, rx in rx_dict.items():
        if rx.search(text):
            labels.append(k)
    return labels

def is_semiconductor_related(text: str, fast_rx) -> bool:
    """반도체 관련 텍스트인지 빠른 필터링"""
    return bool(fast_rx.search(text))

# ====== GPT 분석 함수들 ======
def analyze_with_gpt(text: str):
    """GPT를 사용하여 텍스트 분석 (요약 및 키워드 추출)"""
    try:
        prompt = f"""
        다음 텍스트를 분석해서 200자 이내로 요약하고, 핵심 키워드 5개를 추출해주세요.

        텍스트:
        {text}

        응답 형식:
        요약: [요약 내용]
        키워드: [키워드1, 키워드2, 키워드3, 키워드4, 키워드5]
        """

        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": "당신은 반도체 및 기술 분야 텍스트를 분석하는 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3
        )
        
        result = response.choices[0].message.content.strip()
        
        # 요약과 키워드 분리
        lines = result.split('\n')
        summary = ""
        keywords = ""
        
        for line in lines:
            if line.strip().startswith("요약:"):
                summary = line.replace("요약:", "").strip()
            elif line.strip().startswith("키워드:"):
                keywords = line.replace("키워드:", "").strip()
        
        return summary, keywords
        
    except Exception as e:
        print(f"GPT 분석 중 오류 발생: {e}")
        return "분석 실패", "분석 실패"

def sentiment_analysis(text: str):
    """GPT를 사용한 감정 분석"""
    try:
        prompt = f"""
        다음 텍스트의 감정을 분석해주세요. 긍정적, 부정적, 중립적 중 하나로 답변해주세요.

        텍스트:
        {text}

        답변: (긍정적/부정적/중립적)
        """

        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": "당신은 텍스트의 감정을 분석하는 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0.1
        )
        
        sentiment = response.choices[0].message.content.strip()
        return sentiment
        
    except Exception as e:
        print(f"감정 분석 중 오류 발생: {e}")
        return "분석 실패"

# ====== 메인 크롤링 함수 ======
def crawl_telegram_messages(limit_per_channel=50):
    """텔레그램 메시지 크롤링"""
    
    # 정규식 컴파일
    RX = {
        'Company': compile_any(COMPANY),
        'Memory': compile_any(MEMORY),
        'Product': compile_any(PRODUCT),
        'Tech': compile_any(TECH)
    }
    
    # 빠른 필터용 정규식
    FAST_ANY = compile_any(COMPANY + MEMORY + PRODUCT + TECH)
    
    messages_data = []
    unique_texts = set()
    
    print("텔레그램 메시지 크롤링 시작...")
    
    with TelegramClient('my_session', api_id, api_hash) as client:
        for username in channel_usernames:
            print(f"채널 '{username}' 크롤링 중...")
            
            try:
                for message in client.iter_messages(username, limit=limit_per_channel):
                    if not message.text:
                        continue
                    
                    raw_text = message.text
                    normalized = normalize_text(raw_text)
                    
                    # 길이 필터
                    if len(normalized) < 50:
                        continue
                    
                    # 중복 확인
                    if normalized in unique_texts:
                        continue
                    
                    # 반도체 관련 키워드 필터링
                    if not is_semiconductor_related(normalized, FAST_ANY):
                        continue
                    
                    # 라벨 추출
                    labels = extract_labels(normalized, RX)
                    if not labels:
                        labels = ["Uncategorized"]
                    
                    unique_texts.add(normalized)
                    
                    # 메시지 데이터 구성
                    message_data = {
                        'channel': username,
                        'sender_id': str(message.sender_id).replace("-", ""),
                        'date_utc': message.date.astimezone(timezone.utc).isoformat(),
                        'date_local': message.date.astimezone().strftime('%Y-%m-%d %H:%M:%S'),
                        'labels': ";".join(sorted(set(labels))),
                        'message': raw_text,
                        'normalized_text': normalized,
                        'message_length': len(raw_text)
                    }
                    
                    messages_data.append(message_data)
                    
                print(f"채널 '{username}'에서 {len([m for m in messages_data if m['channel'] == username])}개 메시지 수집")
                
            except Exception as e:
                print(f"채널 '{username}' 크롤링 중 오류 발생: {e}")
                continue
    
    print(f"총 {len(messages_data)}개의 반도체 관련 메시지 수집 완료")
    return pd.DataFrame(messages_data)

def process_with_gpt(df, process_gpt=True):
    """GPT를 사용하여 메시지 분석 (선택적)"""
    if not process_gpt or df.empty:
        df['summary'] = '분석 안함'
        df['keywords'] = '분석 안함'
        df['sentiment'] = '분석 안함'
        return df
    
    print("GPT를 사용하여 메시지 분석 중...")
    
    summaries = []
    keywords_list = []
    sentiments = []
    
    for idx, row in df.iterrows():
        print(f"GPT 분석 진행: {idx + 1}/{len(df)}")
        
        # GPT 분석
        summary, keywords = analyze_with_gpt(row['message'])
        sentiment = sentiment_analysis(row['message'])
        
        summaries.append(summary)
        keywords_list.append(keywords)
        sentiments.append(sentiment)
        
        # API 호출 제한을 위한 대기
        time.sleep(0.5)
    
    df['summary'] = summaries
    df['keywords'] = keywords_list
    df['sentiment'] = sentiments
    
    return df

# ====== 메인 실행 부분 ======
if __name__ == "__main__":
    # 파일명 설정
    csv_filename = "telegram_semiconductor_messages.csv"
    
    # 기존 데이터 로드
    existing_data = load_existing_data(csv_filename)
    
    # 새 메시지 크롤링
    new_messages = crawl_telegram_messages(limit_per_channel=50)
    
    if not new_messages.empty:
        # GPT 분석 여부 선택 (True: GPT 분석 실행, False: 분석 건너뛰기)
        USE_GPT_ANALYSIS = False  # 비용을 고려해 기본값은 False
        
        # GPT 분석 (선택적)
        processed_messages = process_with_gpt(new_messages, USE_GPT_ANALYSIS)
        
        # 기존 데이터와 병합
        updated_data = merge_and_remove_duplicates(existing_data, processed_messages)
        
        # 날짜순 정렬
        if 'date_utc' in updated_data.columns:
            updated_data['date_utc'] = pd.to_datetime(updated_data['date_utc'])
            updated_data = updated_data.sort_values('date_utc', ascending=False)
            updated_data['date_utc'] = updated_data['date_utc'].dt.strftime('%Y-%m-%dT%H:%M:%S+00:00')
        
        # 저장
        save_updated_data(updated_data, csv_filename)
        
        print(f"\n=== 크롤링 완료 ===")
        print(f"새로 수집한 메시지: {len(new_messages)}개")
        print(f"최종 데이터 총 개수: {len(updated_data)}개")
        print(f"저장 파일: {csv_filename}")
        
        # 라벨별 통계
        if not updated_data.empty and 'labels' in updated_data.columns:
            print("\n=== 라벨별 메시지 통계 ===")
            all_labels = []
            for labels_str in updated_data['labels']:
                if pd.notna(labels_str):
                    all_labels.extend(labels_str.split(';'))
            
            from collections import Counter
            label_counts = Counter(all_labels)
            for label, count in label_counts.most_common():
                print(f"{label}: {count}개")
    
    else:
        print("새로 수집된 메시지가 없습니다.")
