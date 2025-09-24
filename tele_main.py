import os
import csv
import re
import time
import pandas as pd
from datetime import datetime, timezone
from telethon.sync import TelegramClient
from openai import OpenAI

# ====== 디버깅을 위한 로깅 함수 추가 ======
def log_debug(message):
    """디버깅 메시지 출력"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] DEBUG: {message}")

# ====== OpenAI 설정 ======
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    log_debug("OPENAI_API_KEY가 설정되지 않았습니다.")
    exit(1)
else:
    log_debug("OPENAI_API_KEY 로드 완료")

client = OpenAI(api_key=api_key)

# ====== 텔레그램 설정 ======
api_id = os.getenv("API_ID")
api_hash = os.getenv("API_HASH")

if not api_id or not api_hash:
    log_debug("텔레그램 API 정보가 누락되었습니다.")
    log_debug(f"API_ID: {'있음' if api_id else '없음'}")
    log_debug(f"API_HASH: {'있음' if api_hash else '없음'}")
    exit(1)

try:
    api_id = int(api_id)
    log_debug(f"텔레그램 API 설정 완료 - API_ID: {api_id}")
except ValueError:
    log_debug("API_ID를 정수로 변환할 수 없습니다.")
    exit(1)

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
    log_debug(f"기존 데이터 로드 시도: {file_name}")
    if os.path.exists(file_name):
        try:
            existing_df = pd.read_csv(file_name, encoding='utf-8-sig')
            log_debug(f"기존 파일 로드 완료: {file_name}, 데이터 수: {len(existing_df)}개")
            return existing_df
        except Exception as e:
            log_debug(f"기존 파일 로드 실패: {e}")
            return pd.DataFrame()
    else:
        log_debug("기존 파일이 없습니다. 새로운 데이터를 생성합니다.")
        return pd.DataFrame()

def merge_and_remove_duplicates(existing_df, new_df):
    """기존 데이터와 새 데이터 병합 및 중복 제거"""
    log_debug(f"데이터 병합 시작 - 기존: {len(existing_df)}개, 새로운: {len(new_df)}개")
    
    if not existing_df.empty and not new_df.empty:
        # 'normalized_text' 컬럼을 기준으로 중복 제거
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        before_dedup = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset='normalized_text', keep='first').reset_index(drop=True)
        after_dedup = len(combined_df)
        log_debug(f"중복 제거 완료: {before_dedup}개 → {after_dedup}개 (제거: {before_dedup - after_dedup}개)")
    elif not new_df.empty:
        combined_df = new_df
        log_debug(f"새 데이터만 사용합니다. 데이터 개수: {len(combined_df)}")
    else:
        combined_df = existing_df
        log_debug("새 데이터가 없습니다.")
    return combined_df

def save_updated_data(data, file_name):
    """데이터를 CSV 파일로 저장"""
    try:
        log_debug(f"데이터 저장 시도: {file_name} ({len(data)}개 행)")
        
        # 빈 데이터프레임이라도 저장
        if data.empty:
            log_debug("빈 데이터프레임을 저장합니다.")
            # 기본 컬럼 구조로 빈 파일 생성
            empty_df = pd.DataFrame(columns=[
                'channel', 'sender_id', 'date_utc', 'date_local', 'labels', 
                'message', 'normalized_text', 'message_length', 'summary', 'keywords', 'sentiment'
            ])
            empty_df.to_csv(file_name, encoding='utf-8-sig', index=False)
            log_debug(f"빈 CSV 파일이 생성되었습니다: {file_name}")
        else:
            data.to_csv(file_name, encoding='utf-8-sig', index=False)
            log_debug(f"업데이트된 데이터를 저장했습니다: {file_name}")
        
        # 파일 생성 확인
        if os.path.exists(file_name):
            file_size = os.path.getsize(file_name)
            log_debug(f"파일 생성 확인됨 - 크기: {file_size} bytes")
        else:
            log_debug("경고: 파일이 생성되지 않았습니다!")
            
    except Exception as e:
        log_debug(f"파일 저장 중 오류 발생: {e}")
        # 강제로라도 파일 생성 시도
        try:
            with open(file_name, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['channel', 'sender_id', 'date_utc', 'date_local', 'labels', 
                               'message', 'normalized_text', 'message_length', 'summary', 'keywords', 'sentiment'])
                if not data.empty:
                    for _, row in data.iterrows():
                        writer.writerow(row.values)
            log_debug(f"CSV 모듈로 파일 생성 완료: {file_name}")
        except Exception as e2:
            log_debug(f"CSV 모듈로도 파일 생성 실패: {e2}")

# ====== 텍스트 처리 함수들 ======
def compile_any(patterns):
    """정규식 패턴들을 하나로 컴파일"""
    try:
        compiled = re.compile(r'(' + '|'.join(patterns) + r')', re.IGNORECASE)
        log_debug(f"정규식 컴파일 완료: {len(patterns)}개 패턴")
        return compiled
    except Exception as e:
        log_debug(f"정규식 컴파일 오류: {e}")
        return re.compile(r'반도체|semiconductor', re.IGNORECASE)  # 기본 패턴

def normalize_text(s: str) -> str:
    """텍스트 정규화 (공백 정리)"""
    return re.sub(r'\s+', ' ', s).strip()

def extract_labels(text: str, rx_dict):
    """텍스트에서 카테고리 라벨 추출"""
    labels = []
    for k, rx in rx_dict.items():
        try:
            if rx.search(text):
                labels.append(k)
        except Exception as e:
            log_debug(f"라벨 추출 중 오류 ({k}): {e}")
            continue
    return labels

def is_semiconductor_related(text: str, fast_rx) -> bool:
    """반도체 관련 텍스트인지 빠른 필터링"""
    try:
        return bool(fast_rx.search(text))
    except Exception as e:
        log_debug(f"키워드 필터링 중 오류: {e}")
        return True  # 오류 시 일단 통과

# ====== GPT 분석 함수들 ======
def analyze_with_gpt(text: str):
    """GPT를 사용하여 텍스트 분석 (요약 및 키워드 추출)"""
    try:
        prompt = f"""다음 텍스트를 분석해서 요약과 키워드를 추출해주세요.

요약 조건:
- 200자 이내로 작성
- 1. 2. 3. 4. 5. 식으로 논리적 순서로 작성
- ~했음, ~함, ~였음 식으로 간결체 사용

키워드 조건:
- 핵심 키워드 5개 추출
- 쉼표로 구분

텍스트: {text}

아래 형식으로만 답변해주세요:
요약: [여기에 요약 내용]
키워드: [키워드1, 키워드2, 키워드3, 키워드4, 키워드5]"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 반도체 및 기술 분야 텍스트를 분석하는 전문가입니다. 정확히 요청된 형식으로만 답변하세요."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=600,
            temperature=0.2
        )
        
        result = response.choices[0].message.content.strip()
        log_debug(f"GPT 원본 응답: {result}")
        
        # 더 유연한 파싱 로직
        summary = ""
        keywords = ""
        
        # 각 줄을 확인하며 파싱
        lines = result.split('\n')
        for line in lines:
            line_clean = line.strip()
            
            # 요약 찾기 (다양한 형태 대응)
            if any(starter in line_clean.lower() for starter in ['요약:', '요약 :', '요약-', '요약 -']):
                # '요약:' 이후 내용 추출
                for separator in ['요약:', '요약 :', '요약-', '요약 -']:
                    if separator in line_clean:
                        summary = line_clean.split(separator, 1)[1].strip()
                        break
                
                # 여러 줄에 걸친 요약 처리
                if not summary and line_clean.endswith(':'):
                    # 다음 줄들을 요약으로 수집
                    line_idx = lines.index(line)
                    for next_line in lines[line_idx + 1:]:
                        next_clean = next_line.strip()
                        if next_clean and not any(kw in next_clean.lower() for kw in ['키워드:', '키워드 :']):
                            if summary:
                                summary += " " + next_clean
                            else:
                                summary = next_clean
                        elif any(kw in next_clean.lower() for kw in ['키워드:', '키워드 :']):
                            break
            
            # 키워드 찾기 (다양한 형태 대응)
            if any(starter in line_clean.lower() for starter in ['키워드:', '키워드 :', '키워드-', '키워드 -']):
                # '키워드:' 이후 내용 추출
                for separator in ['키워드:', '키워드 :', '키워드-', '키워드 -']:
                    if separator in line_clean:
                        keywords = line_clean.split(separator, 1)[1].strip()
                        break
                
                # 여러 줄에 걸친 키워드 처리
                if not keywords and line_clean.endswith(':'):
                    # 다음 줄들을 키워드로 수집
                    line_idx = lines.index(line)
                    for next_line in lines[line_idx + 1:]:
                        next_clean = next_line.strip()
                        if next_clean and not any(summ in next_clean.lower() for summ in ['요약:', '요약 :']):
                            if keywords:
                                keywords += " " + next_clean
                            else:
                                keywords = next_clean
                        elif any(summ in next_clean.lower() for summ in ['요약:', '요약 :']):
                            break
        
        # 결과가 비어있을 경우 전체 텍스트에서 다시 시도
        if not summary or not keywords:
            log_debug("표준 파싱 실패, 전체 텍스트 재분석 시도")
            
            # 요약 부분 찾기
            if not summary:
                summary_match = re.search(r'요약[:\s\-]*(.+?)(?=키워드|$)', result, re.DOTALL | re.IGNORECASE)
                if summary_match:
                    summary = summary_match.group(1).strip()
            
            # 키워드 부분 찾기
            if not keywords:
                keyword_match = re.search(r'키워드[:\s\-]*(.+)', result, re.DOTALL | re.IGNORECASE)
                if keyword_match:
                    keywords = keyword_match.group(1).strip()
        
        # 최종 정리
        summary = summary.strip() if summary else "요약 파싱 실패"
        keywords = keywords.strip() if keywords else "키워드 파싱 실패"
        
        # 200자 초과시 자르기
        if len(summary) > 200:
            summary = summary[:197] + "..."
        
        log_debug(f"파싱된 요약: {summary}")
        log_debug(f"파싱된 키워드: {keywords}")
        
        return summary, keywords
        
    except Exception as e:
        log_debug(f"GPT 분석 중 오류 발생: {e}")
        return "분석 실패", "분석 실패"


def sentiment_analysis(text: str):
    """GPT를 사용한 감정 분석"""
    try:
        prompt = f"""다음 텍스트의 투자/시장 감정을 분석해주세요.

텍스트: {text}

다음 중 하나로만 답변해주세요:
- 긍정적: 주가 상승, 호재, 성장 등 긍정적 내용
- 부정적: 주가 하락, 악재, 위험 등 부정적 내용  
- 중립적: 단순 정보 전달, 객관적 사실

답변: """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 금융/투자 텍스트의 감정을 분석하는 전문가입니다. 정확히 긍정적, 부정적, 중립적 중 하나로만 답변하세요."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0.1
        )
        
        sentiment = response.choices[0].message.content.strip()
        
        # 감정 분류 정규화
        sentiment_lower = sentiment.lower()
        if any(word in sentiment_lower for word in ['긍정', 'positive', '호재', '상승']):
            return "긍정적"
        elif any(word in sentiment_lower for word in ['부정', 'negative', '악재', '하락']):
            return "부정적"
        elif any(word in sentiment_lower for word in ['중립', 'neutral', '객관']):
            return "중립적"
        else:
            return sentiment  # 원본 반환
        
    except Exception as e:
        log_debug(f"감정 분석 중 오류 발생: {e}")
        return "분석 실패"

# ====== 메인 크롤링 함수 ======
def crawl_telegram_messages(limit_per_channel=5):
    """텔레그램 메시지 크롤링"""
    
    log_debug("텔레그램 메시지 크롤링 시작...")
    
    # 정규식 컴파일
    try:
        RX = {
            'Company': compile_any(COMPANY),
            'Memory': compile_any(MEMORY),
            'Product': compile_any(PRODUCT),
            'Tech': compile_any(TECH)
        }
        
        # 빠른 필터용 정규식
        FAST_ANY = compile_any(COMPANY + MEMORY + PRODUCT + TECH)
        log_debug("모든 정규식 컴파일 완료")
    except Exception as e:
        log_debug(f"정규식 컴파일 실패: {e}")
        return pd.DataFrame()
    
    messages_data = []
    unique_texts = set()
    
    # 세션 파일 확인
    session_file = 'my_session.session'
    if os.path.exists(session_file):
        log_debug(f"세션 파일 발견: {session_file}")
    else:
        log_debug(f"세션 파일 없음: {session_file}")
    
    try:
        with TelegramClient('my_session', api_id, api_hash) as telegram_client:
            log_debug("텔레그램 클라이언트 연결 성공")
            
            for channel_idx, username in enumerate(channel_usernames):
                log_debug(f"채널 '{username}' 크롤링 중... ({channel_idx + 1}/{len(channel_usernames)})")
                
                channel_message_count = 0
                
                try:
                    # 채널 정보 확인
                    entity = telegram_client.get_entity(username)
                    log_debug(f"채널 '{username}' 접근 성공 - ID: {entity.id}")
                    
                    for message in telegram_client.iter_messages(username, limit=limit_per_channel):
                        try:
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
                            channel_message_count += 1
                            
                            # 메시지 데이터 구성
                            message_data = {
                                'channel': username,
                                'sender_id': str(message.sender_id).replace("-", "") if message.sender_id else "unknown",
                                'date_utc': message.date.astimezone(timezone.utc).isoformat() if message.date else datetime.now(timezone.utc).isoformat(),
                                'date_local': message.date.astimezone().strftime('%Y-%m-%d %H:%M:%S') if message.date else datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'labels': ";".join(sorted(set(labels))),
                                'message': raw_text,
                                'normalized_text': normalized,
                                'message_length': len(raw_text)
                            }
                            
                            messages_data.append(message_data)
                            
                        except Exception as e:
                            log_debug(f"메시지 처리 중 오류: {e}")
                            continue
                    
                    log_debug(f"채널 '{username}'에서 {channel_message_count}개 메시지 수집")
                    
                except Exception as e:
                    log_debug(f"채널 '{username}' 크롤링 중 오류 발생: {e}")
                    continue
    
    except Exception as e:
        log_debug(f"텔레그램 클라이언트 연결 실패: {e}")
        log_debug("빈 데이터프레임을 반환합니다.")
        return pd.DataFrame()
    
    log_debug(f"총 {len(messages_data)}개의 메시지 수집 완료")
    return pd.DataFrame(messages_data)

def process_with_gpt(df):
    """GPT를 사용하여 메시지 분석 (무조건 실행)"""
    log_debug(f"GPT 분석 시작 - 처리할 메시지: {len(df)}개")
    
    if df.empty:
        df['summary'] = '데이터 없음'
        df['keywords'] = '데이터 없음'
        df['sentiment'] = '데이터 없음'
        log_debug("빈 데이터프레임 - GPT 분석을 건너뜁니다.")
        return df
    
    summaries = []
    keywords_list = []
    sentiments = []
    
    for idx, row in df.iterrows():
        if idx % 10 == 0:  # 10개마다 진행상황 출력
            log_debug(f"GPT 분석 진행: {idx + 1}/{len(df)}")
        
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
    
    log_debug("GPT 분석 완료")
    return df

# ====== 메인 실행 부분 ======
if __name__ == "__main__":
    log_debug("프로그램 시작")
    
    # 파일명 설정
    csv_filename = "telegram_semiconductor_messages.csv"
    log_debug(f"출력 파일명: {csv_filename}")
    
    # 현재 디렉토리 확인
    current_dir = os.getcwd()
    log_debug(f"현재 작업 디렉토리: {current_dir}")
    
    # 기존 데이터 로드
    existing_data = load_existing_data(csv_filename)
    
    # 새 메시지 크롤링
    new_messages = crawl_telegram_messages(limit_per_channel=5)  # 고정값 5개로 설정
    
    # 크롤링 결과 확인
    if not new_messages.empty:
        log_debug(f"새 메시지 수집 성공: {len(new_messages)}개")
        
        # GPT 분석 실행 (항상 활성화)
        processed_messages = process_with_gpt(new_messages)
        
        # 기존 데이터와 병합
        updated_data = merge_and_remove_duplicates(existing_data, processed_messages)
        
        # 날짜순 정렬
        if 'date_utc' in updated_data.columns and not updated_data.empty:
            try:
                updated_data['date_utc'] = pd.to_datetime(updated_data['date_utc'])
                updated_data = updated_data.sort_values('date_utc', ascending=False)
                updated_data['date_utc'] = updated_data['date_utc'].dt.strftime('%Y-%m-%dT%H:%M:%S+00:00')
                log_debug("날짜순 정렬 완료")
            except Exception as e:
                log_debug(f"날짜 정렬 중 오류: {e}")
        
        # 저장
        save_updated_data(updated_data, csv_filename)
        
        log_debug(f"\n=== 크롤링 완료 ===")
        log_debug(f"새로 수집한 메시지: {len(new_messages)}개")
        log_debug(f"최종 데이터 총 개수: {len(updated_data)}개")
        log_debug(f"저장 파일: {csv_filename}")
        
        # 라벨별 통계
        if not updated_data.empty and 'labels' in updated_data.columns:
            log_debug("\n=== 라벨별 메시지 통계 ===")
            all_labels = []
            for labels_str in updated_data['labels']:
                if pd.notna(labels_str):
                    all_labels.extend(labels_str.split(';'))
            
            if all_labels:
                from collections import Counter
                label_counts = Counter(all_labels)
                for label, count in label_counts.most_common():
                    log_debug(f"{label}: {count}개")
    
    else:
        log_debug("새로 수집된 메시지가 없습니다.")
        # 그래도 빈 파일이라도 생성
        empty_df = pd.DataFrame(columns=[
            'channel', 'sender_id', 'date_utc', 'date_local', 'labels', 
            'message', 'normalized_text', 'message_length', 'summary', 'keywords', 'sentiment'
        ])
        save_updated_data(empty_df, csv_filename)
    
    log_debug("프로그램 종료")
