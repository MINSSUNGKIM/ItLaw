# 형법 RAG 시스템 🏛️

AI 기반 형법 정보 검색 및 질의응답 시스템입니다. 한국 형법 조문을 벡터화하여 의미론적 검색과 GPT 기반 자연어 답변을 제공합니다.

## 📁 파일 구조

```
project/
├── api_server.py                 # Flask 백엔드 서버
├── korean_optimized_legal.json   # 벡터화된 형법 데이터
├── law.docx                      # 원본 형법 문서
├── preprocess_vectorize.py       # 데이터 전처리 및 벡터화 스크립트
├── retriever.py                  # 검색 엔진 모듈
├── web_interface.html            # 웹 인터페이스
├── logo.svg                      # 시스템 로고
└── LICENSE                       # 라이선스
```

## 🚀 빠른 시작

### 1. 환경 설정

**Python 3.8+ 필요**

```bash
# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 필수 패키지 설치
pip install flask flask-cors numpy sentence-transformers scikit-learn python-docx
```

**선택사항: OpenAI GPT 사용**
```bash
pip install openai
export OPENAI_API_KEY="your-openai-api-key"  # Windows: set OPENAI_API_KEY=your-key
```

### 2. 데이터 준비 (이미 완료된 경우 건너뛰기)

```bash
# 형법 문서 전처리 및 벡터화
python preprocess_vectorize.py
```

### 3. 서버 실행

```bash
# 백엔드 서버 시작
python api_server.py
```

서버가 성공적으로 시작되면 다음과 같은 메시지가 표시됩니다:
```
형법 RAG 시스템 시작
==================================================
OpenAI API 키 감지됨 - GPT 답변 생성 활성화  # 또는 템플릿 기반 답변 사용
데이터 파일 로드 성공: korean_optimized_legal.json
로드된 문서: 1234개

서버 시작: http://localhost:5002
```

### 4. 웹 인터페이스 접속

브라우저에서 `web_interface.html` 파일을 열거나, 웹서버에서 호스팅하세요.

## 💡 사용 방법

### 웹 인터페이스

1. **문서 검색**: 키워드로 관련 형법 조문 검색
   - 예: "살인죄", "절도", "사기", "업무상횡령"

2. **AI 질문**: 자연어로 형법 관련 질문
   - 예: "살인죄의 구성요건은 무엇인가요?"
   - 예: "절도와 강도의 차이점은?"

3. **통계**: 시스템 상태 및 데이터 현황 확인

### API 직접 사용

#### 문서 검색
```bash
curl -X POST http://localhost:5002/search \
  -H "Content-Type: application/json" \
  -d '{"query": "살인죄", "top_k": 5}'
```

#### AI 질문
```bash
curl -X POST http://localhost:5002/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "살인죄의 구성요건은 무엇인가요?"}'
```

#### 서버 상태 확인
```bash
curl http://localhost:5002/health
```

#### 통계 조회
```bash
curl http://localhost:5002/stats
```

## ⚙️ 시스템 구성

### 백엔드 (api_server.py)
- **Flask** 기반 REST API 서버
- **포트**: 5002
- **CORS** 활성화로 웹 인터페이스 지원

### 벡터 검색 엔진
- **SentenceTransformer**: 한국어 텍스트 임베딩
- **코사인 유사도**: 의미론적 검색
- **모델**: `jhgan/ko-sroberta-multitask` (768차원)

### AI 답변 생성
- **GPT-3.5-turbo**: OpenAI API 사용 (선택)
- **템플릿 기반**: API 키 없을 때 폴백

## 🔧 설정 옵션

### OpenAI 설정
```bash
# GPT 기반 답변 활성화
export OPENAI_API_KEY="sk-..."

# 템플릿 기반 답변만 사용
unset OPENAI_API_KEY
```

### 데이터 파일 우선순위
시스템은 다음 순서로 데이터 파일을 찾습니다:
1. `korean_optimized_legal.json`
2. `criminal_law_optimized.json`
3. `legal_documents.json`

### 포트 변경
`api_server.py`의 마지막 줄에서 포트 수정:
```python
app.run(host='0.0.0.0', port=5002, debug=False)  # 포트 변경
```

## 🐛 문제 해결

### 서버 시작 오류

**"데이터 파일을 찾을 수 없습니다"**
```bash
# 데이터 재생성
python preprocess_vectorize.py
```

**포트 이미 사용 중**
```bash
# 프로세스 확인 및 종료
lsof -i :5002  # macOS/Linux
netstat -ano | findstr :5002  # Windows

# 다른 포트 사용
python api_server.py  # 코드에서 포트 변경 후
```

### 웹 인터페이스 연결 오류

**CORS 오류**
- 백엔드 서버가 실행 중인지 확인
- 브라우저 콘솔에서 네트워크 오류 확인

**서버 연결 실패**
- `http://localhost:5002/health` 직접 접속하여 서버 상태 확인

### OpenAI API 오류

**API 키 오류**
```bash
# API 키 재설정
export OPENAI_API_KEY="올바른-키"
python api_server.py
```

**요청 한도 초과**
- 템플릿 기반 모드로 자동 전환됨
- API 키 제거 시 완전히 템플릿 모드로 작동

## 📊 성능 정보

- **문서 수**: ~1,200개 형법 조문
- **검색 속도**: ~100ms
- **벡터 차원**: 768
- **메모리 사용량**: ~2GB (모델 로딩 포함)
