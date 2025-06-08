from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
from datetime import datetime

# Tokenizers 병렬처리 경고 해결
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# OpenAI 클라이언트 import
try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI 라이브러리가 설치되지 않았습니다. pip install openai로 설치하세요.")


class LegalRAGWithGPT:
    def __init__(self, json_file_path="korean_optimized_legal.json"):
        self.data = self.load_data(json_file_path)
        if self.data:
            vector_dim = len(self.data[0]["vector"])
            model_map = {384: "sentence-transformers/all-MiniLM-L6-v2", 768: "jhgan/ko-sroberta-multitask"}
            self.model = SentenceTransformer(model_map.get(vector_dim, "jhgan/ko-sroberta-multitask"))

        # OpenAI 클라이언트 설정
        self.openai_client = None
        self.use_openai = False

        if OPENAI_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                try:
                    self.openai_client = OpenAI(api_key=api_key)
                    self.use_openai = True
                    print("OpenAI 클라이언트 초기화 성공")
                except Exception as e:
                    print(f"OpenAI 클라이언트 초기화 실패: {e}")
            else:
                print("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")

    def load_data(self, file_path):
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def search(self, query, top_k=5):
        if not self.data or not query:
            return []

        query_vector = self.model.encode([query], convert_to_tensor=False)[0]
        results = []

        for item in self.data:
            stored_vector = np.array(item["vector"])
            similarity = cosine_similarity([query_vector], [stored_vector])[0][0]

            results.append({
                "id": item["id"],
                "text": item["text"],
                "similarity": float(similarity),
                "metadata": item.get("metadata", {})
            })

        return sorted(results, key=lambda x: x["similarity"], reverse=True)[:top_k]

    def generate_gpt_answer(self, query, docs):
        """OpenAI GPT를 사용한 답변 생성"""
        if not docs:
            return {
                "answer": "관련 형법 정보를 찾을 수 없습니다.",
                "confidence": 0.0,
                "sources": [],
                "method": "no_context"
            }

        # 상위 3개 문서의 컨텍스트 구성
        context_parts = []
        sources = []

        for i, doc in enumerate(docs[:3], 1):
            if doc["similarity"] > 0.3:
                text = doc["text"]
                context_text = text
                if len(context_text) > 400:
                    context_text = context_text[:400] + "..."

                context_parts.append(f"[법조문 {i}] {context_text}")

                # sources에는 전체 텍스트 포함
                sources.append({
                    "id": doc["id"],
                    "text": doc["text"],
                    "similarity": round(doc["similarity"], 4),
                    "metadata": doc["metadata"]
                })

        if not context_parts:
            return {
                "answer": "관련성이 높은 형법 정보를 찾을 수 없습니다.",
                "confidence": 0.0,
                "sources": [],
                "method": "low_similarity"
            }

        context = "\n\n".join(context_parts)

        # GPT 프롬프트 구성
        prompt = f"""다음 형법 조문을 바탕으로 질문에 답하세요.

질문: {query}

관련 형법 조문:
{context}

답변 가이드라인:
1. 제공된 형법 조문을 바탕으로 정확하게 답변하세요
2. 형법 조문의 구체적인 내용을 인용하여 설명하세요
3. 구성요건, 처벌 조항이 있다면 명시하세요
4. 간결하고 이해하기 쉽게 작성하세요
5. 답변 마지막에 "이 정보는 참고용이며, 구체적인 사안은 형사법 전문가와 상담하시기 바랍니다."를 추가하세요

답변:"""

        try:
            if self.use_openai and self.openai_client:
                # OpenAI API 호출
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "당신은 한국 형법 전문 AI입니다. 정확하고 신뢰할 수 있는 형법 정보를 제공합니다."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=800,
                    temperature=0.3
                )

                answer = response.choices[0].message.content.strip()
                confidence = min(docs[0]["similarity"] * 1.2, 1.0)
                method = "gpt_generated"

            else:
                # OpenAI가 비활성화된 경우 템플릿 기반 답변
                answer = self.generate_template_answer(query, context_parts)
                confidence = docs[0]["similarity"]
                method = "template_fallback"

            return {
                "answer": answer,
                "confidence": round(confidence, 4),
                "sources": sources,
                "method": method
            }

        except Exception as e:
            # OpenAI API 오류 시 템플릿 기반으로 폴백
            print(f"OpenAI API 오류: {e}")
            answer = self.generate_template_answer(query, context_parts)
            return {
                "answer": answer,
                "confidence": round(docs[0]["similarity"], 4),
                "sources": sources,
                "method": f"fallback_error: {str(e)[:50]}"
            }

    def generate_template_answer(self, query, context_parts):
        """템플릿 기반 답변 생성 (폴백용)"""
        answer = f"'{query}'에 대한 관련 형법 정보:\n\n"

        for i, context in enumerate(context_parts, 1):
            clean_context = context.replace(f"[법조문 {i}] ", "")
            answer += f"관련 조문 {i}:\n{clean_context}\n\n"

        # 키워드 기반 가이드
        keywords_guidance = {
            "살인": "살인죄 관련 사건은 즉시 수사기관에 신고하시기 바랍니다.",
            "절도": "절도 피해 시 증거수집 후 경찰서에 신고하시기 바랍니다.",
            "사기": "사기 피해 시 관련 증거를 수집하여 경찰서에 신고하시기 바랍니다.",
            "횡령": "횡령 사건은 증거자료 확보 후 수사기관에 신고하시기 바랍니다.",
            "폭행": "폭행 피해 시 의료진단서 등 증거수집 후 신고하시기 바랍니다.",
        }

        for keyword, guidance in keywords_guidance.items():
            if keyword in query:
                answer += f"추가 안내: {guidance}\n\n"
                break

        answer += "이 정보는 참고용이며, 구체적인 사안은 형사법 전문가와 상담하시기 바랍니다."

        return answer


# Flask 앱 설정
app = Flask(__name__)
CORS(app)
rag_system = None


def init_system():
    global rag_system
    possible_files = [
        "korean_optimized_legal.json",
        "criminal_law_optimized.json",
        "legal_documents.json"
    ]

    for file_path in possible_files:
        if os.path.exists(file_path):
            rag_system = LegalRAGWithGPT(file_path)
            print(f"데이터 파일 로드 성공: {file_path}")
            return True
    return False


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "documents": len(rag_system.data) if rag_system else 0,
        "openai_enabled": rag_system.use_openai if rag_system else False
    })


@app.route('/search', methods=['POST'])
def search():
    if not rag_system:
        return jsonify({"error": "시스템 초기화 오류"}), 500

    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "query 필드 필요"}), 400

    query = data['query'].strip()
    if not query:
        return jsonify({"error": "질문을 입력하세요"}), 400

    top_k = data.get('top_k', 5)
    results = rag_system.search(query, top_k)

    return jsonify({
        "query": query,
        "results": results,
        "total": len(results),
        "timestamp": datetime.now().isoformat()
    })


@app.route('/ask', methods=['POST'])
def ask():
    if not rag_system:
        return jsonify({"error": "시스템 초기화 오류"}), 500

    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "question 필드 필요"}), 400

    question = data['question'].strip()
    if not question:
        return jsonify({"error": "질문을 입력하세요"}), 400

    # 검색 후 GPT로 답변 생성
    docs = rag_system.search(question, 5)
    answer_data = rag_system.generate_gpt_answer(question, docs)

    return jsonify({
        "question": question,
        "answer": answer_data["answer"],
        "confidence": answer_data["confidence"],
        "sources": answer_data["sources"],
        "method": answer_data["method"],
        "timestamp": datetime.now().isoformat()
    })


@app.route('/stats', methods=['GET'])
def stats():
    if not rag_system or not rag_system.data:
        return jsonify({"error": "데이터 없음"}), 404

    data = rag_system.data
    lengths = [len(item["text"]) for item in data]

    return jsonify({
        "total_documents": len(data),
        "avg_length": round(np.mean(lengths), 1),
        "max_length": max(lengths),
        "min_length": min(lengths),
        "vector_dimension": len(data[0]["vector"]) if data else 0,
        "openai_enabled": rag_system.use_openai
    })


def main():
    print("형법 RAG 시스템 시작")
    print("=" * 50)

    # OpenAI API 키 확인
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print("OpenAI API 키 감지됨 - GPT 답변 생성 활성화")
    else:
        print("OpenAI API 키 없음 - 템플릿 기반 답변 사용")
        print("GPT 사용하려면: export OPENAI_API_KEY='your-api-key'")

    if not init_system():
        print("데이터 파일을 찾을 수 없습니다.")
        print("korean_optimized_processor.py를 먼저 실행하세요.")
        return

    print(f"로드된 문서: {len(rag_system.data)}개")
    print("\n서버 시작: http://localhost:5002")
    print("엔드포인트:")
    print("  GET  /health  - 상태 확인")
    print("  POST /search  - 형법 조문 검색")
    print("  POST /ask     - 형법 질문 답변")
    print("  GET  /stats   - 통계")
    print()

    app.run(host='0.0.0.0', port=5002, debug=False)


if __name__ == "__main__":
    main()