import os
import json
import re
from docx import Document
from sentence_transformers import SentenceTransformer
import numpy as np

class LegalDocumentProcessor:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """법률 문서 처리 클래스 초기화"""
        print("SentenceTransformer 모델 로딩 중...")
        self.model = SentenceTransformer(model_name)
        print("모델 로딩 완료!")
    
    def extract_text_from_docx(self, file_path):
        """DOCX 파일에서 텍스트 추출"""
        try:
            document = Document(file_path)
            text = ""
            for paragraph in document.paragraphs:
                if paragraph.text.strip():
                    text += paragraph.text.strip() + "\n"
            return text
        except Exception as e:
            print(f"파일 읽기 오류: {e}")
            return None
    
    def preprocess_legal_text(self, text):
        """법률 텍스트를 조문 단위로 청킹"""
        # "제N조" 패턴을 기준으로 분리
        chunks = re.split(r'(제\d+조(?:\([^)]*\))?)', text)
        
        result_chunks = []
        current_chunk = ""
        
        for i, part in enumerate(chunks):
            # "제N조" 패턴과 매칭되는 경우
            if re.match(r'제\d+조(?:\([^)]*\))?', part):
                # 이전 청크가 있으면 저장
                if current_chunk.strip():
                    result_chunks.append(current_chunk.strip())
                current_chunk = part  # 새 조문 시작
            else:
                current_chunk += " " + part
        
        # 마지막 청크 저장
        if current_chunk.strip():
            result_chunks.append(current_chunk.strip())
        
        # 텍스트 정제
        cleaned_chunks = []
        for chunk in result_chunks:
            # 불필요한 기호 제거
            chunk = re.sub(r'\[[^\]]*\]', '', chunk)  # [내용] 제거
            chunk = re.sub(r'\<[^>]*\>', '', chunk)   # <내용> 제거
            chunk = re.sub(r'제\d+장[^\n]*', '', chunk)  # 장 제목 제거
            chunk = re.sub(r'제\d+절[^\n]*', '', chunk)  # 절 제목 제거
            
            # 공백 정리
            chunk = re.sub(r'\s+', ' ', chunk).strip()
            
            # 최소 길이 확인 (너무 짧은 청크 제거)
            if len(chunk) > 10:
                cleaned_chunks.append(chunk)
        
        return cleaned_chunks
    
    def create_embeddings(self, chunks):
        """텍스트 청크들을 벡터로 변환"""
        print(f"{len(chunks)}개 청크 벡터화 중...")
        embeddings = self.model.encode(chunks, convert_to_tensor=False, show_progress_bar=True)
        return embeddings
    
    def save_to_json(self, chunks, embeddings, output_path):
        """청크와 벡터를 JSON 파일로 저장"""
        if len(chunks) != len(embeddings):
            raise ValueError("청크 수와 임베딩 수가 일치하지 않습니다!")
        
        # 데이터 구조 생성
        data = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            data.append({
                "id": i,
                "text": chunk,
                "vector": embedding.tolist(),
                "metadata": {
                    "length": len(chunk),
                    "article_pattern": bool(re.search(r'제\d+조', chunk))
                }
            })
        
        # JSON 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ {len(data)}개 청크가 {output_path}에 저장되었습니다.")
        return len(data)
    
    def process_document(self, input_file, output_file):
        """전체 처리 파이프라인 실행"""
        print("=" * 50)
        print("법률 문서 RAG 데이터 전처리 시작")
        print("=" * 50)
        
        # 1. 텍스트 추출
        print("1. 문서에서 텍스트 추출 중...")
        raw_text = self.extract_text_from_docx(input_file)
        if not raw_text:
            return False
        
        # 2. 텍스트 전처리 및 청킹
        print("2. 텍스트 전처리 및 청킹 중...")
        chunks = self.preprocess_legal_text(raw_text)
        print(f"   총 {len(chunks)}개 청크 생성")
        
        # 3. 벡터화
        print("3. 텍스트 벡터화 중...")
        embeddings = self.create_embeddings(chunks)
        
        # 4. JSON 저장
        print("4. 결과 저장 중...")
        saved_count = self.save_to_json(chunks, embeddings, output_file)
        
        print("=" * 50)
        print(f"✅ 전처리 완료! {saved_count}개 법조문이 준비되었습니다.")
        print("=" * 50)
        
        return True
    
    def preview_chunks(self, chunks, num_preview=3):
        """청크 미리보기"""
        print(f"\n📋 청크 미리보기 (총 {len(chunks)}개 중 {min(num_preview, len(chunks))}개):")
        print("-" * 50)
        for i, chunk in enumerate(chunks[:num_preview]):
            print(f"청크 {i+1}:")
            print(f"{chunk[:200]}{'...' if len(chunk) > 200 else ''}")
            print("-" * 50)

def main():
    # 파일 경로 설정 (실제 경로로 수정 필요)
    input_file = "law.docx"  # 입력 파일 경로
    output_file = "legal_documents.json"  # 출력 파일 경로
    
    # 처리 객체 생성
    processor = LegalDocumentProcessor()
    
    # 파일 존재 확인
    if not os.path.exists(input_file):
        print(f"❌ 파일을 찾을 수 없습니다: {input_file}")
        print("파일 경로를 확인하고 다시 시도해주세요.")
        return
    
    # 문서 처리 실행
    success = processor.process_document(input_file, output_file)
    
    if success:
        print(f"🎉 다음 단계: {output_file}을 사용하여 검색 시스템을 구축하세요!")
    else:
        print("❌ 처리 중 오류가 발생했습니다.")

if __name__ == "__main__":
    main()
