import os
import json
import re
from docx import Document
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Tuple


class KoreanLegalProcessor:
    def __init__(self):
        """한국어 법률 문서 전용 처리 시스템"""

        # 한국어 특화 모델 고정 사용
        self.model_name = "jhgan/ko-sroberta-multitask"


        try:
            self.model = SentenceTransformer(self.model_name)

            # 테스트 임베딩으로 차원 확인
            test_embedding = self.model.encode(["테스트"], convert_to_tensor=False)[0]
            self.vector_dimension = len(test_embedding)
            print(f" 벡터 차원: {self.vector_dimension}")

        except Exception as e:
            print(f" 모델 로딩 실패: {e}")
            self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            test_embedding = self.model.encode(["테스트"], convert_to_tensor=False)[0]
            self.vector_dimension = len(test_embedding)

        # 한국 형법 특화 패턴들
        self.legal_patterns = {
            "편": r'제\s*(\d+)\s*편\s*([^\n]*)',
            "장": r'제\s*(\d+)\s*장\s*([^\n]*)',
            "절": r'제\s*(\d+)\s*절\s*([^\n]*)',
            "조": r'제\s*(\d+)\s*조\s*(?:\(([^)]*)\))?\s*([^\n]*)',
            "항": r'^[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮]',
            "호": r'^(?:\d+\.|\s*[가-힣]\.)',
            "삭제": r'삭제\s*<[^>]*>',
            "개정": r'\[([^]]*(?:개정|신설|전문개정)[^]]*)\]',
            "위헌": r'\[([^]]*(?:위헌|헌법불합치)[^]]*)\]',
            "부칙": r'부\s*칙'
        }


    def extract_from_docx(self, file_path: str) -> List[str]:
        """DOCX 파일에서 텍스트 추출"""

        try:
            document = Document(file_path)
            paragraphs = []

            for para in document.paragraphs:
                text = para.text.strip()
                if text and len(text) > 2:  # 너무 짧은 텍스트 제외
                    paragraphs.append(text)

            return paragraphs

        except Exception as e:
            print(f" 문서 읽기 오류: {e}")
            return []

    def analyze_text_structure(self, paragraphs: List[str]) -> List[Dict]:
        """텍스트 구조 분석"""
        print(" 문서 구조 분석 중...")

        structured_items = []
        current_context = {
            "편": "", "편_번호": "",
            "장": "", "장_번호": "",
            "절": "", "절_번호": ""
        }

        for para in paragraphs:
            item_info = self.classify_paragraph(para)

            # 컨텍스트 업데이트
            if item_info["type"] in ["편", "장", "절"]:
                current_context = self.update_structure_context(
                    current_context, item_info["type"], item_info
                )

            # 구조 정보 포함
            item_info["context"] = current_context.copy()
            structured_items.append(item_info)

        # 통계 출력
        type_counts = {}
        for item in structured_items:
            item_type = item["type"]
            type_counts[item_type] = type_counts.get(item_type, 0) + 1

        print("   구조 분석 결과:")
        for type_name, count in sorted(type_counts.items()):
            print(f"     {type_name}: {count}개")

        return structured_items

    def classify_paragraph(self, text: str) -> Dict:
        """문단 분류"""
        # 삭제된 조문
        if re.search(self.legal_patterns["삭제"], text):
            return {"type": "삭제된_조문", "text": text, "original": text}

        # 편
        편_match = re.match(self.legal_patterns["편"], text)
        if 편_match:
            return {
                "type": "편",
                "text": text,
                "번호": 편_match.group(1),
                "제목": 편_match.group(2).strip(),
                "original": text
            }

        # 장
        장_match = re.match(self.legal_patterns["장"], text)
        if 장_match:
            return {
                "type": "장",
                "text": text,
                "번호": 장_match.group(1),
                "제목": 장_match.group(2).strip(),
                "original": text
            }

        # 절
        절_match = re.match(self.legal_patterns["절"], text)
        if 절_match:
            return {
                "type": "절",
                "text": text,
                "번호": 절_match.group(1),
                "제목": 절_match.group(2).strip(),
                "original": text
            }

        # 조
        조_match = re.match(self.legal_patterns["조"], text)
        if 조_match:
            return {
                "type": "조",
                "text": text,
                "번호": 조_match.group(1),
                "부제": 조_match.group(2) or "",
                "제목": 조_match.group(3).strip(),
                "original": text
            }

        # 항
        if re.match(self.legal_patterns["항"], text):
            return {"type": "항", "text": text, "original": text}

        # 호
        if re.match(self.legal_patterns["호"], text):
            return {"type": "호", "text": text, "original": text}

        # 부칙
        if re.match(self.legal_patterns["부칙"], text):
            return {"type": "부칙", "text": text, "original": text}

        # 개정/위헌 정보
        if re.search(self.legal_patterns["개정"], text):
            return {"type": "개정정보", "text": text, "original": text}

        if re.search(self.legal_patterns["위헌"], text):
            return {"type": "위헌정보", "text": text, "original": text}

        # 일반 내용
        return {"type": "내용", "text": text, "original": text}

    def update_structure_context(self, context: Dict, item_type: str, item_info: Dict) -> Dict:
        """구조 컨텍스트 업데이트"""
        new_context = context.copy()

        if item_type == "편":
            new_context.update({
                "편": item_info["제목"],
                "편_번호": item_info["번호"],
                "장": "", "장_번호": "",
                "절": "", "절_번호": ""
            })
        elif item_type == "장":
            new_context.update({
                "장": item_info["제목"],
                "장_번호": item_info["번호"],
                "절": "", "절_번호": ""
            })
        elif item_type == "절":
            new_context.update({
                "절": item_info["제목"],
                "절_번호": item_info["번호"]
            })

        return new_context

    def smart_chunking(self, structured_items: List[Dict]) -> List[Dict]:
        """스마트 청킹 - 조문 단위 최적화"""
        print(" 스마트 청킹 진행 중...")

        chunks = []
        current_article_group = []

        for item in structured_items:
            item_type = item["type"]

            if item_type == "조":
                # 이전 조문 그룹 처리
                if current_article_group:
                    article_chunks = self.process_article_group(current_article_group)
                    chunks.extend(article_chunks)

                # 새 조문 그룹 시작
                current_article_group = [item]

            elif item_type in ["항", "호", "내용", "개정정보", "위헌정보"]:
                # 조문에 속하는 내용들
                if current_article_group:
                    current_article_group.append(item)
                else:
                    # 조문 없는 일반 내용은 별도 처리
                    if len(item["text"]) > 30:
                        chunks.append(self.create_standalone_chunk(item))

            elif item_type == "삭제된_조문":
                # 삭제된 조문도 정보로 보존
                chunks.append(self.create_deleted_chunk(item))

            # 편, 장, 절은 컨텍스트로만 사용

        # 마지막 조문 그룹 처리
        if current_article_group:
            article_chunks = self.process_article_group(current_article_group)
            chunks.extend(article_chunks)

        print(f"   생성된 청크: {len(chunks)}개")
        return chunks

    def process_article_group(self, article_group: List[Dict]) -> List[Dict]:
        """조문 그룹 처리"""
        if not article_group:
            return []

        # 전체 길이 계산
        total_length = sum(len(item["text"]) for item in article_group)

        # 짧은 조문은 하나의 청크로
        if total_length <= 400:
            return [self.create_article_chunk(article_group)]

        # 긴 조문은 항 단위로 분할
        return self.split_long_article(article_group)

    def split_long_article(self, article_group: List[Dict]) -> List[Dict]:
        """긴 조문을 항 단위로 분할"""
        chunks = []
        article_header = article_group[0]  # 조문 제목
        content_items = article_group[1:]  # 조문 내용

        current_chunk_items = [article_header]
        current_length = len(article_header["text"])

        for item in content_items:
            item_length = len(item["text"])
            item_type = item["type"]

            # 새로운 항에서 분할하거나 너무 길어지면 분할
            should_split = (
                    (item_type == "항" and current_length > 150) or
                    (current_length + item_length > 500) or
                    (item_type in ["개정정보", "위헌정보"] and current_length > 200)
            )

            if should_split and len(current_chunk_items) > 1:
                chunks.append(self.create_article_chunk(current_chunk_items))
                current_chunk_items = [article_header, item]  # 조문 제목 포함
                current_length = len(article_header["text"]) + item_length
            else:
                current_chunk_items.append(item)
                current_length += item_length

        # 마지막 청크
        if len(current_chunk_items) > 1:
            chunks.append(self.create_article_chunk(current_chunk_items))

        return chunks

    def create_article_chunk(self, items: List[Dict]) -> Dict:
        """조문 청크 생성"""
        combined_text = self.combine_text_smartly(items)
        article_info = items[0]  # 조문 정보
        context = items[0]["context"]

        # 메타데이터 생성
        metadata = {
            "type": "조문",
            "편": context["편"],
            "편_번호": context["편_번호"],
            "장": context["장"],
            "장_번호": context["장_번호"],
            "절": context["절"],
            "절_번호": context["절_번호"],
            "조_번호": article_info.get("번호", ""),
            "조_제목": article_info.get("제목", ""),
            "조_부제": article_info.get("부제", ""),
            "항_수": len([item for item in items if item["type"] == "항"]),
            "호_수": len([item for item in items if item["type"] == "호"]),
            "has_amendment": any(item["type"] == "개정정보" for item in items),
            "has_constitutional_issue": any(item["type"] == "위헌정보" for item in items),
            "length": len(combined_text),
            "item_count": len(items)
        }

        return {
            "text": combined_text,
            "metadata": metadata
        }

    def create_standalone_chunk(self, item: Dict) -> Dict:
        """독립적인 청크 생성"""
        context = item.get("context", {})

        metadata = {
            "type": "일반내용",
            "편": context.get("편", ""),
            "장": context.get("장", ""),
            "절": context.get("절", ""),
            "length": len(item["text"])
        }

        return {
            "text": item["text"],
            "metadata": metadata
        }

    def create_deleted_chunk(self, item: Dict) -> Dict:
        """삭제된 조문 청크 생성"""
        context = item.get("context", {})

        metadata = {
            "type": "삭제된_조문",
            "편": context.get("편", ""),
            "장": context.get("장", ""),
            "절": context.get("절", ""),
            "is_deleted": True,
            "length": len(item["text"])
        }

        return {
            "text": item["text"],
            "metadata": metadata
        }

    def combine_text_smartly(self, items: List[Dict]) -> str:
        """텍스트 지능적 결합"""
        text_parts = []

        for i, item in enumerate(items):
            text = item["text"]
            item_type = item["type"]

            if item_type == "조":
                # 조문 제목을 굵게 표시
                text_parts.append(f"**{text}**")

            elif item_type == "항":
                # 새 줄에서 항 시작
                if text_parts and not text_parts[-1].endswith('\n'):
                    text_parts.append('\n')
                text_parts.append(text)

            elif item_type == "호":
                # 호는 들여쓰기로 표시
                text_parts.append(f"\n  {text}")

            elif item_type in ["개정정보", "위헌정보"]:
                # 개정/위헌 정보는 별도 줄에
                text_parts.append(f"\n\n{text}")

            else:
                # 일반 내용
                if text_parts and not text_parts[-1].endswith(('\n', ' ')):
                    text_parts.append(' ')
                text_parts.append(text)

        combined = ''.join(text_parts)

        # 후처리
        combined = re.sub(r'\n\s*\n\s*\n', '\n\n', combined)  # 연속된 빈 줄 정리
        combined = re.sub(r' +', ' ', combined)  # 연속된 스페이스 정리

        return combined.strip()

    def create_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        """한국어 특화 임베딩 생성"""
        if not chunks:
            return []

        try:
            texts = [chunk["text"] for chunk in chunks]

            # 배치 처리로 안정성 향상
            embeddings = self.model.encode(
                texts,
                convert_to_tensor=False,
                show_progress_bar=True,
                batch_size=16  # 안정성을 위해 배치 크기 줄임
            )

            # ID와 벡터 추가
            for i, chunk in enumerate(chunks):
                chunk["id"] = i
                chunk["vector"] = embeddings[i].tolist()

            print(f" 벡터화 완료!")
            print(f"   벡터 차원: {len(embeddings[0])}")

            return chunks

        except Exception as e:
            print(f" 벡터화 오류: {e}")
            return []

    def analyze_quality(self, chunks: List[Dict]):
        """청킹 품질 분석"""

        if not chunks:
            print(" 분석할 청크가 없습니다.")
            return

        # 기본 통계
        total_chunks = len(chunks)
        lengths = [len(chunk["text"]) for chunk in chunks]

        # 타입별 분포
        type_counts = {}
        for chunk in chunks:
            chunk_type = chunk["metadata"].get("type", "기타")
            type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1


        # 조문 분석
        article_chunks = [c for c in chunks if c["metadata"].get("type") == "조문"]
        if article_chunks:
            print(f"\n조문 청크 분석:")
            print(f"  조문 청크 수: {len(article_chunks)}개")

            amendment_count = sum(1 for c in article_chunks
                                  if c["metadata"].get("has_amendment", False))
            print(f"  개정 정보 포함: {amendment_count}개")

            constitutional_count = sum(1 for c in article_chunks
                                       if c["metadata"].get("has_constitutional_issue", False))
            print(f"  위헌 정보 포함: {constitutional_count}개")

    def save_chunks(self, chunks: List[Dict], output_path: str):
        """청크 저장"""
        if not chunks:
            print(" 저장할 데이터가 없습니다.")
            return False


        # 저장용 데이터 준비
        save_data = []
        for chunk in chunks:
            save_data.append({
                "id": chunk["id"],
                "text": chunk["text"],
                "vector": chunk["vector"],
                "metadata": chunk["metadata"]
            })

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)

            print(f" 저장 완료 : {output_path}")


            return True

        except Exception as e:
            print(f" 저장 실패: {e}")
            return False

    def process_korean_legal_document(self, input_file: str, output_file: str):

        # 1. 텍스트 추출
        paragraphs = self.extract_from_docx(input_file)
        if not paragraphs:
            return False

        # 2. 구조 분석
        structured_items = self.analyze_text_structure(paragraphs)
        if not structured_items:
            return False

        # 3. 스마트 청킹
        chunks = self.smart_chunking(structured_items)
        if not chunks:
            return False

        # 4. 한국어 특화 벡터화
        chunks = self.create_embeddings(chunks)
        if not chunks:
            return False

        # 5. 품질 분석
        self.analyze_quality(chunks)

        # 6. 저장
        success = self.save_chunks(chunks, output_file)

        if success:
            print(f"   (파일: {output_file})")

        return success


def main():
    # 파일 설정
    input_file = "law.docx"
    output_file = "korean_optimized_legal.json"


    if not input_file:
        print("law.docx 파일 준비가 필요합니다.")


    print(f" 입력 파일: {input_file}")
    print(f" 출력 파일: {output_file}")

    # 처리기 생성 및 실행
    processor = KoreanLegalProcessor()

    # 파일 처리
    success = processor.process_korean_legal_document(input_file, output_file)

if __name__ == "__main__":
    main()