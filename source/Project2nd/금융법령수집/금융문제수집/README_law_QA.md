
# 법령 문제 자동 생성 시스템 (OpenAI API)

이 저장물에는 OpenAI API를 사용하여 `.docx` 법령 문서로부터
**사지선다형(4지선다) + 단답형** 문제를 자동 생성해
엑셀 시트 2개(사지선다형/단답형)로 저장하는 스크립트가 포함됩니다.

## 준비물
1) Python 3.9+
2) OpenAI API Key
3) `.docx` 형식의 법령 파일들 (폴더 안에 모아두기)

## 설치
```bash
pip install -r requirements.txt
```

## 환경 변수
```bash
# macOS/Linux
export OPENAI_API_KEY="sk-..."

# Windows PowerShell
$env:OPENAI_API_KEY="sk-..."
```

## 설정(config.yaml)
- `input_dirs`: `.docx`가 들어있는 폴더 경로(쉼표로 구분)
- `output_excel`: 결과 엑셀 파일 경로
- 기타 비율, 난이도, 배치 사이즈 등 조정 가능

## 실행
```bash
python law_question_generator.py --config config.yaml
```

## 중단/재개
- `processed_state.json`에 처리된 파일 목록이 저장되므로,
  중간에 중단되어도 **다시 실행하면 이어서** 처리합니다.

## 참고
- API 사용법과 Chat Completions 포맷은 OpenAI 공식 문서를 참조하세요.
