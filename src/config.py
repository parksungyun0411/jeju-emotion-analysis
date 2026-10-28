# OpenAI API 설정. 키는 이 파일에 적지 않고 환경변수 OPENAI_API_KEY로 넘긴다.
import os

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
