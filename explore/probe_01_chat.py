from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv,find_dotenv


def mask(secret:str)->str:
    # 遮罩密钥，只显示前8位和后4位
    if not secret:
        return "(没有读到）"
    return  f"{secret[:8]}****{secret[-4:]}"


def build_llm()->ChatOpenAI:
    api_key=os.getenv("LLM_API_KEY")
    if not api_key:
        raise Exception("没有找到LLM_API_KEY,检查.env是否在项目根目录")
    return ChatOpenAI(model=os.getenv("LLM_MODEL","deepseek-v4-flash-0731"),api_key=api_key,base_url=os.getenv("LLM_BASE_URL"),
                      temperature=float(os.getenv("LLM_TEMPERATURE",0.1)),timeout=30)

def main():
    load_dotenv(find_dotenv())
    print(f"读到key：{mask(os.getenv('LLM_API_KEY'))}")
    print(f"读到URL：{os.getenv('LLM_BASE_URL')}")

    llm=build_llm()
    print(llm.invoke("用一句话解释什么是RAG").content)

if __name__ == '__main__':
    main()
