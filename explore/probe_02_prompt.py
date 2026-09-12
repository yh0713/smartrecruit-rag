import os
from dotenv import load_dotenv,find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv(find_dotenv())

def build_llm()->ChatOpenAI:
    return ChatOpenAI(model=os.getenv("LLM_MODEL","deepseek-v4-flash-0731"),
                      api_key=os.getenv("LLM_API_KEY"),
                      base_url=os.getenv("LLM_BASE_URL"),
                      temperature=float(os.getenv("LLM_TEMPERATURE",0.1)),)

#system定规矩，human装变量
prompt=ChatPromptTemplate.from_messages([
    ("system","你是一个资深招聘顾问，只给输出一个json对象，不要任何解释文字"),
    ("human","岗位要求：{jd}\n\n候选人简历：{resume}\n\n"
             "请输出三个字段：score（0-100 的整数）、conclusion（一句话结论）、advice（一条具体建议）")
])

llm=build_llm()
parser=JsonOutputParser()

#管道，数据从左到右传递
chain=prompt|llm|parser

if __name__ == '__main__':
    msgs=prompt.format_messages(jd="Python 后端 3 年", resume="张三，5 年 Python")
    for m in msgs:
        print(f"[{m.type}] {m.content}")

    print("\n-------单条---")
    print(chain.invoke({"jd":"熟悉 LangChain / RAG，3 年以上经验",
            "resume": "李四，5 年后端开发，独立做过 RAG 知识库问答，熟悉 Milvus 和 Elasticsearch",}))

    print("\n-------批量---")
    results=chain.batch([{"jd":"数据分析师", "resume": "王五，精通 SQL 和 Tableau"},
            {"jd": "算法工程师", "resume": "赵六，硕士，CV 方向，发表 2 篇论文"}])

    for i,r in enumerate(results,1):
        print(f"结果 {i}:{r}\n")
