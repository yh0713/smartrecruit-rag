import os
from collections import Counter

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

pdf_path=r"D:\CodeLearning\Heima\12_基于人力资源RAG简历智能推荐系统\个人简历测试.pdf"
PARENT_SIZE, CHILD_SIZE ,OVERLAP= 1000, 400,50
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]

def load_pdf(pdf_path:str):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"{pdf_path} 文件不存在")
    pages=PyPDFLoader(pdf_path).load()
    total=sum(len(p.page_content) for p in pages)
    print(f"PDF共{len(pages)}页，总字数为{total}")
    return pages


def split_two_level(pages):
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_SIZE,
        chunk_overlap=0,
        separators=SEPARATORS,
        add_start_index=True,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_SIZE,
        chunk_overlap=OVERLAP,
        separators=SEPARATORS,
    )
    #第一层：切父块，并打上唯一id
    parents=parent_splitter.split_documents(pages)
    for i,doc in enumerate(parents):
        doc.metadata["parent_id"]=f"p{i}"

    #第二层：切子块。split_documents会自动把父块的metadata复制
    #所有子块的metadata都包含parent_id
    children=child_splitter.split_documents(parents)
    return parents,children

if __name__ == '__main__':
    pages=load_pdf(pdf_path)
    parents,children=split_two_level(pages)
    print(f"切分后，共{len(children)}个子块，{len(parents)}个父块")
    dist=Counter([doc.metadata["parent_id"] for doc in children])
    print(f"平均每个父块有{len(children)/len(dist)}个子块")

    first_pid = parents[0].metadata["parent_id"]
    print(f"\n父块 {first_pid} 被切成 {dist[first_pid]} 个子块：")
    for d in children:
        if d.metadata["parent_id"] == first_pid:
            print(f"  [{len(d.page_content):>4} 字] {d.page_content[:40].strip()}...")


    #检索回溯演示，命中一个子块，找回所属的父块
    hit=children[1]
    parent = next(p for p in parents if p.metadata["parent_id"] == hit.metadata["parent_id"])
    print(f"\n命中子块：{hit.page_content[:40].strip()}...")
    print(f"回溯父块：{parent.page_content[:90].strip()}...")
