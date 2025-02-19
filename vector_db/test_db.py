from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import ollama
import logging
logging.basicConfig(level=logging.INFO)

MODEL_NAME = "mistral-nemo"
EMBEDDING_MODEL = "mxbai-embed-large"

QA_doc = [
    "question: What is Artificial Intelligence (AI)?, answer: AI is the simulation of human intelligence in machines that can perform tasks such as learning, reasoning, and problem-solving.",
    "question: What is the Turing Test?, answer: The Turing Test is a measure of a machine's ability to exhibit human-like intelligence, proposed by Alan Turing in 1950.",
    "question: What is deep learning?, answer: Deep learning is a subset of machine learning that uses artificial neural networks to model and understand complex patterns in data.",
    "question: What are the main types of machine learning?, answer: The three main types are supervised learning, unsupervised learning, and reinforcement learning.",
    "question: What is blockchain technology?, answer: Blockchain is a decentralized digital ledger that records transactions across multiple computers securely and transparently.",
    "question: What is the main ingredient in sushi?, answer: The main ingredient in sushi is vinegared rice, often paired with raw or cooked seafood, vegetables, and seaweed.",
    "question: What is the world's hottest chili pepper?, answer: The Carolina Reaper is considered the world's hottest chili pepper, with an average of over 1.6 million Scoville Heat Units (SHU).",
    "question: What is the difference between vegan and vegetarian diets?, answer: Vegetarians avoid meat, while vegans avoid all animal products, including dairy, eggs, and honey.",
    "question: What is the capital of Japan?, answer: The capital of Japan is Tokyo.",
    "question: Who developed the theory of relativity?, answer: Albert Einstein developed the theory of relativity.",
    "question: What is the smallest unit of matter?, answer: The atom is the smallest unit of matter that retains the properties of an element.",
    "question: What is photosynthesis?, answer: Photosynthesis is the process by which green plants use sunlight to synthesize food from carbon dioxide and water.",
    "question: How many continents are there on Earth?, answer: There are seven continents on Earth: Africa, Antarctica, Asia, Europe, North America, Oceania, and South America.",
    "question: What is the speed of light?, answer: The speed of light is approximately 299,792 kilometers per second (186,282 miles per second) in a vacuum.",
    "question: What is the largest organ in the human body?, answer: The skin is the largest organ in the human body.",
    "question: Who best in the world?, answer: Batman is storangest can fight to everyone."
    # "question: ปัญญาประดิษฐ์ (AI) คืออะไร?, answer: ปัญญาประดิษฐ์ (AI) คือการจำลองความฉลาดของมนุษย์ในเครื่องจักรที่สามารถเรียนรู้ วิเคราะห์ และแก้ปัญหาได้",
    # "question: การทดสอบทัวริง (Turing Test) คืออะไร?, answer: การทดสอบทัวริงเป็นการวัดความสามารถของเครื่องจักรในการแสดงพฤติกรรมที่เหมือนมนุษย์ ซึ่งถูกเสนอโดยอลัน ทัวริงในปี ค.ศ. 1950",
    # "question: ดีพเลิร์นนิง (Deep Learning) คืออะไร?, answer: ดีพเลิร์นนิงเป็นแขนงหนึ่งของแมชชีนเลิร์นนิงที่ใช้โครงข่ายประสาทเทียมในการเรียนรู้และทำความเข้าใจรูปแบบข้อมูลที่ซับซ้อน",
    # "question: ประเภทของแมชชีนเลิร์นนิงมีอะไรบ้าง?, answer: มี 3 ประเภทหลัก ได้แก่ การเรียนรู้แบบมีผู้สอน (Supervised Learning), การเรียนรู้แบบไม่มีผู้สอน (Unsupervised Learning) และการเรียนรู้แบบเสริมกำลัง (Reinforcement Learning)",
    # "question: เทคโนโลยีบล็อกเชน (Blockchain) คืออะไร?, answer: บล็อกเชนเป็นบัญชีแยกประเภทดิจิทัลแบบกระจายศูนย์ที่ใช้ในการบันทึกธุรกรรมอย่างปลอดภัยและโปร่งใส",
    # "question: ส่วนประกอบหลักของซูชิคืออะไร?, answer: ส่วนประกอบหลักของซูชิคือข้าวปรุงรสด้วยน้ำส้มสายชู มักรับประทานคู่กับปลาดิบ อาหารทะเล หรือผัก",
    # "question: พริกที่เผ็ดที่สุดในโลกคืออะไร?, answer: พริกที่เผ็ดที่สุดในโลกคือพริกคาโรไลนา รีปเปอร์ (Carolina Reaper) ซึ่งมีค่าความเผ็ดเฉลี่ยมากกว่า 1.6 ล้านหน่วยสโกวิลล์ (SHU)",
    # "question: อาหารมังสวิรัติและวีแกนต่างกันอย่างไร?, answer: อาหารมังสวิรัติหลีกเลี่ยงเนื้อสัตว์ ส่วนอาหารวีแกนหลีกเลี่ยงผลิตภัณฑ์จากสัตว์ทุกชนิด รวมถึงนม ไข่ และน้ำผึ้ง",
    # "question: เมืองหลวงของประเทศญี่ปุ่นคือเมืองอะไร?, answer: เมืองหลวงของประเทศญี่ปุ่นคือโตเกียว",
    # "question: ใครเป็นผู้พัฒนา 'ทฤษฎีสัมพัทธภาพ'?, answer: อัลเบิร์ต ไอน์สไตน์ เป็นผู้พัฒนาทฤษฎีสัมพัทธภาพ",
    # "question: อะไรคือหน่วยที่เล็กที่สุดของสสาร?, answer: อะตอมเป็นหน่วยที่เล็กที่สุดของสสารที่ยังคงคุณสมบัติของธาตุอยู่",
    # "question: การสังเคราะห์ด้วยแสง (Photosynthesis) คืออะไร?, answer: การสังเคราะห์ด้วยแสงคือกระบวนการที่พืชใช้พลังงานจากแสงแดดเพื่อผลิตอาหารจากคาร์บอนไดออกไซด์และน้ำ",
    # "question: โลกมีทั้งหมดกี่ทวีป?, answer: โลกมีทั้งหมด 7 ทวีป ได้แก่ แอฟริกา, แอนตาร์กติกา, เอเชีย, ยุโรป, อเมริกาเหนือ, โอเชียเนีย และอเมริกาใต้",
    # "question: ความเร็วของแสงคือเท่าไหร่?, answer: ความเร็วของแสงประมาณ 299,792 กิโลเมตรต่อวินาที (186,282 ไมล์ต่อวินาที) ในสุญญากาศ",
    # "question: อวัยวะที่ใหญ่ที่สุดในร่างกายมนุษย์คืออะไร?, answer: ผิวหนังเป็นอวัยวะที่ใหญ่ที่สุดในร่างกายมนุษย์"
]

def create_retriever(QA_doc):
    # **Vector-based Search**
    ollama.pull(EMBEDDING_MODEL)
    vector_db = Chroma.from_texts(QA_doc, embedding=OllamaEmbeddings(model=EMBEDDING_MODEL))
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": 3})  # Top 3 matches
    logging.info("Vector database created.")

    return vector_retriever

# 2️⃣ **Set Up RAG Chain**
def setup_qa_chain(vector_retriever):
    llm = ChatOllama(model=MODEL_NAME)  # Ollama for LLM
    # **Hybrid Search Retriever (Merging Both)**
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
            different versions of the given user question to retrieve relevant documents from
            a vector database. By generating multiple perspectives on the user question, your
            goal is to help the user overcome some of the limitations of the distance-based
            similarity search. Provide these alternative questions separated by newlines.
            Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_retriever, llm, prompt=QUERY_PROMPT
    )

    logging.info("Retriever created.")
        # RAG prompt
    template = """Answer the question based ONLY on the following context:
        {context}
        Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    logging.info("Chain created successfully.")

    return chain

# 3️⃣ **Run Hybrid Search Chatbot**
def chatbot():
    print("🤖 Hybrid Search Chatbot (type 'exit' to quit)")
    retriever = create_retriever(QA_doc)
    qa_chain = setup_qa_chain(retriever)

    while True:
        query = input("\n📝 Ask a question: ")
        if query.lower() == "exit":
            break
        
        # Retrieve relevant documents
        retrieved_docs = retriever.invoke(query)
        # Extract the text from the documents
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        # Invoke the chain with the retrieved context
        result = qa_chain.invoke({"context": context, "question": query})
        print("\nSource Documents:")
        for i, doc in enumerate(retrieved_docs, 1):
            print(f"{i}. {doc.page_content}")
        print(f"\n✅ Answer: {result}")

if __name__ == "__main__":
    chatbot()