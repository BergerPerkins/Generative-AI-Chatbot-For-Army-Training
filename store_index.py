from src.helper import load_pdf_file, text_splitter, download_hugging_face_embeddings
import pinecone
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os


load_dotenv()





extracted_data=load_pdf_file(data='data/')
text_chunks=text_splitter(extracted_data)
embeddings = download_hugging_face_embeddings()


pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))

index_name = "ubotz"
pc.create_index(name=index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud='aws', region='us-east-1'))

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings, 
)