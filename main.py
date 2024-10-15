import os
import pickle
import re
import queue
from io import BytesIO
from urllib.parse import urlparse, urlunparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import gradio as gr
import boto3
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from theme import JS_LIGHT_THEME, CSS
# from langchain import hub
# from langchain_core.output_parsers import StrOutputParser


local_llm="llama3"
ollama_host = os.getenv("OLLAMA_HOST", "localhost") #enable if you want to run locally and not on docker
# ollama_host = os.getenv("OLLAMA_HOST", "ollama-container")
ollama_port = os.getenv("OLLAMA_PORT", "11434")
ollama_url = os.getenv("OLLAMA_URL", f"http://{ollama_host}:{ollama_port}")

executor = ThreadPoolExecutor(max_workers=14)

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "test1")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "test2")
AWS_REGION_NAME = os.getenv("AWS_REGION_NAME", "us-east-1")

# Configure Boto3 to use LocalStack
session = boto3.Session(
    aws_access_key_id= AWS_ACCESS_KEY_ID,         # Dummy AWS access key
    aws_secret_access_key= AWS_SECRET_ACCESS_KEY,     # Dummy AWS secret key
    region_name= AWS_REGION_NAME           # AWS region
)

# Get the LocalStack host and port from environment variables
# localstack_host = os.getenv('LOCALSTACK_HOST', 'localstack-main')
localstack_host = os.getenv('LOCALSTACK_HOST', 'localhost') #enable for local host
localstack_port = os.getenv('LOCALSTACK_PORT', '4566')

# Construct the endpoint URL
endpoint_url = os.getenv("AWS_S3_URL" , f'http://{localstack_host}:{localstack_port}')

s3 = session.client('s3', endpoint_url=endpoint_url)

BUCKET_NAME = os.getenv("AWS_BUCKET_NAME",'my-first-bucket')


# compiled ENV variables needed
# OLLAMA_HOST=ollama-container
# OLLAMA_PORT=11434
# OLLAMA_URL=http://ollama-container:11434
# AWS_ACCESS_KEY_ID=test1
# AWS_SECRET_ACCESS_KEY=test2
# AWS_REGION_NAME=us-east-1
# LOCALSTACK_HOST=localstack-main
# LOCALSTACK_PORT=4566
# AWS_S3_URL=http://localstack-main:4566
# AWS_BUCKET_NAME=my-first-bucket



### Document grader
llm = ChatOllama(base_url=ollama_url, model=local_llm, format="json", temperature = 0)
prompt= PromptTemplate(
    template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> 
    You are a grader assessing relevance of a retrieved document or dataframe to a user question. 
    If the document contains keywords related to the user question, grade it as relevant.
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate wherther the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.\n
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the retrieved document: \n\n {document}\n\n
    Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question","document"],
)
validation_agent = prompt | llm | JsonOutputParser()
# Have a key 'data' which value is the data passed in
# have another key 'reason' which value is the reason why the binary score was given

###Error identifier
#LLM
llm = ChatOllama(base_url=ollama_url, model=local_llm, format="json", temperature = 0)
error_identifier_prompt= PromptTemplate(
    template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> 
    You are an indentifier.
    Identify the error codes from the given question / sentence. An example could be VCA373 . Error codes usually start with a few capital alphabets followed by a few numbers\n
    It does not need to be a stringent test. The goal is to identify only the error codes and return that. \n
    It MUST have both the alpahbets and numbers together.\n
    Provide the error codes as a JSON with a single key 'error_code' and no preamble or explanation.\n
    Return the error codes if they are present, else return an the JSON with the value as an empty listt for example 'error_code' : []. \n
    Put the error codes in a list. For example 'error_code' : ['VCA373'] .\n
    If there are multiple error codes then for example 'error_code' : ['VCA373', 'CHC016'].\n
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the sentence: \n\n {question}\n\n
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question"],
)
error_identifier = error_identifier_prompt | llm | JsonOutputParser()

### Hallucination checker
llm = ChatOllama(base_url=ollama_url, model=local_llm, format="json", temperature=0)
prompt= PromptTemplate(
    template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> 
    You are a grader assessing whether an answer is grounded in / supported by a set of facts.\n
    Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts.\n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.\n
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here are the facts: 
    \n-------\n 
    {documents}
    \n-------\n
    Here is the answer: {generation} <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["generation","document"],
)
hallucination_checker_agent = prompt | llm | JsonOutputParser()


### Answer grader
llm = ChatOllama(base_url=ollama_url, model=local_llm, format="json", temperature=0)
prompt= PromptTemplate(
    template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> 
    You are a grader assessing whether an answer is addressing the question properly.\n
    Give a binary score 'yes' or 'no' score to indicate whether the answer addresses the question well.\n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.\n
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question}
    Here is the answer: {generation} <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question", "generation"],
)
answer_grader = prompt | llm | JsonOutputParser()

### chunk_filter_agent
llm = ChatOllama(base_url=ollama_url, model=local_llm, format="json", temperature=0)
prompt= PromptTemplate(
    template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> 
    You are a grader assessing whether an chunk is useful in answering the query properly.\n
    Give a binary score 'yes' or 'no' score to indicate whether the chunk will help in answering the question well.\n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.\n
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the query: {query}
    Here is the chunk: {chunk} 
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["query", "chunk"],
)
chunk_filter_agent = prompt | llm | JsonOutputParser()



### Context formatter
context_formatting_template = PromptTemplate(
    template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> 
    You are an assistant for formatting tasks. Do not alter the content or add in any new content.\n
    You are only in charge of reformatting the context given to you such that it is nice to read and in markdown format. \n
    The context is parsed from a pdf and might be part of code, descriptions or tables. Do your best to format it nicely in a human readable format.\n

    Only return the "definition" and "documentation" blocks of code/text and ignore the rest.\n
    For the "definition" block of code, add appropriate line breaks so it lo
    Format the given "context" variable and return it.\n
    If no "context" variable is given, just return an empty string.\n
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Context: {context}
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["context"],
)
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm_formatter = ChatOllama(base_url=ollama_url, model=local_llm, temperature=0, callbacks=callback_manager)


prompt_template= PromptTemplate(
    template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> 
    You are an assistant for question-answering tasks. Often, the context might be in code, descriptions, tables and JSON, do your best to analyse them.\n
    You can try to identify the specific things mentioned in the question and work from there.\n
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. \n
    If you dont know, also mention what you could discern specifically from the context as well as what you think you might need to answer the given question.\n
    
    For FAQ questions, you can answer in this format:
    Provide a concise FAQ on ERR001:
    what is the error
    what are the triggers
    What are the fields needed to rectify the error
    How to resolve this error

    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question}
    Context: {context}
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question","context"],
)
# llm_main = prompt_template | llm | StrOutputParser()

prompt_template_history= PromptTemplate(
    template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> 
    You are an assistant for question-answering tasks. Often, the context might be in code, descriptions, tables and JSON, do your best to analyse them.\n
    You can try to identify the specific things mentioned in the question and work from there.\n
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. \n
    If you dont know, also mention what you could discern specifically from the context as well as what you think you might need to answer the given question.\n
    If your Chat history is provided to you, please take into account the history of the chat and answer with it in mind.\n
    The format of the chat history is a list of lists. An example is [["my question", "your response"],["my question", None]].\n
    
    For FAQ questions, you can answer in this format, but answer to the point concisely and factually based on the context:
    Provide a concise FAQ on ERR001:
    what is the error
    what are the triggers
    What are the fields needed to rectify the error
    How to resolve this error

    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question}
    Context: {context}
    Chat History: {history}
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question","context", "history"],
)

prompt_template_general= PromptTemplate(
    template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> 
    You are an assistant for question-answering tasks in the context of Singapore National portal for Healthcare claim portal working with mediclaim.\n
    In the question, you should be provided with a query and possibly provided with a XOM to analyse based on the query. \n
    Else it will be a general query, or maybe about the process or maybe what to do in certain situations (playbook)\n
    If you don't know the answer, just say that you don't know. \n
    
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question}
    Context: {context}
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question","context"],
)

prompt_template_general_history= PromptTemplate(
    template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> 
    You are an assistant for question-answering tasks in the context of Singapore National portal for Healthcare claim portal working with mediclaim.\n
    In the question, you should be provided with a query and possibly provided with a XOM to analyse based on the query. \n
    Else it will be a general query, or maybe about the process or maybe what to do in certain situations (playbook)\n
    If you don't know the answer, just say that you don't know. \n

    If your Chat history is provided to you, please take into account the history of the chat and answer with it in mind.\n
    The format of the chat history is a list of lists. An example is [["my question", "your response"],["my question", None]].\n
    
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question}
    Context: {context}
    Chat History: {history}
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question","context","history"],
)

llm_main = ChatOllama(base_url=ollama_url, model=local_llm, temperature=0, callbacks=callback_manager)

### Extract from Excel
def parse_and_upload_excel_as_pkl(file_path, sheet_name, s3_file_key):
    """
    Parse a specific sheet in the Excel file and upload the DataFrame as a .pkl to S3.
    
    :param file_path: Path to the Excel file.
    :param sheet_name: Name or index of the sheet to parse.
    :param s3_file_key: S3 key (including folder path) to save the .pkl file.
    """
    # Load the specific sheet into a DataFrame
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    # Save the DataFrame to a bytes buffer
    buffer = BytesIO()
    pickle.dump(df, buffer)
    buffer.seek(0)
    
    # Upload the .pkl file to S3
    s3.put_object(Bucket=BUCKET_NAME, Key=s3_file_key, Body=buffer.getvalue())
    print(f"DataFrame from sheet '{sheet_name}' uploaded to s3://{BUCKET_NAME}/{s3_file_key}")


def load_df(s3_file_key):
    """
    Load the saved DataFrame from a file in S3.
    
    :param s3_file_key: S3 key to the saved DataFrame.
    :return: Loaded DataFrame.
    """
    # Download the file from S3 into a buffer
    response = s3.get_object(Bucket=BUCKET_NAME, Key=s3_file_key)
    buffer = BytesIO(response['Body'].read())
    
    # Load the DataFrame from the buffer
    df = pickle.load(buffer)
    return df

def search_df(df, search_column, search_value):
    """
    Search for the specified value in the given column of the DataFrame.
    :param df: The DataFrame to search.
    :param search_column: The column to search in.
    :param search_value: The value to search for.
    :return: Filtered DataFrame or None if no match is found.
    """
    filtered_df = df[df[search_column] == search_value]
    
    if not filtered_df.empty:
        return filtered_df  # Return the DataFrame directly
    else:
        print(f"No match found for {search_value} in column {search_column}")
        return None

    
def update_excel_df(df):
    if isinstance(df, pd.DataFrame):
        print("The variable is a DataFrame.")

    else:
        print("The variable is not a DataFrame.")
    
    print("dataframeeee" , df.to_string()) 
    return gr.update(value=df)

default_headers = ["S/N", "Field Name", "Business Rule", "Error Code", "Error Description", "Associated Policy", "For Claim/ Non-Claim", "Tag", "Removed"]
empty_df = pd.DataFrame(columns=default_headers)


def extract_error_code(question):
    error_code_JSON = error_identifier.invoke({"question": question})
    print("error_code_JSON: ", error_code_JSON)
    return error_code_JSON["error_code"]

def list_pkl_files_in_s3_folder(bucket_name):
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix="pkl")
    if 'Contents' in response:
        return [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.pkl')]
    return []

def list_txt_files_in_s3_folder(bucket_name):
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix="txt")
    if 'Contents' in response:
        return [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.txt')]
    return []

def list_img_files_in_s3_folder(bucket_name):
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix="image")
    if 'Contents' in response:
        return [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith(('.png', '.jpg', '.jpeg'))]
    return []

def read_txt_file_from_s3(bucket_name, file_key):
    response = s3.get_object(Bucket=bucket_name, Key=file_key)
    content = response['Body'].read().decode('utf-8')
    return content

### Document Retrieval and Combination
def retrieve_and_combine_documents(question):
    error_code_list = extract_error_code(question)
    combined_docs = []
    page_list = []
    lines = []

    if error_code_list != []:
        print(f"Extracted Error Code(s): {error_code_list}")

        file_keys = list_txt_files_in_s3_folder(BUCKET_NAME)
        for file_key in file_keys:
            content = read_txt_file_from_s3(BUCKET_NAME, file_key)
            lines.extend(content.splitlines())

        for error_index in range(0, len(error_code_list)):

            matched_snippets = []
            skip_lines = 0
            skip = False
            doc_count = 0
            page_numbers = []

            for i in range(0, len(lines)):

                skip_lines -= 1

                if error_code_list[error_index] in lines[i] and not skip:
                    doc_count += 1
                    start = max(0, i - 1)
                    end = min(len(lines), i + 5)
                    snippet = ''.join(lines[start:end])

                    header = f"Document {doc_count} for error {error_code_list[error_index]}:\n"
            
                    # Prepend the header to the snippet
                    snippet_with_header = header + snippet
                    
                    # Append the modified snippet to the matched snippets list
                    matched_snippets.append(snippet_with_header)

                    skip_lines = 10
                    skip = True

                    # Find the next two occurrences of page numbers from the `start` line
                    page_pattern = r'(\d+)/495'
                    pages = re.findall(page_pattern, '\n'.join(lines[start:]))
                    print("page num: ", pages[:5])
                    # Decrement each page number by one before adding to page_numbers
                    page_numbers.extend([str(int(page) - 1) for page in pages[:2]])

                    if len(matched_snippets) >= 4:
                        break

                if skip_lines <= 0:
                    skip = False
                    skip = 0


            result = matched_snippets[:4]
            combined_docs.append(result)
            page_list.extend(page_numbers[:2])
            # print(result)

            if result != []:
                print("-----------Exact match found in txt for error" + error_code_list[error_index] + "-----------")
            else:
                 print(f"-----------No Error Code {error_code_list[error_index]} found in the documents-----------")

    else:
        print("-----------No Error Code found in the question-----------")

    # print("combined docs:",combined_docs)
    return combined_docs, error_code_list, page_list

def quick_test_retrieve_and_combine_documents(question):
    combined_docs, error_code_list, page_list = retrieve_and_combine_documents(question)
    return combined_docs, error_code_list, page_list



#GALLERY FUNCTIONS
def update_gallery(page_list):
    if page_list == []:
        new_gallery_items = create_gallery_items()
    else:
        new_gallery_items = create_gallery_items(page_list)
    return gr.update(value=new_gallery_items)

def adjust_presigned_url(url, hostname):
    parts = urlparse(url)
    new_parts = parts._replace(netloc=hostname)
    return urlunparse(new_parts)

def create_gallery_items(page_list=None):
    file_keys = list_img_files_in_s3_folder(BUCKET_NAME)
    # Sort files numerically based on the number at the end of their names
    file_keys.sort(key=lambda x: int(x.split('.')[-2].split('_')[-1]))  # Assumes file names are like 'file_1.jpg'
    gallery_items = []
    hostname = 'localhost:4566'  # Ensure this matches where LocalStack is accessible
    if page_list:
        for page in page_list:
            try:
                index = int(page) - 1
                if 0 <= index < len(file_keys):
                    content_type = f"image/{file_keys[index].split('.')[-1]}"
                    file_url = s3.generate_presigned_url(
                        'get_object', 
                        Params={'Bucket': BUCKET_NAME, 'Key': file_keys[index], 'ResponseContentType': content_type}, 
                        ExpiresIn=3600
                    )
                    file_url = adjust_presigned_url(file_url, hostname)
                    gallery_items.append((file_url, f"Page {page}"))
            except ValueError:
                print(f"Invalid page number: {page}")
    else:
        page_num = 0
        for key in file_keys:
            content_type = f"image/{key.split('.')[-1]}"
            file_url = s3.generate_presigned_url(
                'get_object', 
                Params={'Bucket': BUCKET_NAME, 'Key': key, 'ResponseContentType': content_type}, 
                ExpiresIn=3600
            )
            file_url = adjust_presigned_url(file_url, hostname)
            gallery_items.append((file_url, f"Page {page_num + 1}"))
            page_num += 1
    return gallery_items


# Load all images initially
gallery_items = create_gallery_items()


#MESSAGES AND RESPONSE FUNCTIONS

def clear_s3_folder(folder_name):
    objects_to_delete = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=folder_name)
    if 'Contents' in objects_to_delete:
        delete_keys = [{'Key': obj['Key']} for obj in objects_to_delete['Contents']]
        s3.delete_objects(Bucket=BUCKET_NAME, Delete={'Objects': delete_keys})

def upload_file_to_s3(file_paths, folder_name):
    if not file_paths:
        return "No files to upload."
    
    # Clear the S3 folder first
    clear_s3_folder(folder_name)

    if folder_name == "pkl":
        sheet_name = "ecommerce_error_data"
        file_path = file_paths[0]
        base_filename = os.path.splitext(os.path.basename(file_path.name))[0] + '.pkl'
        upload_path = os.path.join(folder_name, base_filename)
        print("File path:", file_path.name)
        print("Upload path:", upload_path)
        parse_and_upload_excel_as_pkl(file_path.name, sheet_name, upload_path)
        return f"{len(file_paths)} file(s) uploaded successfully to {BUCKET_NAME}/{folder_name}/."

    for file_path in file_paths:
        file_name = os.path.basename(file_path.name)
        key = folder_name + '/' + file_name  # Add folder path to the key
        with open(file_path.name, 'rb') as data:
            s3.upload_fileobj(data, BUCKET_NAME, key)
    return f"{len(file_paths)} file(s) uploaded successfully to {BUCKET_NAME}/{folder_name}/."

def upload_and_clear(file_paths, folder_name):
    result = upload_file_to_s3(file_paths, folder_name)
    return result


def format_history(msg, history):
    chat_history = [{"role": "system", "content": "hi"}]
    for query, response in history:
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": response})
    chat_history.append({"role": "user", "content": msg})
    return chat_history

def format_context(context):
    context_prompt = context_formatting_template.format(context=context)
    context_response = llm_formatter.stream(context_prompt)
    formatted_context = ""
    for partial_context in context_response:
        formatted_context += partial_context.content
    return formatted_context


def hallucination_check(docs, response):
    hallucination_result = hallucination_checker_agent.invoke({"documents": docs, "generation": response})
    return "\nBot is likely hallucinating as answer is not grounded in the context :(" if hallucination_result["score"] == "no" else "\nBot is unlikely hallucinating and answer is grounded in context! :)"

def grade_response(question, response):
    grading_result = answer_grader.invoke({"question": question, "generation": response})
    return "\nBot response answers the question poorly :(" if grading_result["score"] == "no" else "\nBot response answers the question well! :)"


def chunk_filter(chunk, query):
    # Check relevance of the chunk
    # This function should return True if the chunk is relevant, False otherwise
    chunk_filter_result = chunk_filter_agent.invoke({"query": query, "chunk": chunk})
    print("chunk filter result--",chunk_filter_result)
    return chunk, chunk_filter_result["score"]

def chunk_data(data, lines_per_chunk):
    # Function to chunk data into sections of `lines_per_chunk` lines
    lines = data.splitlines()
    for i in range(0, len(lines), lines_per_chunk):
        yield "\n".join(lines[i:i + lines_per_chunk])


def generate_response(message, msg_context, history, top_k, top_p, temperature, chat_history=False, check_error=True, validate=False, check_hallucination=False, grade_answer=False, page_list=None):
    if history is None:
        history = []

    default_headers = ["S/N", "Field Name", "Business Rule", "Error Code", "Error Description", "Associated Policy", "For Claim/ Non-Claim", "Tag", "Removed"]
    excel_result = pd.DataFrame(columns=default_headers)

    yield history, gr.update(), excel_result
    page_list.clear()
    futures = {}

    if check_error:
        #TEXT / PDF PROCESSING
        # # Retrieve and combine documents
        combined_docs, error_code_list, new_page_list = retrieve_and_combine_documents(message)
        page_list.extend(new_page_list)

        # Validate documents
        if validate:
            valid_docs = []
            for doc in combined_docs:
                validation_result = validation_agent.invoke({"question": message, "document": doc})
                print(f"text document: {doc}")
                print(f"text document result: {validation_result}")
                if validation_result["score"] == "yes":
                    valid_docs.append(doc)
            combined_docs = valid_docs

        #EXCEL PROCESSING
        #Load the saved DataFrame
        pkl_files = list_pkl_files_in_s3_folder(BUCKET_NAME)
        loaded_df = load_df(pkl_files[0])

        #search the loaded DataFrame
        search_col = 'Error Code'

        if error_code_list:
            for error_index in range(len(error_code_list)):
                excel_searched = search_df(loaded_df, search_col, error_code_list[error_index])
                # print(f"Search result for {error_code_list[error_index]}: {excel_searched}")  # Debug statement
                if isinstance(excel_searched, pd.DataFrame) and not excel_searched.empty:
                    if validate:
                        excel_serached_string = excel_searched.to_string()
                        excel_validation_result = validation_agent.invoke({"question": message, "document": excel_serached_string})
                        # print(f"Excel document: {excel_searched}")
                        print(f"Excel document result: {excel_validation_result}")
                        if excel_validation_result["score"] == "yes":
                            excel_result = pd.concat([excel_result, excel_searched])
                    else:
                        excel_result = pd.concat([excel_result, excel_searched])

        context_without_excel = "\n\n".join([str(doc) for doc in combined_docs]) 
        combined_docs_string =  context_without_excel + "\n\n" + "From Excel:" + "\n" + excel_result.to_string()
        print(f"final context: {combined_docs_string}")
        print(" -----------final context formulated-----------")
        
        # Setup for context formatting in a separate thread

        #add in error handling!!!!!!!
        futures['context'] = executor.submit(format_context, context_without_excel)
        print("-----------starting to process context concurrently-----------")

    else:
        context_without_excel = ""
        combined_docs_string = ""

        #process context via chunking if context is present

        # Example data and query
        context_data = msg_context  # Your JSON data as a string
        query = message  # User's query
        chunks = list(chunk_data(context_data, lines_per_chunk=150))

        filtered_chunks = []
        chunk_futures = {executor.submit(chunk_filter, chunk, query): chunk for chunk in chunks}
    
        for chunk_future in as_completed(chunk_futures):
            try:
                chunk, score = chunk_future.result()
                if score == "yes":  # Check if the chunk is relevant based on the score
                    filtered_chunks.append(chunk)
                    print(f"\nchunk added to list: {chunk[:100]}...")  # Print first 100 chars of chunk
                else:
                    print("\nchunk not added")
            except Exception as e:
                chunk = chunk_futures[chunk_future]
                print(f"Error retrieving result for chunk: {chunk[:100]}... Error: {e}")

    if check_error:
        # Generate the main response
        if chat_history:
            prompt = prompt_template_history.format(question=message, context=combined_docs_string, history=history)
            # print("prompting with history: ", history)
            print("-----------FAQ prompting with history-----------")

        else:
            prompt = prompt_template.format(question=message, context=combined_docs_string) 
            print("-----------FAQ prompting without history-----------")


    # else:
    #     if chat_history:
    #         prompt = prompt_template_general_history.format(question=message, context=filtered_chunks, history=history)

    #     else:
    #         prompt = prompt_template_general.format(question=message, context=filtered_chunks)

    else:
        if chat_history:
            prompt = prompt_template_history.format(question=message, context=filtered_chunks, history=history)
            print("XOM / general prompt with history")

        else:
            prompt = prompt_template.format(question=message, context=filtered_chunks)
            print("XOM / general prompt without history")
            print("filtered chunks", filtered_chunks)

    print("-----------starting main response-----------")
    result = ""
    # num_ctx = 4096
    # num_ctx = 8192
    # Stream the main response directly
    for partial_result in main_response(prompt, top_k, top_p, temperature):
        result += partial_result
        history[-1][1] = result
        yield history, gr.update(), excel_result

    main_result_response = result

    if check_hallucination:
        print("hallucination checker running")
        futures['hallucination'] = executor.submit(hallucination_check, combined_docs_string, main_result_response)

    if grade_answer:
        print("answer grader running")
        futures['grading'] = executor.submit(grade_response, message, main_result_response)


    print("-----------starting checks and returning formatted context-----------")
    if futures['context']:
        formatted_context = futures['context'].result()
    else:
        formatted_context = ""
    
    if formatted_context != "":
        result = result + "\n\n**Formatted Context:**\n" + formatted_context
        history[-1][1] = result
        yield history, gr.update(), excel_result
    
    # Perform hallucination check if requested
    if check_hallucination:
        hallucination_message = futures['hallucination'].result()
        print("hallucination message --------", hallucination_message)
        result += "\n\n**Hallucination Check:**\n" + hallucination_message
        history[-1][1] = result
        yield history, gr.update(), excel_result

    # Perform answer quality check if requested
    if grade_answer:
        grading_message = futures['grading'].result()
        print("grader message --------", grading_message)
        result += "\n\n**Answer Grading:**\n" + grading_message
        history[-1][1] = result
        yield history, gr.update(), excel_result


    history[-1][1] = result
    yield history, gr.update(), excel_result

    # executor.shutdown()
    # print("shut down------------")
    return history, page_list, excel_result

def add_message(history, message):
    if message is not None:
        history.append([message, None])
    return history, gr.update(value="")

def show_uploading_message():
    return gr.update(value="Uploading files...", visible=True)

# Function to undo the last message
def undo_message(state):
    if state:
        state.pop()
    return state, state

# Function to clear all messages
def clear_messages():
    return [], []

# Function to stop the current processing
def stop_processing(stop_flag):
    stop_flag[0] = True
    executor.shutdown(wait=False)  # Gracefully shutdown the thread pool
    return "Stopped!"

def toggle_textbox(show):
    return gr.update(visible=not show)

def handle_query(query, msg_context, history, top_k, top_p, temperature, chat_history, check_error, validate, check_hallucination, grade_answer, page_state, q):
    generator = generate_response(query,msg_context, history, top_k, top_p, temperature, chat_history, check_error, validate, check_hallucination, grade_answer, page_list=page_state)
    try:
        for response in generator:
            q.put(response)  # Put each response in the queue
    finally:
        q.put(None)  # Signal that the generator is done


def main_response(prompt, top_k, top_p, temperature):
    main_response = llm_main.stream(prompt, top_k=int(top_k), top_p=float(top_p), temperature=float(temperature))
    for partial_answer in main_response:
        yield partial_answer.content

#GRADIO INTERFACE

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="slate",).set(
        button_primary_background_fill="#FF0000",
        button_primary_background_fill_hover="#FF0000",
    ),
    js=JS_LIGHT_THEME,
    css=CSS,
) as demo:
    # gr.Markdown("## Day2Ops Chatbot 🤖")

    with gr.Tab("Interface"):
        sidebar_state = gr.State(True)
        with gr.Row():
            with gr.Column(
                variant="panel", scale=10, visible=sidebar_state.value
            ) as setting:
                with gr.Column():
                    status = gr.Textbox(
                        label="Status", value="Ready!", interactive=False
                    )
                    language = gr.Radio(
                        label="Language",
                        choices=["Eng"],
                        value="Eng",
                        interactive=True,
                        elem_id="language"
                        
                    )
                    model = gr.Dropdown(
                        label="Choose Model:",
                        choices=[
                            "llama3:latest",
                        ],
                        value="llama3:latest",
                        interactive=True,
                        allow_custom_value=True,
                    )
                    error_checkbox = gr.Checkbox(label="Error code focused (Enabled by default)", value='True')
                    chat_history = gr.Checkbox(label="Enable message memory (takes longer the more you query)")
                    validate_checkbox = gr.Checkbox(label="Validate Documents if relevant (takes longer)")
                    hallucination_checkbox = gr.Checkbox(label="Check if there is hallucination of answer (takes longer)")
                    check_answer_checkbox = gr.Checkbox(label="Check answer if relevant to question (takes longer)")

            with gr.Column(scale=30, variant="panel"):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    layout="bubble",
                    value=[],
                    height=550,
                    scale=2,
                    show_copy_button=True,
                    bubble_full_width=False,
                    avatar_images=("assets/user.png","assets/bot.png"),
                )
                with gr.Row():
                    message_input = gr.Textbox(
                        placeholder="Enter query...", 
                        show_label=False, 
                        scale=3, 
                        lines=1,
                    )

                    submit_btn = gr.Button("Submit",scale=1, elem_id="submit_btn")
                
                with gr.Row():
                    context_textbox = gr.Textbox(
                        visible=False, 
                        placeholder="Add in important context eg XOM...", 
                        interactive=True,
                        show_label=False
                    )

                    error_checkbox.change(fn=toggle_textbox, inputs=error_checkbox, outputs=context_textbox)


                with gr.Row(variant="panel"):
                    undo_btn = gr.Button(value="Undo", min_width=20)
                    clear_btn = gr.Button(value="Clear", min_width=20)
                    stop_btn = gr.Button(value="Stop", min_width=20, elem_id='stop_btn')
        
        gr.Markdown("### Context Excel Output")
        with gr.Row():
            
            excel_df = gr.Dataframe(
            empty_df,
            headers=default_headers, label="Context Excel Output", show_label=False, elem_id="excel_df", wrap=True,
            column_widths = ['4%','10%','28%','7%','17%','15%','8%','5%','5%']
            )

        gr.Markdown("### Documentation Collection Output")
        with gr.Row():
            
            gallery = gr.Gallery(
            gallery_items,
            label="Documentation Collection Output", show_label=False, elem_id="gallery",
            columns=3, rows=3, object_fit="contain", height=2000
            )

    with gr.Tab("Settings"):
        with gr.Row():
            top_k = gr.Slider(0.0,100.0, label="top_k", value=40, info="Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40)")
            top_p = gr.Slider(0.0,1.0, label="top_p", value=0.9, info=" Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)")
            temp = gr.Slider(0.0,2.0, label="temperature", value=0.8, info="The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.8)")


    with gr.Tab("Upload"):
        gr.Markdown("#### Please upload all the files required in the category that you want to update as this will overwrite ALL files in that category")
        result_output = gr.Textbox(visible=False)

        gr.Markdown("##### Use PDF Gear -> PDF to JPEG -> upload your pdf file there -> split from 1 to max page -> upload all jpeg files below")
        with gr.Row():
            file_output_image = gr.File(label="Upload and update all pdf image files", interactive=True, file_count="multiple", file_types=["image"])
            folder_name_image = gr.Textbox(value="image", visible=False) 
        
        with gr.Row():
            upload_button_image = gr.Button("Click to Upload image to S3", elem_id='upload_btn_img')
            upload_button_image.click(
                lambda: gr.Info("Uploading files..."),  # Show intermediate message
                inputs=None,
                outputs=None  # Update the result textbox
            ).then(
                upload_and_clear,  # Perform upload
                inputs=[file_output_image, folder_name_image],
                outputs=result_output  # Capture the output to update the result textbox
            ).then(
                update_gallery,  # Perform update
                inputs=None, #refresh the gallery to show full s3 bucket
                outputs=gallery,  # update gallery element
            ).then(
                lambda result: gr.Info(result),  # Show final message using gr.Info
                inputs=result_output,  # Pass the final result to gr.Info
                outputs=None  # No need to update any component
            )
        
        gr.Markdown("##### Use Foxit(for PDFs) or use Microsoft WORD -> open your PDF / Word docx -> Ctrl + A  to select all -> Ctrl + C to copy -> paste into a .txt file (any name is fine) -> upload file below")
        with gr.Row():
            file_output_txt = gr.File(label="Upload and update all txt files", interactive=True, file_count="multiple", file_types=["text"])
            folder_name_txt = gr.Textbox(value="txt", visible=False) 

        with gr.Row():
            upload_button_text = gr.Button("Click to Upload text to S3",  elem_id='upload_btn_txt')
            upload_button_text.click(
                lambda: gr.Info("Uploading files..."),  # Show intermediate message
                inputs=None,
                outputs=None  # Update the result textbox
            ).then(
                upload_and_clear,  # Perform upload
                inputs=[file_output_txt, folder_name_txt],
                outputs=result_output  # Capture the output to update the result textbox
            ).then(
                lambda result: gr.Info(result),  # Show final message using gr.Info
                inputs=result_output,  # Pass the final result to gr.Info
                outputs=None  # No need to update any component
            )

        gr.Markdown("##### Upload your Excel .xlsx file below")
        with gr.Row():
            file_output_pkl = gr.File(label="Upload and update excel file as .pkl", interactive=True, file_count="multiple", file_types=[".xlsx"] )
            folder_name_pkl = gr.Textbox(value="pkl", visible=False) 

        with gr.Row():
            upload_button_pkl = gr.Button("Click to Upload Excel to S3 as .pkl", elem_id='upload_btn_pkl')
            upload_button_pkl.click(
                lambda: gr.Info("Uploading files..."),  # Show intermediate message
                inputs=None,
                outputs=None  # Update the result textbox
            ).then(
                upload_and_clear,  # Perform upload
                inputs=[file_output_pkl, folder_name_pkl],
                outputs=result_output  # Capture the output to update the result textbox
            ).then(
                lambda result: gr.Info(result),  # Show final message using gr.Info
                inputs=result_output,  # Pass the final result to gr.Info
                outputs=None  # No need to update any component
            )
                
    state = gr.State([])  # Initialize state as an empty list to hold the chat history
    page_state = gr.State([])  # Initialize state to hold the page list
    excel_state = gr.State(empty_df)
    stop_flag = gr.State([False])
    
    def gradio_chat_ollama(history, msg_context, top_k, top_p, temperature, chat_history, check_error, validate, check_hallucination, grade_answer, page_state, excel_state):
        message = history[-1][0] if history else ""
        q = queue.Queue()

        # Submit the task to the thread pool
        executor.submit(handle_query, message, msg_context, history, top_k, top_p, temperature, chat_history, check_error, validate, check_hallucination, grade_answer, page_state, q)

        # Stream results from the queue
        while True:
            response = q.get()
            if response is None:  # Check for the end signal
                break
            history, page_state, excel_state = response # Capture history and other values
            yield history, history, page_state, excel_state

        return history, history, page_state, excel_state

    
    chat_msg = message_input.submit(add_message, [state, message_input], [state, chatbot], show_progress="hidden"
    ).then(
        lambda: "Processing...", None, status
    ).then(
        lambda: "", None, message_input  # Clear the textbox
    ).then(
        gradio_chat_ollama, [state, context_textbox, top_k, top_p, temp, chat_history, error_checkbox, validate_checkbox, hallucination_checkbox, check_answer_checkbox, page_state, excel_state], [chatbot, state, page_state, excel_state],  show_progress="hidden"
    ).then(
        lambda excel_state: update_excel_df(excel_state), [excel_state], [excel_df]
    ).then(
        lambda page_state: update_gallery(page_state), [page_state], [gallery]
    ).then(
        lambda: "Completed!", None, status
    )

    submit_btn.click(add_message, [state, message_input], [state, chatbot], show_progress="hidden"
    ).then(
        lambda: "Processing...", None, status
    ).then(
        lambda: "", None, message_input  # Clear the textbox
    ).then(
        gradio_chat_ollama, [state, context_textbox, top_k, top_p, temp, chat_history, error_checkbox, validate_checkbox, hallucination_checkbox, check_answer_checkbox, page_state, excel_state], [chatbot, state, page_state, excel_state],  show_progress="minimal"
    ).then(
        lambda excel_state: update_excel_df(excel_state), [excel_state], [excel_df]
    ).then(
        lambda page_state: update_gallery(page_state), [page_state], [gallery]
    ).then(
        lambda: "Completed!", None, status
    )
    
    undo_btn.click(undo_message, [state], [state, chatbot], queue=False)
    clear_btn.click(clear_messages, None, [state, chatbot], queue=False)
    stop_btn.click(stop_processing, [stop_flag], status, queue=False)

    

    demo.queue(default_concurrency_limit=10)
    # demo.launch(show_error=True)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
