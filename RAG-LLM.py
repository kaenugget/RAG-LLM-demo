import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import pandas as pd
import pickle
import gradio as gr
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager


local_llm="llama3"

###Extract from PDF
def initialize_vector_store():
    # Step 1: Read the contents of documents.txt
    with open('db/documents.txt', 'r') as file:
        documents = file.readlines()

    # Ensure the documents are in the right format (list of strings)
    documents = [doc.strip() for doc in documents]

    # Step 2: Initialize the embedding model
    embedding = FastEmbedEmbeddings()

    # Step 3: Initialize the Chroma vector store with the embedding function
    persist_directory = "db"
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embedding)

    # Step 4: Add documents to the vector store
    for document in documents:
        vector_store.add_texts([document])

    return vector_store

def get_retriever():
    persist_directory = "db"
    
    # Check if vector store already exists
    if not os.path.exists(persist_directory):
        vector_store = initialize_vector_store()
    else:
        # Initialize Chroma vector store without re-adding texts
        embedding = FastEmbedEmbeddings()
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=embedding)

    # Step 5: Create the retriever
    retriever = vector_store.as_retriever()
    return retriever

retriever = get_retriever()
print("Retriever created successfully!")


### Extract from Excel
def parse_and_save_excel(file_path, sheet_name, save_path):
    """
    Parse a specific sheet in the Excel file and save the DataFrame to a file.
    
    :param file_path: Path to the Excel file.
    :param sheet_name: Name or index of the sheet to parse.
    :param save_path: Path to save the DataFrame.
    """
    # Load the specific sheet into a DataFrame
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    # Save the DataFrame to a file
    with open(save_path, 'wb') as file:
        pickle.dump(df, file)
    print(f"DataFrame from sheet '{sheet_name}' saved to {save_path}")

def load_df(save_path):
    """
    Load the saved DataFrame from a file.
    
    :param save_path: Path to the saved DataFrame.
    :return: Loaded DataFrame.
    """
    with open(save_path, 'rb') as file:
        df = pickle.load(file)
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
        return filtered_df.to_json(orient='records')
    else:
        print(f"No match found for {search_value} in column {search_column}")
        return None


### Document grader
llm = ChatOllama(model=local_llm, format="json", temperature = 0)
prompt= PromptTemplate(
    template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> 
    You are a grader assessing relevance of a retrieved document to a user question. 
    If the document contains keywords related to the user question, grade it as relevant.
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate wherther the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the retrieved document: \n\n {document}\n\n
    Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question","document"],
)
retrieval_grader = prompt | llm | JsonOutputParser()


###Error identifier
#LLM
llm = ChatOllama(model=local_llm, format="json", temperature = 0)
prompt= PromptTemplate(
    template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> 
    You are an indentifier.
    Identify the error code from the given question / sentence. An example could be VCA373 . Error codes usually start with a few capital alphabets followed by a few numbers
    It does not need to be a stringent test. The goal is to identify only the error code and return that. \n
    Return the error code if it is present, else return an empty string ''. \n
    Provide the error code as a JSON with a single key 'error_code' and no preamble or explanation.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the sentence: \n\n {question}\n\n
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question"],
)
error_identifier = prompt | llm | JsonOutputParser()

def extract_error_code(question):
    error_code_JSON = error_identifier.invoke({"question": question})
    return error_code_JSON["error_code"]

### Document Retrieval and Combination
def retrieve_and_combine_documents(question, retriever):
    error_code = extract_error_code(question)
    combined_docs = []

    if error_code:
        print(f"Extracted Error Code: {error_code}")

        with open('db/documents.txt', 'r') as file:
            lines = file.readlines()

        matched_snippets = set()
        skip_lines = 0
        skip = False
        for i, line in enumerate(lines):

            skip_lines -= 1

            if error_code in line and not skip:
                start = max(0, i - 15)
                end = min(len(lines), i + 11)
                snippet = ''.join(lines[start:end])
                matched_snippets.add(snippet)

                skip_lines = 10
                skip = True

                if len(matched_snippets) >= 3:
                    break

            if skip_lines <= 0:
                skip = False
                skip = 0

        combined_docs = list(matched_snippets)[:3]
        print("Exact match found in documents.txt")
        print("-------------------------------")
        for idx, doc in enumerate(combined_docs):
            print(f"combined_doc {idx}: {doc}")

    else:
        print("No Error Code found in the question.")
        # Retrieve documents only for the general question if no error code is found
        combined_docs = retriever(question)

    return combined_docs


### Hallucination checker
llm = ChatOllama(model=local_llm, format="json", temperature=0)
prompt= PromptTemplate(
    template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> 
    You are a grader assessing whether an answer is grounded in / supported by a set of facts.
    Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts.
    Provide the binary score as a JSON with a single ey 'score' and no preamble or explanation.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here are the facts: 
    \n-------\n 
    {documents}
    \n-------\n
    Here is the answer: {generation} <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["generation","document"],
)
hallucination_grader = prompt | llm | JsonOutputParser()


### Answer grader
llm = ChatOllama(model=local_llm, format="json", temperature=0)
prompt= PromptTemplate(
    template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> 
    You are a grader assessing whether an answer is addressing the question properly.
    Give a binary score 'yes' or 'no' score to indicate whether the answer addresses the question well.
    Provide the binary score as a JSON with a single ey 'score' and no preamble or explanation.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question}
    Here is the answer: {generation} <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["generation","question"],
)
answer_grader = prompt | llm | JsonOutputParser()


### Context formatter
context_formatting_template = PromptTemplate(
    template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> 
    You are an assistant for formatting tasks. Do not alter the content or add in any new content.
    You are only in charge of reformatting the context given to you. 
    The context is parsed from a pdf and might be part of code, descriptions or tables. Do your best to format it nicely into markdown.
    The context could also be in the form of a JSON, format it into a table if suitable.
    Format the given "context" variable and return it in the format "Context: (formatted context)". 
    If no "context" variable is given, just return an empty string.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Context: {context}
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["context"],
)
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm_formatter = ChatOllama(model=local_llm, temperature=0, callbacks=callback_manager)



prompt_template= PromptTemplate(
    template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> 
    You are an assistant for question-answering tasks. Often, the context might be in code, descriptions or tables, do your best to format it nicely and then analyse them.
    You can try to identify the specific things mentioned in the question and work from there.
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
    If you dont know, also mention what you could discern specifically from the context as well as what you think you might need to answer the given question.
    
    
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question}
    Context: {context}
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question","context"],
)
llm_main = ChatOllama(model=local_llm, temperature=0, callbacks=callback_manager)


def format_history(msg, history):
    chat_history = [{"role": "system", "content": "hi"}]
    for query, response in history:
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": response})
    chat_history.append({"role": "user", "content": msg})
    return chat_history

def generate_response(message, history, validate=False, check_hallucination=False, check_context=False):
    if history is None:
        history = []

    validation_agent = retrieval_grader
    hallucination_agent = hallucination_grader

    # # Retrieve and combine documents
    # combined_docs = retrieve_and_combine_documents(message, retriever)
    # combined_docs_string = "\n\n".join([str(doc) for doc in combined_docs])

    error_code = extract_error_code(message)
    combined_docs_string = search_df(loaded_df, search_col , error_code)
    print(combined_docs_string)

    # Generate the main response
    prompt = prompt_template.format(question=message, context=combined_docs_string)
    response = llm_main.stream(prompt)
    result = ""
    for partial_answer in response:
        result += partial_answer.content
        history[-1][1] = result
        yield history, gr.update()  # Yield the main response first

    # Append the context check result to the main response if requested
    if check_context:
        context_prompt = context_formatting_template.format(context=combined_docs_string)
        context_response = llm_formatter.stream(context_prompt)
        result += "\n\nFormatted Context:\n"
        for partial_context in context_response:
            result += partial_context.content
            history[-1][1] = result
            yield history, gr.update()

    # Perform validation check if requested
    if validate:
        documents = validation_agent.invoke({"question": message, "document": doc_text})
        validation_result = "Validation Check: Documents validated successfully."
        result += "\n\n" + validation_result
        history[-1][1] = result
        yield history, gr.update()

    # Perform hallucination check if requested
    if check_hallucination:
        hallucination_result = hallucination_agent.invoke({"documents": combined_docs, "generation": result})
        if hallucination_result:
            hallucination_result_text = "\n\nHallucination Check:\n" + hallucination_result
            result += hallucination_result_text
            history[-1][1] = result
            yield history, gr.update()

    history[-1][1] = result
    yield history, gr.update()

def add_message(history, message):
    if message is not None:
        history.append([message, None])
    return history, gr.update(value="")

with gr.Blocks() as demo:
    chatbot = gr.Chatbot([], elem_id="chatbot", bubble_full_width=False)
    message_input = gr.Textbox(placeholder="Enter message...", show_label=False)
    context_checkbox = gr.Checkbox(label="Show Context")
    validate_checkbox = gr.Checkbox(label="Validate Documents")
    hallucination_checkbox = gr.Checkbox(label="Check Hallucination")
    

    with gr.Row():
        top_k = gr.Slider(0.0,100.0, label="top_k", value=40, info="Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40)")
        top_p = gr.Slider(0.0,1.0, label="top_p", value=0.9, info=" Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)")
        temp = gr.Slider(0.0,2.0, label="temperature", value=0.8, info="The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.8)")


    clear_btn = gr.Button("Clear")
    state = gr.State([])  # Initialize state as an empty list to hold the chat history

    def gradio_chat_ollama(history, validate, check_hallucination, check_context):
        message = history[-1][0] if history else ""
        generator = generate_response(message, history, validate, check_hallucination, check_context)
        for response in generator:
            yield response

    chat_msg = message_input.submit(add_message, [state, message_input], [state, chatbot]).then(
        gradio_chat_ollama, [state, validate_checkbox, hallucination_checkbox, context_checkbox], [chatbot, state]
    )
    clear_btn.click(lambda: [], None, chatbot, queue=False)  # Clear the chat

    demo.queue()
    demo.launch(show_error=True, auth=("day2ops", "nphc"))