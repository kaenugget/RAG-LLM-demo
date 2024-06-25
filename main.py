import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import pandas as pd
import pickle
import gradio as gr
from theme import JS_LIGHT_THEME, CSS
from logger import Logger
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager


local_llm="llama3"
logger_instance = Logger("logfile.txt")
ollama_host = os.getenv("OLLAMA_HOST", "ollama-container")
ollama_port = os.getenv("OLLAMA_PORT", "11434")
ollama_url = f"http://{ollama_host}:{ollama_port}"

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
llm = ChatOllama(base_url=ollama_url, model=local_llm, format="json", temperature = 0)
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
llm = ChatOllama(base_url=ollama_url, model=local_llm, format="json", temperature = 0)
prompt= PromptTemplate(
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
error_identifier = prompt | llm | JsonOutputParser()

def extract_error_code(question):
    error_code_JSON = error_identifier.invoke({"question": question})
    print("error_code_JSON: ", error_code_JSON)
    return error_code_JSON["error_code"]

### Document Retrieval and Combination
def retrieve_and_combine_documents(question):
    error_code_list = extract_error_code(question)
    combined_docs = []

    if error_code_list != []:
        print(f"Extracted Error Code: {error_code_list}")

        with open('db/healthcare.txt', 'r') as file1:
            lines = file1.readlines()

        with open('db/UCFMSG_validation_codes.txt', 'r') as file2:
            lines += file2.readlines()

        for error_index in range(0, len(error_code_list)):

            matched_snippets = []
            skip_lines = 0
            skip = False
            doc_count = 0

            for i in range(0, len(lines)):

                skip_lines -= 1

                if error_code_list[error_index] in lines[i] and not skip:
                    doc_count += 1
                    start = max(0, i - 15)
                    end = min(len(lines), i + 7)
                    snippet = ''.join(lines[start:end])

                    header = f"Document {doc_count} for error {error_code_list[error_index]}:\n"
            
                    # Prepend the header to the snippet
                    snippet_with_header = header + snippet
                    
                    # Append the modified snippet to the matched snippets list
                    matched_snippets.append(snippet_with_header)

                    skip_lines = 10
                    skip = True

                    if len(matched_snippets) >= 4:
                        break

                if skip_lines <= 0:
                    skip = False
                    skip = 0


            result = matched_snippets[:4]
            combined_docs.append(result)
            print(result)

            if result != []:
                print("Exact match found in healthcare.txt for error", error_code_list[error_index])
                print("-------------------------------")
            else:
                 print(f"No Error Code {error_code_list[error_index]} found in the documents.")

    else:
        print("No Error Code found in the question.")

    return combined_docs, error_code_list


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
hallucination_grader = prompt | llm | JsonOutputParser()


### Answer grader
llm = ChatOllama(base_url=ollama_url, model=local_llm, format="json", temperature=0)
prompt= PromptTemplate(
    template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> 
    You are a grader assessing whether an answer is addressing the question properly.\n
    Give a binary score 'yes' or 'no' score to indicate whether the answer addresses the question well.\n
    Provide the binary score as a JSON with a single ey 'score' and no preamble or explanation.\n
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
    You are an assistant for formatting tasks. Do not alter the content or add in any new content.\n
    You are only in charge of reformatting the context given to you. \n
    The context is parsed from a pdf and might be part of code, descriptions or tables. Do your best to format it nicely and do not format everything into code by adding "```" at the start.\n
    Only the definition blocks should be formatted into code by adding "```" at the front and back for all "definition" blocks which can appear more than once.\n  
    At the end of the context, there should be a JSON string, format it into a table if suitable and ensure it is not in a '<code>' block or within a "```".\n
    Format the given "context" variable and return it in the format "Context: (formatted context)". \n
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
    
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question}
    Context: {context}
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question","context"],
)

prompt_template_history= PromptTemplate(
    template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> 
    You are an assistant for question-answering tasks. Often, the context might be in code, descriptions, tables and JSON, do your best to analyse them.\n
    You can try to identify the specific things mentioned in the question and work from there.\n
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. \n
    If you dont know, also mention what you could discern specifically from the context as well as what you think you might need to answer the given question.\n
    If your Chat history is provided to you, please take into account the history of the chat and answer with it in mind.\n
    The format of the chat history is a list of lists. An example is [["my question", "your response"],["my question", None]].\n
    
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question}
    Context: {context}
    Chat History: {history}
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question","context", "history"],
)

llm_main = ChatOllama(base_url=ollama_url, model=local_llm, temperature=0, callbacks=callback_manager)


def format_history(msg, history):
    chat_history = [{"role": "system", "content": "hi"}]
    for query, response in history:
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": response})
    chat_history.append({"role": "user", "content": msg})
    return chat_history

def generate_response(message, history, top_k, top_p, temperature, chat_history=False, validate=False, check_hallucination=False, check_context=False):
    if history is None:
        history = []

    print("history", history)
    validation_agent = retrieval_grader
    hallucination_agent = hallucination_grader

    # # Retrieve and combine documents
    combined_docs, error_code_list = retrieve_and_combine_documents(message)

    #Load the saved DataFrame
    loaded_df = load_df("db/MediclaimFS.pkl")

    #search the loaded DataFrame
    search_col = 'Error Code'

    excel_result = ""
    if error_code_list != []:
        for error_index in range(0, len(error_code_list)):
            excel_searched = search_df(loaded_df, search_col , error_code_list[error_index])
            if excel_searched:
                excel_result += excel_searched
                excel_result += "\n\n"

    combined_docs_string = "\n\n".join([str(doc) for doc in combined_docs]) + "\n\n" + "From Excel:" + "\n" + excel_result

    print(combined_docs_string)

    # Generate the main response
    if chat_history:
        prompt = prompt_template_history.format(question=message, context=combined_docs_string, history=history)
        print("prompting with history: ", history)
    else:
        prompt = prompt_template.format(question=message, context=combined_docs_string) 

    response = llm_main.stream(prompt, top_k = int(top_k), top_p = float(top_p), temperature = float(temperature))
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
        documents = validation_agent.invoke({"question": message, "document": combined_docs_string})
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

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="slate"),
    js=JS_LIGHT_THEME,
    css=CSS,
) as demo:
    gr.Markdown("## Day2Ops Chatbot 🤖")

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
                        choices=["eng"],
                        value="eng",
                        interactive=True,
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
                    chat_history = gr.Checkbox(label="Enable message memory (takes longer the more you query)")
                    context_checkbox = gr.Checkbox(label="Show Context (takes longer)")
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
                    avatar_images=("assets/user.png","assets/bot.png")
                )
                with gr.Row():
                    message_input = gr.Textbox(
                        placeholder="Enter message...", 
                        show_label=False, 
                        scale=3, 
                        lines=1
                    )

                    submit_btn = gr.Button("Submit",scale=1)
                

                with gr.Row(variant="panel"):
                    undo_btn = gr.Button(value="Undo", min_width=20)
                    clear_btn = gr.Button(value="Clear", min_width=20)
                    reset_btn = gr.Button(value="Reset", min_width=20)


    with gr.Tab("Settings"):
        with gr.Row():
            top_k = gr.Slider(0.0,100.0, label="top_k", value=40, info="Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40)")
            top_p = gr.Slider(0.0,1.0, label="top_p", value=0.9, info=" Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)")
            temp = gr.Slider(0.0,2.0, label="temperature", value=0.8, info="The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.8)")


    with gr.Tab("Output"):
        with gr.Row(variant="panel"):
            log = gr.Code(
                label="", language="markdown", interactive=False, lines=30
            )
            demo.load(
                logger_instance.read_logs,
                outputs=[log],
                every=1,
                show_progress="hidden",
                scroll_to_output=True,
            )
                
    state = gr.State([])  # Initialize state as an empty list to hold the chat history

    def gradio_chat_ollama(history, top_k, top_p, temperature, chat_history, validate, check_hallucination, check_context):
        message = history[-1][0] if history else ""
        generator = generate_response(message, history, top_k, top_p, temperature, chat_history, validate, check_hallucination, check_context)
        for response in generator:
            yield response

    chat_msg = message_input.submit(add_message, [state, message_input], [state, chatbot]
    ).then(
        lambda: "Processing...", None, status
    ).then(
        gradio_chat_ollama, [state, top_k, top_p, temp, chat_history, validate_checkbox, hallucination_checkbox, context_checkbox], [chatbot, state]
    ).then(
        lambda: "", None, message_input  # Clear the textbox
    ).then(
        lambda: "Completed!", None, status
    )

    submit_btn.click(add_message, [state, message_input], [state, chatbot]
    ).then(
        lambda: "Processing...", None, status
    ).then(
        gradio_chat_ollama, [state, top_k, top_p, temp, chat_history, validate_checkbox, hallucination_checkbox, context_checkbox], [chatbot, state]
    ).then(
        lambda: "", None, message_input  # Clear the textbox
    ).then(
        lambda: "Completed!", None, status
    )
    
    clear_btn.click(lambda: [], None, chatbot, queue=False)  # Clear the chat

    

    demo.queue()
    # demo.launch(show_error=True)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
