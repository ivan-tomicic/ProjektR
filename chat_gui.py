import gradio as gr
import string
from typing import Generator
from main import create_qa_instance

allowedText = string.digits + string.ascii_letters + string.punctuation + " "


qa = create_qa_instance()


def chat(message: str, history: list[tuple[str, str]]
) -> Generator[str, None, None]:
    output = qa(message)
    response = output['result']
    yield response


def process_chat(message: str, history: list[tuple[str, str]]) -> Generator[tuple[str, list[tuple[str, str]]], None, None]:
    # First, add the user's message to the history and yield it immediately
    updated_history = history + [(message, "")]
    yield "", updated_history

    # Then, process the chat as before and yield the response
    for response in chat(message, history):
        yield "", updated_history[:-1] + [(message, response)]
def process_file(filePath: str) -> str:
    with open(filePath, "r", encoding="utf-8") as file:
        text = " ".join(
            map(
                lambda line: line.encode("ascii", errors="replace").decode(),
                file.readlines(),
            )
        )
    cleanText = "".join(filter(lambda x: x in allowedText, text))
    return cleanText


def clear_chat() -> list[tuple[str, str]]:
    return []


def repeat_chat(history: list[tuple[str, str]]
) -> Generator[list[tuple[str, str]], None, None]:
    message = history[-1][0]
    for response in chat(message, history[:-1]):
        yield history + [(message, response)]


def undo_chat(history: list[tuple[str, str]]) -> list[tuple[str, str]]:
    history.pop()
    return history


if __name__ == "__main__":
    with gr.Blocks() as app:
        chatbot = gr.Chatbot(bubble_full_width=False)
        with gr.Row():
            textInput = gr.Textbox(show_label=False, container=False, scale=5)
        with gr.Row():
            repeatButton = gr.Button("Repeat")
            undoButton = gr.Button("Undo")
            clearButton = gr.Button("Clear")

        textInput.submit(
            process_chat,
            inputs=[textInput, chatbot],
            outputs=[textInput, chatbot],
        )
        repeatButton.click(
            repeat_chat, inputs=[chatbot], outputs=[chatbot]
        )
        undoButton.click(undo_chat, inputs=[chatbot], outputs=[chatbot])
        clearButton.click(clear_chat, outputs=chatbot)

    app.launch(share=False, inbrowser=True)
