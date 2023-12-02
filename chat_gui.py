import gradio as gr
import string
import time
from typing import Generator


models = ["llama", "palm", "bloom"]


allowedText = string.digits + string.ascii_letters + string.punctuation + " "


def chat(
    model: str, message: str, history: list[tuple[str, str]]
) -> Generator[str, None, None]:
    response = ""
    for s in "Hello, how are you?".split(" "):
        response += s + " "
        yield response
        time.sleep(0.3)


def process_chat(
    model: str, message: str, history: list[tuple[str, str]]
) -> Generator[tuple[str, list[tuple[str, str]]], None, None]:
    for response in chat(model, message, history):
        yield "", history + [(message, response)]


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


def repeat_chat(
    model: str, history: list[tuple[str, str]]
) -> Generator[list[tuple[str, str]], None, None]:
    message = history[-1][0]
    for response in chat(model, message, history[:-1]):
        yield history + [(message, response)]


def undo_chat(history: list[tuple[str, str]]) -> list[tuple[str, str]]:
    history.pop()
    return history


if __name__ == "__main__":
    with gr.Blocks() as app:
        modelsDropdown = gr.Dropdown(
            models, value=models[0], label="Model", container=False
        )
        chatbot = gr.Chatbot(bubble_full_width=False)
        with gr.Row():
            textInput = gr.Textbox(show_label=False, container=False, scale=5)
            uploadButton = gr.UploadButton(
                "Upload", file_count="single", file_types=["text"]
            )
        with gr.Row():
            repeatButton = gr.Button("Repeat")
            undoButton = gr.Button("Undo")
            clearButton = gr.Button("Clear")

        textInput.submit(
            process_chat,
            inputs=[modelsDropdown, textInput, chatbot],
            outputs=[textInput, chatbot],
        )
        uploadButton.upload(process_file, inputs=[uploadButton], outputs=[textInput])
        repeatButton.click(
            repeat_chat, inputs=[modelsDropdown, chatbot], outputs=[chatbot]
        )
        undoButton.click(undo_chat, inputs=[chatbot], outputs=[chatbot])
        clearButton.click(clear_chat, outputs=chatbot)

    app.launch(share=False, inbrowser=True)