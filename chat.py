import chainlit as cl
from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained(
    "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
)


def get_prompt(instruction: str, history: list[str] = None) -> str:
    system = "You are an AI assistant that follows instruction extremely well. Help as much as you can. Give short answers."
    prompt = f"### System:\n{system}\n\n### User:\n"
    if history is None:
        history = []
    if len(history) > 0:
        prompt += f"This is the conversation history: {''.join(history)}. Now answer this question: "
    prompt += f"{instruction}\n\n### Response:\n"
    return prompt


@cl.on_message
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history")
    msg = cl.Message(content="")
    await msg.send()

    prompt = get_prompt(message.content)
    response = ''
    for word in llm(prompt=prompt, stream=True):
        await msg.stream_token(word)
        response += word

    await msg.update()
    message_history.append(response)


@cl.on_chat_start
async def on_chat_start():
    global llm
    llm = AutoModelForCausalLM.from_pretrained(
        "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
    )

    cl.user_session.set("message_history", [])
    await cl.Message("Model initialized, How can I help you?").send()
