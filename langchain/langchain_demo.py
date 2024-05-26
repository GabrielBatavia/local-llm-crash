from langchain_community.llms import CTransformers

llm = CTransformers(
    model="zoltanctoth/orca_mini_3B-GGUF", 
    model_file="orca-mini-3b.q4_0.gguf",
    model_type="llama2",
    max_new_token=20
)

# Prompt for the model
prompt = "Which city is the capital of Indonesia?"

# Invoke the model
try:
    answer = llm.invoke(prompt)
    print("Model Output: ", answer)
except Exception as e:
    print("Error during model invocation: ", str(e))
