
from utils import llama
# define the prompt
prompt = "Help me write a birthday card for my dear friend Andrew."
# pass prompt to the llama function, store output as 'response' then print
response = llama(prompt)
print(response)

prompt = "what is the best places to go in Puerto Rico if I want to do advanture?"
response = llama(prompt)
print(response)

prompt = "what is the situation with google stock? "
response = llama(prompt)
print(response)


# ------------------------------
# Set verbose to True to see the full prompt that is passed to the model.
prompt = "Help me write one sentence as a birthday card for my dear friend Andrew. "
response = llama(prompt, verbose=True)


# ------------------------------
# Chat vs. base models
#Ask model a simple question to #demonstrate the different behavior of #chat vs. base models.

### chat model
prompt = "What is the capital of France?"
response = llama(prompt, 
                 verbose=True,
                 model="togethercomputer/llama-2-7b-chat")

print (response)

### base model
prompt = "What is the capital of France?"
response = llama(prompt, 
                 verbose=True,
                 add_inst=False,
                 model="togethercomputer/llama-2-7b")
print (response)

prompt = "I am going to the store for milk."
response = llama(prompt, 
                 verbose=True,
                 add_inst=False,
                 model="togethercomputer/llama-2-7b")
print(response)
# ------------------------------

prompt = """
Help me write a birthday card for my dear friend Andrew.
Here are details about my friend:
He likes long walks on the beach and reading in the bookstore.
His hobbies include reading research papers and speaking at conferences.
His favorite color is light blue.
He likes pandas.
"""
# temperature=0.0 is deterministic, temperature=1.0 is random
response = llama(prompt, temperature=0.0)
print(response)


prompt = """
Help me write a birthday card for my dear friend Andrew.
Here are details about my friend:
He likes long walks on the beach and reading in the bookstore.
His hobbies include reading research papers and speaking at conferences.
His favorite color is light blue.
He likes pandas.
"""
response = llama(prompt, temperature=0.9)
print(response)

# limit the number of tokens generated
response = llama(prompt, temperature=0.9, max_tokens=20)
print(response)

# ------------------------------
with open ("TheVelveteenRabbit.txt", "r")  as file:
    text=file.read()


prompt = f"""
Give me a summary of the following text in 50 words:\n\n
{text}
"""
response = llama(prompt)
print(response)
# this output error below:
# `inputs` tokens + `max_new_tokens` must be <= 4097. Given: 3974 `inputs` tokens and 1024 `max_new_tokens`'
# allowed token to output 4097-3974 =123
response = llama(prompt, max_tokens=123)
print(response)

# ------------------------------
# notice in below that the llm does not remember the previous prompt


prompt = """
Help me write a birthday card for my dear friend Andrew.
Here are details about my friend:
He likes long walks on the beach and reading in the bookstore.
His hobbies include reading research papers and speaking at conferences.
His favorite color is light blue.
He likes pandas.
"""
# temperature=0.0 is deterministic, temperature=1.0 is random
response = llama(prompt)
print(response)

# we can see in the below that the model does not remember the previous prompt
prompt_2= "he also likes teaching. please add that to the birthday card."
response = llama(prompt_2)
print(response)

# in the next lesson, we will learn how to make the model remember the previous prompt

# ------------------------------
# ------------------------------
# session 3: "Multi-turn Conversations"
# ------------------------------

from utils import llama

prompt_1 = """
    What are fun activities I can do this weekend?
"""
response_1 = llama(prompt_1)
print(response_1)

prompt_2 = """
Which of these would be good for my health?
"""
response_2 = llama(prompt_2)
print(response_2)
# ------------------------------
# Constructing multi-turn prompts - You need to provide prior prompts and responses as part of the context of each new turn in the conversation.

chat_prompt = f""" <s> 
    [INST] {prompt_1} [/INST] {response_1} 
                    </s> 
    <s> [INST] {prompt_2} [/INST] """

chat_prompt = f"""
<s>[INST] {prompt_1} [/INST]
{response_1}
</s>
<s>[INST] {prompt_2} [/INST]
"""
print(chat_prompt)


# add_inst=False will remove the [INST] and [/INST] tokens from the prompt
# verbose = True will show the full prompt that is passed to the model
response_2=llama(chat_prompt,
                add_inst=False,
                verbose=True,
                )
print(response_2)

# ------------------------------
# doing the same as above which is multi-turn conversation but with a different way using helper function
from utils import llama_chat 
prompts=[prompt_1, prompt_2]
responses=[response_1]
response_2=llama_chat(prompts, responses, verbose=True)
print(response_2)

#---------------------------
# add more prompts and responses to the conversation
# replace prompt_3 with your own question!
prompt_3 = "tell me the single activity from the list provided that can burn the most calories."
prompts = [prompt_1, prompt_2, prompt_3]
responses = [response_1, response_2]

response_3 = llama_chat(prompts, responses, verbose=True)

print(response_3)

# ------------------------------
# ------------------------------
# Lesson 4: Prompt Engineering with Llama 2&3
# ------------------------------