
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
# session 4: Prompt Engineering with Llama 2&3
# ------------------------------
from utils import llama, llama_chat
"""
In-Context Learning
Standard prompt with instruction
So far, you have been stating the instruction explicitly in the prompt:
"""
prompt = """
What is the sentiment of:
Hi Amit, thanks for the thoughtful birthday card!
"""
response = llama(prompt)
print(response)
# ------------------------------
"""
Zero-shot Prompting
Here is an example of zero-shot prompting.
You are prompting the model to see if it can infer the task from the structure of your prompt.
In zero-shot prompting, you only provide the structure to the model, but without any examples of the completed task.
"""
prompt = """
Message: Hi Amit, thanks for the thoughtful birthday card!
Sentiment:?
"""
response = llama(prompt)
print(response) #this is what gets printed: 'The sentiment in the message is "Thanks". '
# this does not look right. Lets use the bigget model (70B parameters) to see if it can do better.
response = llama(prompt, model="togethercomputer/llama-2-70b-chat")
print(response) # this is what gets printed: 'The sentiment in the message is GRATITUDE. The message expresses appreciation and thanks for the thoughtful birthday card received from Amit, which implies that the speaker values the effort and thoughtfulness put into the card.' (this is better)

# ------------------------------
"""
Few-shot Prompting
Here is an example of few-shot prompting.
In few-shot prompting, you not only provide the structure to the model, but also two or more examples.
You are prompting the model to see if it can infer the task from the structure, as well as the examples in your prompt.
"""
prompt = """
Message: Hi Dad, you're 20 minutes late to my piano recital!
Sentiment: Negative

Message: Can't wait to order pizza for dinner tonight
Sentiment: Positive

Message: Hi Amit, thanks for the thoughtful birthday card!
Sentiment: ?
"""
response = llama(prompt)
print(response) # this is what gets printed in few shots prompting - which is good, but it has the extra lines printed too: 
# 1. Message: Hi Dad, you're 20 minutes late to my piano recital!
# Sentiment: Negative
# 2. Message: Can't wait to order pizza for dinner tonight
# Sentiment: Positive
# 3. Message: Hi Amit, thanks for the thoughtful birthday card!
# Sentiment: Positive

# ------------------------------
"""
Specifying the Output Format
You can also specify the format in which you want the model to respond.
In the example below, you are asking to "give a one word response".
"""
prompt = """
Message: Hi Dad, you're 20 minutes late to my piano recital!
Sentiment: Negative

Message: Can't wait to order pizza for dinner tonight
Sentiment: Positive

Message: Hi Amit, thanks for the thoughtful birthday card!
Sentiment: ?

Give a one word response.
"""
response = llama(prompt)
print(response) # this is what gets printed which is not very good:
#  Sure! Here are the one-word responses for each message:
# 1. Negative: Disappointed
# 2. Positive: Excited
# 3. ? (Uncertain): Grateful

# use the bigger model to see if it can do better
response = llama(prompt, model="togethercomputer/llama-2-70b-chat")
print(response) # this is what gets printed which is perfect: 'Positive'

# ------------------------------
"""
Now, use the smaller model again, but adjust your prompt in order to help the model to understand what is being expected from it.
Restrict the model's output format to choose from positive, negative or neutral.
"""
prompt = """
Message: Hi Dad, you're 20 minutes late to my piano recital!
Sentiment: Negative

Message: Can't wait to order pizza for dinner tonight
Sentiment: Positive

Message: Hi Amit, thanks for the thoughtful birthday card!
Sentiment: ?

Give a one word response: positive, negative, or neutral.
"""
response = llama(prompt)
print(response) 
# this is what gets printed which is good, but it has the extra lines printed too: 
# 1. Message: Hi Dad, you're 20 minutes late to my piano recital!
# Sentiment: Negative
# 2. Message: Can't wait to order pizza for dinner tonight
# Sentiment: Positive
# 3. Message: Hi Amit, thanks for the thoughtful birthday card!
# Sentiment: Positive

# ------------------------------
"""
Role Prompting
Roles give context to LLMs what type of answers are desired.
Llama 2 often gives more consistent responses when provided with a role.
First, try standard prompt and see the response.
"""
prompt = """
How can I answer this question from my friend:
What is the meaning of life?
"""
response = llama(prompt)
print(response)

# Now, try it by giving the model a "role", and within the role, a "tone" using which it should respond with.
role = " you are an agent who is providing a philosophical answer in a poetic tone."
prompt = f"""
Role: {role}
How can I answer this question from my friend:
What is the meaning of life?
"""
response = llama(prompt)
print(response)

# another role
role = """
Your role is a life coach who gives advice to people about living a good life. You attempt to provide unbiased advice. You respond in the form of rhythmic poems.
"""

prompt = f"""
{role}
How can I answer this question from my friend:
What is the meaning of life?
"""
response = llama(prompt)
print(response)

# ------------------------------
"""
Summarization
Summarizing a large text is another common use case for LLMs. Let's try that!
"""
email = """
Dear Amit,

An increasing variety of large language models (LLMs) are open source, or close to it. The proliferation of models with relatively permissive licenses gives developers more options for building applications.

Here are some different ways to build applications based on LLMs, in increasing order of cost/complexity:

Prompting. Giving a pretrained LLM instructions lets you build a prototype in minutes or hours without a training set. Earlier this year, I saw a lot of people start experimenting with prompting, and that momentum continues unabated. Several of our short courses teach best practices for this approach.
One-shot or few-shot prompting. In addition to a prompt, giving the LLM a handful of examples of how to carry out a task — the input and the desired output — sometimes yields better results.
Fine-tuning. An LLM that has been pretrained on a lot of text can be fine-tuned to your task by training it further on a small dataset of your own. The tools for fine-tuning are maturing, making it accessible to more developers.
Pretraining. Pretraining your own LLM from scratch takes a lot of resources, so very few teams do it. In addition to general-purpose models pretrained on diverse topics, this approach has led to specialized models like BloombergGPT, which knows about finance, and Med-PaLM 2, which is focused on medicine.
For most teams, I recommend starting with prompting, since that allows you to get an application working quickly. If you’re unsatisfied with the quality of the output, ease into the more complex techniques gradually. Start one-shot or few-shot prompting with a handful of examples. If that doesn’t work well enough, perhaps use RAG (retrieval augmented generation) to further improve prompts with key information the LLM needs to generate high-quality outputs. If that still doesn’t deliver the performance you want, then try fine-tuning — but this represents a significantly greater level of complexity and may require hundreds or thousands more examples. To gain an in-depth understanding of these options, I highly recommend the course Generative AI with Large Language Models, created by AWS and DeepLearning.AI.

(Fun fact: A member of the DeepLearning.AI team has been trying to fine-tune Llama-2-7B to sound like me. I wonder if my job is at risk? 😜)

Additional complexity arises if you want to move to fine-tuning after prompting a proprietary model, such as GPT-4, that’s not available for fine-tuning. Is fine-tuning a much smaller model likely to yield superior results than prompting a larger, more capable model? The answer often depends on your application. If your goal is to change the style of an LLM’s output, then fine-tuning a smaller model can work well. However, if your application has been prompting GPT-4 to perform complex reasoning — in which GPT-4 surpasses current open models — it can be difficult to fine-tune a smaller model to deliver superior results.

Beyond choosing a development approach, it’s also necessary to choose a specific model. Smaller models require less processing power and work well for many applications, but larger models tend to have more knowledge about the world and better reasoning ability. I’ll talk about how to make this choice in a future letter.

Keep learning!

Andrew
"""

prompt = f"""
Give me a summary of the following email in 50 words:
email: {email}
"""
response = llama(prompt)
print(response)

# another prompt for that email
prompt = f"""
Summarize this email and extract some key points.
What did the author say about llama models?:

email: {email}
"""

response = llama(prompt)
print(response)

# ------------------------------
"""
Providing New Information in the Prompt¶

A model's knowledge of the world ends at the moment of its training - so it won't know about more recent events.
Llama 2 was released for research and commercial use on July 18, 2023, and its training ended some time before that date.
Ask the model about an event, in this case, FIFA Women's World Cup 2023, which started on July 20, 2023, and see how the model responses.
"""
prompt = """
Who won the 2023 Women's World Cup?
"""
response = llama(prompt)
print(response) #As you can see, the model still thinks that the tournament is yet to be played, even though we are now in 2024!


context = """
The 2023 FIFA Women's World Cup (Māori: Ipu Wahine o te Ao FIFA i 2023)[1] was the ninth edition of the FIFA Women's World Cup, the quadrennial international women's football championship contested by women's national teams and organised by FIFA. The tournament, which took place from 20 July to 20 August 2023, was jointly hosted by Australia and New Zealand.[2][3][4] It was the first FIFA Women's World Cup with more than one host nation, as well as the first World Cup to be held across multiple confederations, as Australia is in the Asian confederation, while New Zealand is in the Oceanian confederation. It was also the first Women's World Cup to be held in the Southern Hemisphere.[5]
This tournament was the first to feature an expanded format of 32 teams from the previous 24, replicating the format used for the men's World Cup from 1998 to 2022.[2] The opening match was won by co-host New Zealand, beating Norway at Eden Park in Auckland on 20 July 2023 and achieving their first Women's World Cup victory.[6]
Spain were crowned champions after defeating reigning European champions England 1–0 in the final. It was the first time a European nation had won the Women's World Cup since 2007 and Spain's first title, although their victory was marred by the Rubiales affair.[7][8][9] Spain became the second nation to win both the women's and men's World Cup since Germany in the 2003 edition.[10] In addition, they became the first nation to concurrently hold the FIFA women's U-17, U-20, and senior World Cups.[11] Sweden would claim their fourth bronze medal at the Women's World Cup while co-host Australia achieved their best placing yet, finishing fourth.[12] Japanese player Hinata Miyazawa won the Golden Boot scoring five goals throughout the tournament. Spanish player Aitana Bonmatí was voted the tournament's best player, winning the Golden Ball, whilst Bonmatí's teammate Salma Paralluelo was awarded the Young Player Award. England goalkeeper Mary Earps won the Golden Glove, awarded to the best-performing goalkeeper of the tournament.
Of the eight teams making their first appearance, Morocco were the only one to advance to the round of 16 (where they lost to France; coincidentally, the result of this fixture was similar to the men's World Cup in Qatar, where France defeated Morocco in the semi-final). The United States were the two-time defending champions,[13] but were eliminated in the round of 16 by Sweden, the first time the team had not made the semi-finals at the tournament, and the first time the defending champions failed to progress to the quarter-finals.[14]
Australia's team, nicknamed the Matildas, performed better than expected, and the event saw many Australians unite to support them.[15][16][17] The Matildas, who beat France to make the semi-finals for the first time, saw record numbers of fans watching their games, their 3–1 loss to England becoming the most watched television broadcast in Australian history, with an average viewership of 7.13 million and a peak viewership of 11.15 million viewers.[18]
It was the most attended edition of the competition ever held.
"""

prompt = f"""
Given the following context, who won the 2023 Women's World cup?
context: {context}
"""
response = llama(prompt)
print(response)


prompt = f"""
Given the following context, what is the highlight of information about Germany?
context: {context}
"""
response = llama(prompt)
print(response) # this is what gets printed which is wrong: ' Based on the provided context, the highlight of information about Germany is: * Germany is not mentioned anywhere in the text. '

# using bigger model for the same task as above.
response = llama(prompt, model="togethercomputer/llama-2-70b-chat")
print(response) # resposne which is good: The highlight of information about Germany in the given context is that Germany became the second nation to win both the women's and men's World Cup, after Spain won the 2023 FIFA Women's World Cup. Germany won the men's World Cup in 2003.

# ------------------------------
"""
Try it Yourself!
Try asking questions of your own! Modify the code below and include your own context to see how the model responds:
"""
# general schema
context = """
<paste context in here>
"""
query = "<your query here>"

prompt = f"""
Given the following context,
{query}

context: {context}
"""

# example - information about Stephen Hawking https://en.wikipedia.org/wiki/Stephen_Hawking
with open("Hawking.txt", "r") as file:
    context=file.read()

query = "1- who this text is about?; 2-where did he study?; 3- what was his major?; 4- what was his thesis about?; 5- what was the title of his essay that won the Adams Prize?"

prompt = f"""
Given the following context,
{query}

context: {context}
"""

response = llama(prompt, model="togethercomputer/llama-2-7b-chat")
print(response)

# ------------------------------
"""
Chain-of-thought Prompting¶
LLMs can perform better at reasoning and logic problems if you ask them to break the problem down into smaller steps. This is known as chain-of-thought prompting.
"""

prompt = """
15 of us want to go to a restaurant.
Two of them have cars
Each car can seat 5 people.
Two of us have motorcycles.
Each motorcycle can fit 2 people.

Can we all get to the restaurant by car or motorcycle?
"""
response = llama(prompt)
print(response) # answer was yes with a wrong explanation

response = llama(prompt, model="togethercomputer/llama-2-70b-chat") 
print(response) # using bigger model, answer was correct. it said no. 

# ------------------------------
#Modify the prompt to ask the model to "think step by step" about the math problem you provided.
prompt = """
15 of us want to go to a restaurant.
Two of them have cars
Each car can seat 5 people.
Two of us have motorcycles.
Each motorcycle can fit 2 people.

Can we all get to the restaurant by car or motorcycle?

Think step by step.
"""
response = llama(prompt, model="togethercomputer/llama-2-70b-chat")
print(response) # both 7b and 70b still provided wrong answer. 


# ------------------------------
#Provide the model with additional instructions.

prompt = """
15 of us want to go to a restaurant.
Two of them have cars
Each car can seat 5 people.
Two of us have motorcycles.
Each motorcycle can fit 2 people.

Can we all get to the restaurant by car or motorcycle?

Think step by step.
Explain each intermediate step.
Only when you are done with all your steps,
provide the answer based on your intermediate steps.
"""
response = llama(prompt, model="togethercomputer/llama-2-70b-chat")
print(response) # Interestingly, 7b is correct, 70b was incorrect!

# ------------------------------
#The order of instructions matters!
# Ask the model to "answer first" and "explain later" to see how the output changes.
prompt = """
15 of us want to go to a restaurant.
Two of them have cars
Each car can seat 5 people.
Two of us have motorcycles.
Each motorcycle can fit 2 people.

Can we all get to the restaurant by car or motorcycle?
Think step by step.
Provide the answer as a single yes/no answer first.
Then explain each intermediate step.
"""

response = llama(prompt, model="togethercomputer/llama-2-70b-chat")
print(response) #7b says yes which is wrong, 70b says no which is correct

# ------------------------------
# ------------------------------
# session 5: Comparing different llama 2 and 3 models
# ------------------------------
from utils import llama, llama_chat

"""
Task 1: Sentiment Classification
Compare the models on few-shot prompt sentiment classification.
You are asking the model to return a one word response.
"""

prompt = '''
Message: Hi Amit, thanks for the thoughtful birthday card!
Sentiment: Positive
Message: Hi Dad, you're 20 minutes late to my piano recital!
Sentiment: Negative
Message: Can't wait to order pizza for dinner tonight!
Sentiment: ?

Give a one word response for the ? mark. Response should be positive, negative, or neutral.
'''
response = llama(prompt, model="togethercomputer/llama-2-70b-chat")
print(response)

# ------------------------------
"""
Task 2: Summarization¶
Compare the models on summarization task.
This is the same "email" as the one you used previously in the course.
"""

email = """
Dear Amit,

An increasing variety of large language models (LLMs) are open source, or close to it. The proliferation of models with relatively permissive licenses gives developers more options for building applications.

Here are some different ways to build applications based on LLMs, in increasing order of cost/complexity:

Prompting. Giving a pretrained LLM instructions lets you build a prototype in minutes or hours without a training set. Earlier this year, I saw a lot of people start experimenting with prompting, and that momentum continues unabated. Several of our short courses teach best practices for this approach.
One-shot or few-shot prompting. In addition to a prompt, giving the LLM a handful of examples of how to carry out a task — the input and the desired output — sometimes yields better results.
Fine-tuning. An LLM that has been pretrained on a lot of text can be fine-tuned to your task by training it further on a small dataset of your own. The tools for fine-tuning are maturing, making it accessible to more developers.
Pretraining. Pretraining your own LLM from scratch takes a lot of resources, so very few teams do it. In addition to general-purpose models pretrained on diverse topics, this approach has led to specialized models like BloombergGPT, which knows about finance, and Med-PaLM 2, which is focused on medicine.
For most teams, I recommend starting with prompting, since that allows you to get an application working quickly. If you’re unsatisfied with the quality of the output, ease into the more complex techniques gradually. Start one-shot or few-shot prompting with a handful of examples. If that doesn’t work well enough, perhaps use RAG (retrieval augmented generation) to further improve prompts with key information the LLM needs to generate high-quality outputs. If that still doesn’t deliver the performance you want, then try fine-tuning — but this represents a significantly greater level of complexity and may require hundreds or thousands more examples. To gain an in-depth understanding of these options, I highly recommend the course Generative AI with Large Language Models, created by AWS and DeepLearning.AI.

(Fun fact: A member of the DeepLearning.AI team has been trying to fine-tune Llama-2-7B to sound like me. I wonder if my job is at risk? 😜)

Additional complexity arises if you want to move to fine-tuning after prompting a proprietary model, such as GPT-4, that’s not available for fine-tuning. Is fine-tuning a much smaller model likely to yield superior results than prompting a larger, more capable model? The answer often depends on your application. If your goal is to change the style of an LLM’s output, then fine-tuning a smaller model can work well. However, if your application has been prompting GPT-4 to perform complex reasoning — in which GPT-4 surpasses current open models — it can be difficult to fine-tune a smaller model to deliver superior results.

Beyond choosing a development approach, it’s also necessary to choose a specific model. Smaller models require less processing power and work well for many applications, but larger models tend to have more knowledge about the world and better reasoning ability. I’ll talk about how to make this choice in a future letter.

Keep learning!

Andrew
"""

prompt = f"""
Summarize this email and extract some key points.

What did the author say about llama models?
'''
email: {email}
'''
"""
response_7b = llama(prompt, model="togethercomputer/llama-2-7b-chat")
print(response_7b)

response_70b = llama(prompt, model="togethercomputer/llama-2-7b-chat")
print(response_70b)

# ------------------------------
"""
Model-Graded Evaluation: Summarization¶
Interestingly, you can ask a LLM to evaluate the responses of other LLMs.
This is known as Model-Graded Evaluation.
Create a prompt that will evaluate these three responses using 70B parameter chat model (llama-2-70b-chat).
In the prompt, provide the "email", "name of the models", and the "summary" generated by each model.
"""
# my own prompt
prompt_ = f"""
This is the prompt we used: {prompt}
and these are two responses we got from two different models:
resosne_1: {response_7b} and response2: {response_70b}

Evaluate the performance of the two models"""

response = llama(prompt_, model="togethercomputer/llama-2-70b-chat")
print(response)

# another way to write prompt for evaluationv from the course.
prompt = f"""
Given the original text denoted by `email`
and the name of several models: `model:<name of model>
as well as the summary generated by that model: `summary`

Provide an evaluation of each model's summary:
- Does it summarize the original text well?
- Does it follow the instructions of the prompt?
- Are there any other interesting characteristics of the model's output?

Then compare the models based on their evaluation \
and recommend the models that perform the best.

email: ```{email}`

model: llama-2-7b-chat
summary: {response_7b}


model: llama-2-70b-chat
summary: {response_70b}

"""

response_eval = llama(prompt,
                model="togethercomputer/llama-2-70b-chat")
print(response_eval)

# ------------------------------
"""
Task 3: Reasoning¶
Compare the three models' performance on reasoning tasks.
"""

context = """
Jeff and Tommy are neighbors

Tommy and Eddy are not neighbors
"""
query = """
Are Jeff and Eddy neighbors?
"""

prompt = f"""
context: ''' {context} '''
query: ''' {query} '''
given the context, please answer the question in the query. If there is not enough informaton to answer, please say "I do not have enough information to answer this question."
"""

response_7b = llama(prompt, model="togethercomputer/llama-2-7b-chat")
print(response_7b)

response_13b = llama(prompt, model="togethercomputer/llama-2-13b-chat")
print(response_13b)

response_70b = llama(prompt, model="togethercomputer/llama-2-70b-chat")
print(response_70b)

# ------------------------------
"""
Model-Graded Evaluation: Reasoning¶
Again, ask a LLM to compare the three responses.
Create a prompt that will evaluate these three responses using 70B parameter chat model (llama-2-70b-chat).
In the prompt, provide the context, query,"name of the models", and the "response" generated by each model.
"""

prompt_ = f"""
Given the context as ''' {context} '''
and the query as ''' {query} '''

as well as the response generated by the models:
model_7b (llama-2-7b-chat): ''' {response_7b} '''
model_13b (llama-2-13b-chat): ''' {response_13b} '''
model_70b (llama-2-70b-chat): ''' {response_70b} '''

Provide an evaluation of each model's response:
- Does it answer the query accurately?
- Does it provide a contradictory response?
- Are there any other interesting characteristics of the model's output?

Then compare the models based on their evaluation \
and recommend the models that perform the best.
"""

response_eval = llama(prompt_,
                model="togethercomputer/llama-2-70b-chat")
print(response_eval)