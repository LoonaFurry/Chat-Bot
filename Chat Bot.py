import discord
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import random

# Load the pre-trained Mistral-7B-Instruct-v0.2 language model and tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move the model to the GPU if available
if torch.cuda.is_available():
    model.to('cuda')

# Discord bot setup
intents = discord.Intents.all()
intents.typing = False
intents.presences = False

client = discord.Client(intents=intents)

# Function to generate a response from the chatbot
def generate_response(prompt, max_length=1024, num_return_sequences=1):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    if torch.cuda.is_available():
        input_ids = input_ids.to('cuda')
    output = model.generate(input_ids, max_length=max_length, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=num_return_sequences)
    responses = [tokenizer.decode(output[i], skip_special_tokens=True) for i in range(num_return_sequences)]
    return responses

# Function to fine-tune the language model
def fine_tune_model(prompt, response):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output_ids = tokenizer.encode(response, return_tensors='pt')
    if torch.cuda.is_available():
        input_ids = input_ids.to('cuda')
        output_ids = output_ids.to('cuda')
    loss = model(input_ids, labels=output_ids)[0]
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Discord bot event handlers
@client.event
async def on_ready():
    print(f'Logged in as {client.user}')

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    # Check if the bot is mentioned
    if client.user.mentioned_in(message):
        prompt = message.content
        responses = generate_response(prompt, num_return_sequences=3)
        best_response = max(responses, key=len)

        # Add some conversational flair
        if "how are you" in prompt.lower():
            responses = ["I'm doing well, thanks for asking!", "I'm great, thanks for checking in!", "I'm feeling wonderful today!"]
            best_response = random.choice(responses)
        elif "what's up" in prompt.lower():
            responses = ["Not much, just chatting with you!", "Just hanging out, how about you?", "Not too much, how about you?"]
            best_response = random.choice(responses)
        elif "hello" in prompt.lower():
            responses = ["Hello there!", "Hi there!", "Hey there!"]
            best_response = random.choice(responses)

        await message.channel.send(best_response)

        # Fine-tune the model with the user's message and the bot's response
        fine_tune_model(prompt, best_response)

# Discord bot token
TOKEN = 'Your-Discord-Token'

# Set up the optimizer for fine-tuning
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Start the bot
client.run(TOKEN)


