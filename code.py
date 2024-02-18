!pip install openai==0.27.0 gtts
!pip install openai gtts pydub
!pip install SpeechRecognition
!apt-get install -y python3-pyaudio
!pip install SpeechRecognition

from gtts import gTTS
from IPython.display import Audio, display
import openai
import requests

# Set your OpenAI API key
openai.api_key = 'your open ai api key'

def generate_text_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )
    return response['choices'][0]['message']['content'].strip()

def text_to_speech(text):
    tts = gTTS(text, lang='en')
    tts.save("output.mp3")
    display(Audio("output.mp3", autoplay=True))

def generate_image_response(prompt):
    # Make a POST request to the ClipDrop API endpoint
    r = requests.post('https://clipdrop-api.co/text-to-image/v1',
      files = {
          'prompt': (None, prompt, 'text/plain')
      },
      headers = { 'x-api-key':'your openai api key'}
    )

    # Check if the request was successful
    if r.ok:
        # Save the image to a file
        with open('output_image.jpg', 'wb') as f:
            f.write(r.content)
        print("Image saved as 'output_image.jpg'")
    else:
        r.raise_for_status()

def choose_response_format():
    while True:
        response_format = input("Do you want the response in text, speech, or image format? ").strip().lower()
        if response_format in ['text', 'speech', 'image']:
            return response_format
        else:
            print("Invalid choice. Please enter 'text', 'speech', or 'image'.")

while True:
    # Prompt user for input
    user_input = input("How can I assist you today? ")

    # Generate response based on user input format preference
    response_format = choose_response_format()

    if response_format == 'text':
        # Generate text response using ChatGPT
        chatgpt_response = generate_text_response(user_input)
        print(chatgpt_response)
    elif response_format == 'speech':
        # Generate text response using ChatGPT and convert it to speech
        chatgpt_response = generate_text_response(user_input)
        text_to_speech(chatgpt_response)
    else:  # response_format == 'image'
        # Generate image response using ClipDrop API
        generate_image_response(user_input)

    # Ask for input again or break the loop
    user_choice = input("Do you want to ask something else? (yes/no) ").lower()
    if user_choice != 'yes':
        break
