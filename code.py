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




###for video bot
!apt -y install -qq aria2
!pip install -q torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 torchtext==0.14.1 torchdata==0.5.1 --extra-index-url https://download.pytorch.org/whl/cu116 -U
!pip install pandas-gbq==0.18.1 -U
# !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/text-to-video-synthesis/resolve/main/hub/damo/text-to-video-synthesis/VQGAN_autoencoder.pth -d /content/models -o VQGAN_autoencoder.pth
# !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/text-to-video-synthesis/resolve/main/hub/damo/text-to-video-synthesis/open_clip_pytorch_model.bin -d /content/models -o open_clip_pytorch_model.bin
# !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/text-to-video-synthesis/resolve/main/hub/damo/text-to-video-synthesis/text2video_pytorch_model.pth -d /content/models -o text2video_pytorch_model.pth
# !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/text-to-video-synthesis/raw/main/hub/damo/text-to-video-synthesis/configuration.json -d /content/models -o configuration.json

# from https://huggingface.co/kabachuha/modelscope-damo-text2video-pruned-weights
!aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/kabachuha/modelscope-damo-text2video-pruned-weights/resolve/main/VQGAN_autoencoder.pth -d /content/models -o VQGAN_autoencoder.pth
!aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/kabachuha/modelscope-damo-text2video-pruned-weights/resolve/main/open_clip_pytorch_model.bin -d /content/models -o open_clip_pytorch_model.bin
!aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/kabachuha/modelscope-damo-text2video-pruned-weights/resolve/main/text2video_pytorch_model.pth -d /content/models -o text2video_pytorch_model.pth
!aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/kabachuha/modelscope-damo-text2video-pruned-weights/raw/main/configuration.json -d /content/models -o configuration.json

!pip install -q open_clip_torch pytorch_lightning
!pip install -q git+https://github.com/camenduru/modelscope
!sed -i -e 's/\"tiny_gpu\": 1/\"tiny_gpu\": 0/g' /content/models/configuration.json

import os
os._exit(0)


import torch, random, gc
from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys

torch.manual_seed(random.randint(0, 2147483647))
pipe = pipeline('text-to-video-synthesis', '/content/models')

!mkdir /content/videos


import gc
import datetime
from IPython.display import HTML

# Get user input for the text prompt
prompt_text = input("Enter the text prompt: ")

# Assuming `pipe` function is defined elsewhere and `OutputKeys` is imported properly
with torch.no_grad():
    torch.cuda.empty_cache()
gc.collect()

test_text = {
    'text': prompt_text,  # Use the prompt_text variable here
}
output_video_path = pipe(test_text,)[OutputKeys.OUTPUT_VIDEO]

new_video_path = f'/content/videos/{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}.mp4'
!ffmpeg -y -i {output_video_path} -c:v libx264 -c:a aac -strict -2 {new_video_path} >/dev/null 2>&1



from IPython.display import HTML
from base64 import b64encode

!cp {new_video_path} /content/videos/tmp.mp4
mp4 = open('/content/videos/tmp.mp4','rb').read()

decoded_vid = "data:video/mp4;base64," + b64encode(mp4).decode()
HTML(f'<video width=400 controls><source src="{decoded_vid}" type="video/mp4"></video>')
