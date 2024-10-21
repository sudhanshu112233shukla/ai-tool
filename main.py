import openai
import os
import subprocess
from google.cloud import storage
from google.cloud import speech_v1
from google.cloud import texttospeech
from google.cloud.speech_v1 import enums
from pydub.utils import mediainfo
from datetime import datetime
import srt
import sys
from dotenv import load_dotenv

# Set up Azure OpenAI credentials
AZURE_OPENAI_KEY = "22ec84421ec24230a3638d1b51e3a7dc"
ENDPOINT_URL ="https:// internshala.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview"

def correct_transcription(transcription):
    openai.api_key = AZURE_OPENAI_KEY

    response = openai.ChatCompletion.create(
        engine="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that fixes grammatical mistakes."},
            {"role": "user", "content": transcription}
        ],
        max_tokens=1000
    )

    corrected_text = response['choices'][0]['message']['content']
    return corrected_text

def video_info(video_filepath):
    """Returns number of channels, bit rate, and sample rate of the video"""
    video_data = mediainfo(video_filepath)
    channels = video_data["channels"]
    bit_rate = video_data["bit_rate"]
    sample_rate = video_data["sample_rate"]
    return channels, bit_rate, sample_rate

def video_to_audio(video_filepath, audio_filename, video_channels, video_bit_rate, video_sample_rate):
    """Converts video into audio and uploads the audio to Google Cloud Storage."""
    command = f"ffmpeg -i {video_filepath} -ac {video_channels} -ab {video_bit_rate} -ar {video_sample_rate} {audio_filename}"
    subprocess.call(command, shell=True)
    blob_name = f"audios/{audio_filename}"
    upload_blob(BUCKET_NAME, audio_filename, blob_name)
    return blob_name

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        print(f"File {source_file_name} uploaded to {destination_blob_name}.")
    except Exception as e:
        print(f"Error uploading file: {e}")

def long_running_recognize(storage_uri, channels, sample_rate):
    """Transcribes the audio."""
    client = speech_v1.SpeechClient()
    config = {
        "language_code": LANG,
        "sample_rate_hertz": int(sample_rate),
        "encoding": enums.RecognitionConfig.AudioEncoding.LINEAR16,
        "audio_channel_count": int(channels),
        "enable_word_time_offsets": True,
        "model": "default",
        "enable_automatic_punctuation": True
    }
    audio = {"uri": storage_uri}
    print(f"Using the config: {config}")
    print(f"Audio file location: {audio}")

    operation = client.long_running_recognize(config=config, audio=audio)
    print(u"Waiting for operation to complete...")
    response = operation.result(timeout=1000000)

    subs = []
    for result in response.results:
        subs = break_sentences(MAX_CHARS, subs, result.alternatives[0])

    print("Transcribing finished")
    return subs

def break_sentences(max_chars, subs, alternative):
    """Breaks sentences by punctuations and maximum sentence length."""
    firstword = True
    charcount = 0
    idx = len(subs) + 1
    content = ""

    for w in alternative.words:
        if firstword:
            start = w.start_time.ToTimedelta()
        charcount += len(w.word)
        content += " " + w.word.strip()

        if ("." in w.word or "!" in w.word or "?" in w.word or
                charcount > max_chars or
                ("," in w.word and not firstword)):
            subs.append(srt.Subtitle(index=idx,
                                     start=start,
                                     end=w.end_time.ToTimedelta(),
                                     content=srt.make_legal_content(content)))
            firstword = True
            idx += 1
            content = ""
            charcount = 0
        else:
            firstword = False
    return subs

def write_srt(subs):
    """Writes SRT file."""
    srt_file = f"{get_timestamp()}_subtitles.srt"
    with open(srt_file, mode="w", encoding="utf-8") as f:
        content = srt.compose(subs)
        f.writelines(str(content))

def write_txt(subs):
    """Writes TXT file."""
    txt_file = f"{get_timestamp()}_subtitles.txt"
    with open(txt_file, mode="w", encoding="utf-8") as f:
        for s in subs:
            content = s.content.strip() + "\n"
            f.write(str(content))

def get_timestamp():
    """Gets current date and time.""" 
    current_datetime = datetime.now()
    return str(current_datetime).replace(" ", "_").replace(":", "_")

def text_to_speech(text, output_audio_file):
    """Converts text to speech using Google Text-to-Speech."""
    client = texttospeech.TextToSpeechClient()
    input_text = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code='en-US',
        name='en-US-Journey'
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        speaking_rate=1.0
    )

    response = client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)

    with open(output_audio_file, 'wb') as out:
        out.write(response.audio_content)
    print(f"Audio content written to file {output_audio_file}")

def replace_audio_in_video(video_filepath, new_audio_filepath, output_video_filepath):
    """Replaces the audio in the video with new audio."""
    command = f"ffmpeg -i {video_filepath} -i {new_audio_filepath} -c:v copy -c:a aac -strict experimental {output_video_filepath}"
    subprocess.call(command, shell=True)
    print(f"Replaced audio in video. Output file: {output_video_filepath}")

# Load configuration from .env file
load_dotenv()
BUCKET_NAME = str(os.getenv('BUCKET_NAME'))
MAX_CHARS = int(os.getenv('MAX_CHARS'))
FFMPEG_LOCATION = str(os.getenv('FFMPEG_LOCATION'))
FFPROBE_LOCATION = str(os.getenv('FFPROBE_LOCATION'))

# Load ffmpeg location
mediainfo.converter = FFMPEG_LOCATION
mediainfo.ffmpeg = FFMPEG_LOCATION
mediainfo.ffprobe = FFPROBE_LOCATION

# Take CLI arguments
if len(sys.argv) != 3:
    print("Missing command-line argument. Usage: python main.py example.wav en-US")
    exit(1)
video_path = sys.argv[1]
LANG = sys.argv[2]

# Access GCP credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"

# Call the main function
def main():
    # Calculate media info from video
    channels, bit_rate, sample_rate = video_info(video_path)

    # Convert to audio
    audio_filename = f"{get_timestamp()}_audio.wav"
    blob_name = video_to_audio(video_path, audio_filename, channels, bit_rate, sample_rate)

    # Upload to Google storage
    gcs_uri = f"gs://{BUCKET_NAME}/{blob_name}"

    # Transcribe
    response = long_running_recognize(gcs_uri, channels, sample_rate)

    # Write SRT and TXT files
    write_srt(response)
    write_txt(response)

    # Correct transcription
    corrected_text = correct_transcription(' '.join([s.content for s in response]))

    # Convert corrected text to speech
    audio_output_path = f"{get_timestamp()}_corrected_audio.wav"
    text_to_speech(corrected_text, audio_output_path)

    # Replace audio in original video
    output_video_path = f"{get_timestamp()}_final_video.mp4"
    replace_audio_in_video(video_path, audio_output_path, output_video_path)

if __name__ == "__main__":
    main()
