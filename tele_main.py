import os
from telethon.sync import TelegramClient

api_id = int(os.getenv("API_ID"))
api_hash = os.getenv("API_HASH")
channel = '@sunstudy11'

with TelegramClient('my_session', api_id, api_hash) as client:
    for message in client.iter_messages(channel, limit=5):
        print(message.sender_id, message.text)
