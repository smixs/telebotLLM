import asyncio
import logging
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

import openai
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    ContextTypes,
    MessageHandler,
    CommandHandler,
    CallbackQueryHandler,
    filters
)
# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

class VoiceTranscriptionBot:
    def __init__(self, token: str):
        """Initialize the bot with the given token."""
        self.application = Application.builder().token(token).build()
        self.setup_handlers()
        self.transcription_cache = {}  # Add cache for transcriptions
        
    def setup_handlers(self):
        """Set up message handlers."""
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, self.handle_audio))
        self.application.add_handler(CallbackQueryHandler(self.handle_callback))
        
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /start command."""
        await update.message.reply_text(
            "👋 Hi! I'm a voice transcription bot. Send me a voice message and "
            "I'll convert it to text using OpenAI's Whisper model."
        )

    async def handle_audio(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming voice messages and audio files."""
        try:
            processing_msg = await update.message.reply_text(
                "🎵 Транскрибация..."
            )
            
            # Получаем аудиофайл
            if update.message.voice:
                audio_file = await update.message.voice.get_file()
                file_extension = "ogg"
            else:
                audio_file = await update.message.audio.get_file()
                file_extension = "mp3"
            
            # Создаем временную директорию, если не существует
            Path("temp").mkdir(exist_ok=True)
            
            # Скачиваем аудиофайл
            file_path = f"temp/{audio_file.file_id}.{file_extension}"
            await audio_file.download_to_drive(file_path)
            
            # Транскрибируем с помощью Whisper
            transcript = await self.transcribe_audio(file_path)
            
            if transcript:
                self.transcription_cache[update.message.chat_id] = transcript
                
                keyboard = [
                    [
                        InlineKeyboardButton("✍️ shima' style", callback_data="edit_text"),
                        InlineKeyboardButton("📋 make task", callback_data="make_task")
                    ]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                # Используем stream_text для отправки транскрипции
                await self.stream_text(update.message.chat_id, transcript, context.bot)
                
                # Отправляем клавиатуру отдельным сообщением
                await context.bot.send_message(
                    chat_id=update.message.chat_id,
                    text="Что с этим сделать:",
                    reply_markup=reply_markup
                )
                
                # Удаляем сообщение о транскрибации
                await processing_msg.delete()
                
            else:
                await processing_msg.delete()
                await update.message.reply_text(
                    "❌ Извините, не удалось транскрибировать ваше сообщение. Пожалуйста, попробуйте снова."
                )
                
            # Очистка
            os.remove(file_path)
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            await update.message.reply_text(
                "❌ Возникла ошибка при обработке вашего сообщения."
            )

    async def transcribe_audio(self, file_path: str) -> Optional[str]:
        """Transcribe audio file using OpenAI's Whisper model."""
        try:
            with open(file_path, "rb") as audio_file:
                client = openai.OpenAI()  # Create client instance
                transcript = await asyncio.to_thread(
                    client.audio.transcriptions.create,
                    model="whisper-1",
                    file=audio_file
                )
                return transcript.text
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return None

    async def stream_text(self, chat_id, text, bot, delay=0.5):
        """Stream text to a chat by sending it in parts."""
        MAX_MESSAGE_LENGTH = 4000  # Немного меньше максимума Telegram

        # Разбиваем текст на части
        parts = [text[i:i + MAX_MESSAGE_LENGTH] for i in range(0, len(text), MAX_MESSAGE_LENGTH)]

        for part in parts:
            await bot.send_message(chat_id=chat_id, text=part)
            await asyncio.sleep(delay)  # Задержка между отправками

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle callback queries from inline keyboards."""
        query = update.callback_query
        await query.answer()
        
        chat_id = query.message.chat_id
        if chat_id not in self.transcription_cache:
            await query.message.reply_text(
                "❌ Sorry, I couldn't find the original transcription. "
                "Please send your voice message again."
            )
            return

        processing_msg = await query.message.reply_text("Processing...")
        
        try:
            if query.data == "edit_text":
                prompt_file = "prompts/edit_text.md"
            elif query.data == "make_task":
                prompt_file = "prompts/task.md"
            
            with open(prompt_file, "r") as f:
                prompt_template = f.read()
            
            prompt = prompt_template.replace("{{text}}", self.transcription_cache[chat_id])
            
            client = openai.OpenAI()
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model="gpt-4o",
                temperature=0.5,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            result_text = response.choices[0].message.content
            await processing_msg.delete()  # Remove the processing message
            await self.stream_text(chat_id, result_text, context.bot)  # Stream the text
            
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            await processing_msg.edit_text(
                "❌ Sorry, something went wrong while processing the text."
            )

    def run(self):
        """Run the bot."""
        self.application.run_polling()

def main():
    """Main function to run the bot."""
    # Get token from environment variable
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError("Please set the TELEGRAM_BOT_TOKEN environment variable")

    # Create and run bot
    bot = VoiceTranscriptionBot(token)
    bot.run()

if __name__ == "__main__":
    main() 