import os
import logging
import asyncio
import json
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

from solar import SolarAPI

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

class TelegramBot:
    def __init__(self, token):
        self.application = Application.builder().token(token).build()
        self.solar_api = SolarAPI()
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Set up command and message handlers"""
        # Command handlers
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("help", self.help_command))
        
        # Text handler
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text)
        )
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Send a message when the command /start is issued."""
        await update.message.reply_text(
            "Hello! Send me any question and I'll search for an answer using Solar API."
        )
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Send help message when the command /help is issued."""
        help_text = (
            "☀️ <b>Welcome to Solar Bot!</b>\n\n"
            "📚 <b>Basic Commands:</b>\n"
            "• /start - Start the bot\n"
            "• /help - Show this help message\n\n"
            "💡 <b>How to use:</b>\n"
            "Simply type any question, and I'll use Solar API with grounding to find you an answer!\n\n"
            "⚡️ Powered by <a href='https://console.upstage.ai'>Upstage SolarLLM</a>"
        )

        await update.message.reply_text(
            help_text, parse_mode="HTML", disable_web_page_preview=True
        )
    
    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Process user's question using Solar API with grounding."""
        user_question = update.message.text
        
        # Show "Searching..." message
        status_message = await update.message.reply_text("🔍 Searching for information...")
        
        try:
            # Create a thread-safe container for our response
            from threading import Lock
            
            class ThreadSafeBuffer:
                def __init__(self):
                    self.buffer = ""
                    self.lock = Lock()
                
                def append(self, content):
                    with self.lock:
                        self.buffer += content
                        return len(self.buffer)
                
                def get(self):
                    with self.lock:
                        return self.buffer
            
            # Initialize our buffer and sources list
            buffer = ThreadSafeBuffer()
            sources = []
            last_update_length = 0
            
            # Callback for stream updates - this runs in a different thread
            def handle_stream_update(content):
                buffer.append(content)
            
            # Callback for when search is completed
            def search_done(search_sources):
                nonlocal sources
                sources = search_sources
                
                # Print search sources in console for debugging
                if sources:
                    print("\n=== SEARCH SOURCES ===")
                    for idx, source in enumerate(sources):
                        print(f"Source {idx+1}:")
                        print(f"  Title: {source.get('title', 'N/A')}")
                        print(f"  URL: {source.get('url', 'N/A')}")
                        print(f"  Content: {source.get('content', 'N/A')[:100]}...")
                    print("=====================\n")
            
            # Start the API call in a separate thread
            import threading
            api_thread = threading.Thread(
                target=self.solar_api.complete,
                kwargs={
                    "prompt": user_question,
                    "search_grounding": True,
                    "return_sources": True,
                    "stream": True,
                    "on_update": handle_stream_update,
                    "search_done_callback": search_done
                }
            )
            api_thread.start()
            
            # Periodically check and update the message while the API call is running
            while api_thread.is_alive():
                current_text = buffer.get()
                current_length = len(current_text)
                
                # Update message if we have new content and it's been at least 50 chars
                if current_length > last_update_length and current_length - last_update_length >= 50:
                    try:
                        await status_message.edit_text(
                            f"<b>Answer:</b> {current_text}",
                            parse_mode="HTML",
                            disable_web_page_preview=True
                        )
                        last_update_length = current_length
                    except Exception as e:
                        print(f"Error updating message: {str(e)}")
                
                # Wait a short time before checking again
                await asyncio.sleep(0.1)
            
            # Make sure the thread completes
            api_thread.join()
            
            # Get the final text
            final_text = buffer.get()
            
            # Send the final response if we have one
            if final_text:
                try:
                    # First update with the complete response
                    if len(final_text) > last_update_length:
                        await status_message.edit_text(
                            f"<b>Answer:</b> {final_text}",
                            parse_mode="HTML",
                            disable_web_page_preview=True
                        )
                    
                    # If we have sources, add citations
                    if sources:
                        # Process citations synchronously
                        citation_result = await asyncio.to_thread(
                            self.solar_api.fill_citation,
                            response_text=final_text,
                            sources=sources
                        )
                        
                        # Try to parse the citation result as JSON
                        try:
                            citation_data = json.loads(citation_result)
                            cited_text = citation_data.get("cited_text", final_text)
                            references = citation_data.get("references", [])
                            
                            # Build the final message with citations and references
                            message = f"✅<b>Answer:</b> {cited_text}"
                            
                            if references:
                                message += "\n"
                                source_links = []
                                
                                for ref in references:
                                    ref_num = ref.get("number", "")
                                    url = ref.get("url", "")
                                    
                                    # Extract and clean domain name
                                    try:
                                        from urllib.parse import urlparse
                                        full_domain = urlparse(url).netloc
                                        
                                        # Remove 'www.' prefix if present
                                        if full_domain.startswith('www.'):
                                            full_domain = full_domain[4:]
                                            
                                        # Extract main domain name (without TLD)
                                        parts = full_domain.split('.')
                                        if len(parts) >= 2:
                                            # For domains like domain.com or domain.org
                                            main_domain = parts[0]
                                            
                                            # For domains like domain.co.kr, get the main part
                                            if len(parts) > 2 and parts[-2] in ['co', 'com', 'org', 'net', 'gov']:
                                                main_domain = parts[-3]
                                        else:
                                            main_domain = full_domain
                                    except:
                                        # Fallback if parsing fails
                                        main_domain = url.split("/")[2] if len(url.split("/")) > 2 else "source"
                                        if main_domain.startswith('www.'):
                                            main_domain = main_domain[4:]
                                        main_domain = main_domain.split('.')[0]
                                    
                                    # Create hyperlinked domain name
                                    source_links.append(f"[{ref_num}] <a href='{url}'>{main_domain}</a>")
                                
                                # Join all sources with commas in a single line
                                message += ", ".join(source_links)
                            
                            # Update with the cited version
                            await status_message.edit_text(
                                message,
                                parse_mode="HTML",
                                disable_web_page_preview=True
                            )
                        except (json.JSONDecodeError, Exception) as e:
                            print(f"Error processing citations: {str(e)}")
                            # If citation processing fails, just keep the final text
                except Exception as e:
                    print(f"Error updating final message: {str(e)}")
            
            # If you want to add citations or further processing with the sources, do it here
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            await status_message.edit_text(f"❌ Error generating answer: {str(e)}")
    
    async def update_to_thinking_state(self, context, source_count):
        """Update the status message to indicate we're in the thinking state."""
        status_message = context.user_data.get("status_message")
        if status_message:
            if source_count > 0:
                await status_message.edit_text(f"🧠 Thinking... (Found {source_count} sources)")
            else:
                await status_message.edit_text("🧠 Thinking...")
    
    def run(self):
        """Start the bot."""
        self.application.run_polling()


def main():
    """Initialize and start the bot."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN environment variable not set!")
        return
    
    bot = TelegramBot(token)
    logger.info("Starting bot...")
    bot.run()


if __name__ == "__main__":
    main() 