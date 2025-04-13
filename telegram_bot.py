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
            # First, update to thinking state
            await status_message.edit_text("🧠 Thinking...")
            
            # Run the API call in a separate thread to avoid blocking
            response = await asyncio.to_thread(
                self.solar_api.complete,
                prompt=user_question,
                search_grounding=True,
                return_sources=True,
                stream=False
            )
            
            # Print search sources in console for debugging
            if isinstance(response, dict) and 'sources' in response:
                print("\n=== SEARCH SOURCES ===")
                for idx, source in enumerate(response['sources']):
                    print(f"Source {idx+1}:")
                    print(f"  Title: {source.get('title', 'N/A')}")
                    print(f"  URL: {source.get('url', 'N/A')}")
                    print(f"  Content: {source.get('content', 'N/A')[:100]}...")
                print("=====================\n")
                
                answer = response.get('response', '')
                sources = response.get('sources', [])
                
                # Send initial answer immediately for better UX
                await status_message.edit_text(
                    f" <b>Answer:</b> {answer}", 
                    parse_mode="HTML", 
                    disable_web_page_preview=True
                )
                
                # Then process citations in the background
                try:
                    ## Show processing status
                    #await status_message.edit_text(
                    #    f"✅ <b>Answer:</b>\n\n{answer}\n\n<i>Adding citations...</i>",
                    #    parse_mode="HTML",
                    #    disable_web_page_preview=True
                    #)
                    
                    # Add citations to the answer
                    citation_result = await asyncio.to_thread(
                        self.solar_api.fill_citation,
                        response_text=answer,
                        sources=sources
                    )
                    
                    # Try to parse the citation result as JSON
                    try:
                        citation_data = json.loads(citation_result)
                        cited_text = citation_data.get("cited_text", answer)
                        references = citation_data.get("references", [])
                        
                        # Format the message with citations
                        formatted_answer = cited_text
                        
                        # Add reference links if there are any
                        if references:
                            for i, ref in enumerate(references):
                                # Get domain name for shorter display
                                url = ref.get('url', '')
                                try:
                                    from urllib.parse import urlparse
                                    domain = urlparse(url).netloc
                                except:
                                    domain = url
                                
                                if i > 0:
                                    formatted_answer += ", "
                                else:
                                    formatted_answer += "\n"

                                formatted_answer += f"<a href='{url}'>[{ref.get('number')}] {domain}</a>"
                        
                        # Update the message with cited version
                        await status_message.edit_text(
                            f"✅ <b>Answer:</b> {formatted_answer}",
                            parse_mode="HTML",
                            disable_web_page_preview=True
                        )
                    except json.JSONDecodeError:
                        # If parsing fails, keep the original answer
                        print("Failed to parse citation result as JSON, keeping original answer")
                except Exception as e:
                    print(f"Error adding citations: {str(e)}")
                    # The original answer is already displayed, so we don't need to update again
            else:
                print("\n=== NO SEARCH SOURCES FOUND ===")
                print(f"Raw response type: {type(response)}")
                print(f"Raw response preview: {str(response)[:200]}...")
                print("==============================\n")
                answer = response
                
                # Send the answer without citations since no sources were found
                await status_message.edit_text(
                    f"✅ <b>Answer:</b>\n\n{answer}",
                    parse_mode="HTML",
                    disable_web_page_preview=True
                )
            
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