import os
import logging
import asyncio
import json
from asyncio import Queue
import time
import re
from urllib.parse import urlparse

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
        #self.solar_api = SolarAPI()
        self.solar_api = SolarAPI(base_url="https://r-api.toy.x.upstage.ai/v1/chat/completions", api_key=os.environ.get("SOLAR_R_API_KEY"))

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
    
    def _format_markdown_for_telegram(self, text: str) -> str:
        """Convert common Markdown syntax to Telegram-compatible HTML format."""
        # Handle bold text: **text** or __text__ -> <b>text</b>
        text = re.sub(r'\*\*(.*?)\*\*|__(.*?)__', lambda m: f'<b>{m.group(1) or m.group(2)}</b>', text)
        
        # Handle italic text: *text* or _text_ -> <i>text</i>
        text = re.sub(r'\*(.*?)\*|_(.*?)_(?![*_])', lambda m: f'<i>{m.group(1) or m.group(2)}</i>', text)
        
        # Handle code blocks: ```text``` -> <pre>text</pre>
        text = re.sub(r'```(.*?)```', lambda m: f'<pre>{m.group(1)}</pre>', text, flags=re.DOTALL)
        
        # Handle inline code: `text` -> <code>text</code>
        text = re.sub(r'`(.*?)`', lambda m: f'<code>{m.group(1)}</code>', text)
        
        # Handle links: [text](url) -> <a href="url">text</a>
        text = re.sub(r'\[(.*?)\]\((.*?)\)', lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>', text)
        
        # Process numbered lists with preservation of structure
        def process_numbered_list(match):
            number = match.group(1)
            content = match.group(2)
            return f"{number}. <b>{content}</b>\n"
            
        # Handle numbered lists with item title formatting (assumes format: "1. **Title** - content")
        text = re.sub(r'(\d+)\.\s+\*\*(.*?)\*\*\s+(.*?)(?=\n\d+\.|\n\n|$)', 
                      lambda m: f"{m.group(1)}. <b>{m.group(2)}</b>\n{m.group(3)}\n", 
                      text, flags=re.DOTALL)
        
        # Handle bullet points with proper formatting
        text = re.sub(r'^\s*[-*+]\s+(.*?)$', r'• \1', text, flags=re.MULTILINE)
        
        # Ensure proper paragraph breaks (double newlines)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Handle soft breaks (replace single newlines within paragraphs with space)
        # text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        
        return text

    def _clean_text(self, text: str) -> str:
        """Clean text by formatting think tags and markdown into Telegram-compatible HTML."""
        def escape_html(text):
            """Escape HTML special characters."""
            html_escape_table = {
                "&": "&amp;",
                '"': "&quot;",
                "'": "&apos;",
                ">": "&gt;",
                "<": "&lt;",
            }
            return "".join(html_escape_table.get(c, c) for c in text)

        def replace_think_section(match):
            think_content = match.group(1).strip()
            if not think_content:  # Skip empty thinking sections
                return ""
            # Format thinking content with markdown support
            think_content = self._format_markdown_for_telegram(think_content)
            # Format as a visually distinct section
            return (
                "\n\n🤔 <b>Reasoning:</b> (tap to copy)\n"
                f"<pre>{think_content}</pre>\n"
            )
            
        # Process structured restaurant lists before markdown formatting
        text = self._format_restaurant_list(text)

        # Handle think tags
        text = re.sub(r'<think>(.*?)</think>', replace_think_section, text, flags=re.DOTALL)
        
        # Then format remaining text with markdown
        text = self._format_markdown_for_telegram(text)
        
        # Clean up any remaining think tags
        text = text.replace('<think>', '').replace('</think>', '')
        
        return text.strip()

    def _format_restaurant_list(self, text: str) -> str:
        """Process restaurant or numbered list patterns with proper formatting."""
        # Pattern for numbered list items with titles and descriptions
        # Example: 1. **Restaurant Name** (Location) - Description
        pattern = r'(\d+)\.\s+\*\*(.*?)\*\*\s*(\(.*?\))?\s*(?:-|\n-)\s*(.*?)(?=\n\d+\.|\Z)'
        
        def format_restaurant_item(match):
            number = match.group(1)
            name = match.group(2)
            location = match.group(3) or ""
            description = match.group(4).strip()
            
            # Extract citation references like [1][2] and preserve them
            citation_refs = re.findall(r'\[\d+\]', description)
            if citation_refs:
                citation_str = " ".join(citation_refs)
                # Remove citations from main text to reposition them
                description = re.sub(r'\[\d+\]', '', description)
                # Clean up spacing after citation removal
                description = re.sub(r'\s+', ' ', description)
                description = description.strip()
                # Add citation refs at the end of title line
                location_with_citations = f"{location} {citation_str}".strip()
            else:
                location_with_citations = location
            
            # Format bullet points in description if they exist
            description = re.sub(r'^\s*-\s+', '\n• ', description, flags=re.MULTILINE)
            # Ensure description starts with newline for proper formatting
            if not description.startswith('\n') and description:
                description = '\n' + description
                
            # Format with line breaks for better readability
            formatted_item = f"{number}. <b>{name}</b> {location}\n{description}\n"
            return formatted_item
            
        # Apply pattern with flags to handle multiline entries
        text = re.sub(pattern, format_restaurant_item, text, flags=re.DOTALL)
        
        return text

    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Process user's question using Solar API with grounding (Async Version)."""
        user_question = update.message.text
        status_message = await update.message.reply_text("🔍 Searching for information...")

        update_queue = Queue()
        # Store a reference to avoid potential issues with `self` in the thread
        solar_api_instance = self.solar_api 
        loop = asyncio.get_running_loop() # Get loop in the main async context

        def run_blocking_solar_call(question: str, queue: Queue):
            """Runs the blocking Solar API call and puts updates onto the queue."""
            try:
                search_sources_holder = [] # Use a list to hold sources found by callback
                final_text_pieces = [] # Collect pieces from stream callback

                def sync_stream_callback(content: str):
                    """Callback for stream updates (runs in API thread)."""
                    if content:
                        final_text_pieces.append(content)
                        # Use the captured loop to put items onto the queue thread-safely
                        loop.call_soon_threadsafe(queue.put_nowait, ('content', content))

                def sync_search_done_callback(sources: list):
                    """Callback for when search is done (runs in API thread)."""
                    nonlocal search_sources_holder
                    search_sources_holder.extend(sources) # Add sources to the holder
                    # Use the captured loop to put items onto the queue thread-safely
                    loop.call_soon_threadsafe(queue.put_nowait, ('sources', sources))

                    # Optional: Print search sources in console for debugging
                    if sources:
                        print("\n=== SEARCH SOURCES (from thread) ===")
                        for idx, source in enumerate(sources):
                            print(f"Source {idx+1}: Title: {source.get('title', 'N/A')}, URL: {source.get('url', 'N/A')}")
                        print("=====================\n")

                # This call blocks until the API request (including streaming) is complete
                solar_api_instance.complete(
                    prompt=question,
                    search_grounding=True,
                    return_sources=True,
                    stream=True,
                    on_update=sync_stream_callback,
                    search_done_callback=sync_search_done_callback
                )

                # Once complete() returns, signal completion with the final text and sources
                final_text = "".join(final_text_pieces)
                loop.call_soon_threadsafe(queue.put_nowait, ('done', {'text': final_text, 'sources': search_sources_holder}))

            except Exception as e:
                logger.error(f"Error in Solar API thread: {e}", exc_info=True)
                loop.call_soon_threadsafe(queue.put_nowait, ('error', str(e)))


        # Run the blocking function in a separate thread managed by asyncio
        api_task = asyncio.create_task(asyncio.to_thread(
            run_blocking_solar_call, user_question, update_queue
        ))

        accumulated_text = ""
        sources = []
        processing_done = False
        error_message = None

        # Throttling parameters
        last_update_time = time.time()
        last_update_length = 0
        min_update_interval = 0.5  # Min seconds between edits
        min_update_chars = 50      # Min new characters before attempting edit

        try:
            while not processing_done:
                try:
                    # Wait for updates from the queue with a timeout
                    update_type, data = await asyncio.wait_for(update_queue.get(), timeout=90.0) # Increased timeout slightly

                    if update_type == 'content':
                        accumulated_text += data
                        current_time = time.time()
                        current_length = len(accumulated_text)

                        # Check throttling conditions
                        if (current_length > last_update_length and
                            current_length - last_update_length >= min_update_chars and
                            current_time - last_update_time >= min_update_interval):
                            try:
                                # Clean the text before sending to Telegram
                                cleaned_text = self._clean_text(accumulated_text)
                                # Use a temporary status prefix during streaming
                                await status_message.edit_text(
                                    f"⏳<b>Answer:</b> {cleaned_text}...",
                                    parse_mode="HTML",
                                    disable_web_page_preview=True
                                )
                                last_update_length = current_length
                                last_update_time = current_time
                            except Exception as e:
                                # Ignore potential transient errors like message not modified
                                logger.warning(f"Error updating message (ignored): {e}")

                    elif update_type == 'sources':
                        # Sources are now received with 'done' message, 
                        # but we could update status here if needed earlier
                        pass 

                    elif update_type == 'done':
                        processing_done = True
                        accumulated_text = data['text'] # Get final text from 'done' payload
                        sources = data['sources']       # Get final sources from 'done' payload
                        logger.info("API call processing finished successfully.")

                    elif update_type == 'error':
                        processing_done = True
                        error_message = data
                        logger.error(f"Error received from API thread: {error_message}")

                    update_queue.task_done()

                except asyncio.TimeoutError:
                    logger.error("Timeout waiting for update from API queue.")
                    error_message = "Request timed out while waiting for response."
                    processing_done = True
                    # Ensure the background task is cancelled if it's still running
                    if not api_task.done():
                        api_task.cancel()


            # Final processing after the loop
            if error_message:
                 await status_message.edit_text(f"❌ Error generating answer: {error_message}")
                 return # Stop processing on error

            # Ensure the final accumulated text is displayed before citation
            if len(accumulated_text) > 0: # Check if we actually got text
                 try:
                     # Clean the text before sending to Telegram
                     cleaned_text = self._clean_text(accumulated_text)
                     await status_message.edit_text(
                         f"⌛<b>Answer:</b> {cleaned_text}", # Indicate final pre-citation
                         parse_mode="HTML",
                         disable_web_page_preview=True
                     )
                     last_update_length = len(accumulated_text) # Update length for citation check
                 except Exception as e:
                     logger.warning(f"Error updating final pre-citation message (ignored): {e}")
            else:
                 # Handle case where API finished but returned no text
                 await status_message.edit_text("No answer found for your query.")
                 return


            # Process citations if sources are available
            if sources:
                logger.info(f"Attempting citation processing with {len(sources)} sources.")
                try:
                    start_time = time.time()
                    # Run blocking citation fill in a separate thread - this should be async
                    citation_result_json = await asyncio.to_thread(
                        self.solar_api.add_citations,
                        response_text=accumulated_text,
                        sources=sources
                    )

                    logger.info(f"Done time: {time.time() - start_time}")
                    logger.info(f"Citation result: {citation_result_json}")

                    # Parse the citation result
                    try:
                        citation_data = json.loads(citation_result_json)
                        references = citation_data.get("references", [])

                        # If no references were found from parsing, use the source data directly
                        if not references and sources:
                            logger.info("No citations found in text, using direct sources instead")
                            references = []
                            for idx, source in enumerate(sources):
                                references.append({
                                    "number": idx + 1,  # 1-based indexing for display
                                    "url": source.get("url", ""),
                                    "title": source.get("title", "")
                                })

                        # If there are references, send them as a separate message
                        if references:
                            # Create the citations message
                            citations_message = "<b>Sources:</b>"
                            source_links = []
                            # Sort references by number for consistent ordering
                            references.sort(key=lambda r: int(r.get("number", 0)))

                            for ref in references:
                                ref_num = ref.get("number", "")
                                url = ref.get("url", "")
                                title = ref.get("title", "") # Use title if available

                                # Extract domain for display
                                display_name = title if title else "Source" # Default to title or "Source"
                                if url:
                                    try:
                                        domain = urlparse(url).netloc
                                        if domain.startswith('www.'):
                                            domain = domain[4:]
                                        # Use domain if title is generic or missing
                                        if not title or title.lower() in ["source", "untitled"]:
                                            display_name = domain or "source"
                                    except Exception:
                                        pass # Keep default display_name if URL parsing fails
                                
                                # Ensure display name is not empty and escape HTML
                                display_name = display_name or "source"
                                # Basic HTML escaping to prevent parsing errors
                                display_name = display_name.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

                                # Create link only if URL is present
                                if url:
                                    source_links.append(f"[{ref_num}] <a href='{url}'>{display_name}</a>")
                                else:
                                    source_links.append(f"[{ref_num}] {display_name}")

                            # Join sources with newlines for better readability
                            citations_message += "\n" + "\n".join(source_links)

                            try:
                                # Send citations as a separate message
                                await update.message.reply_text(
                                    citations_message,
                                    parse_mode="HTML",
                                    disable_web_page_preview=True
                                )
                                logger.info("Successfully sent citations as a separate message.")
                            except Exception as send_error:
                                logger.error(f"Error sending citations message: {send_error}")
                                # Try without HTML parsing as a fallback
                                try:
                                    plain_message = "Sources:\n" + "\n".join([f"[{ref.get('number', '')}] {ref.get('title', 'Source')}: {ref.get('url', '')}" for ref in references])
                                    await update.message.reply_text(
                                        plain_message,
                                        disable_web_page_preview=True
                                    )
                                    logger.info("Sent plaintext citations as fallback.")
                                except Exception as plain_error:
                                    logger.error(f"Failed to send plaintext citations: {plain_error}")
                        else:
                            logger.info("No citations available to send.")

                    except (json.JSONDecodeError, Exception) as e:
                        logger.error(f"Error processing citation JSON: {e}", exc_info=True)
                        # Fallback to display sources directly if citation processing fails
                        try:
                            if sources:
                                plain_message = "Sources:\n" + "\n".join([f"[{idx+1}] {source.get('title', 'Source')}: {source.get('url', '')}" for idx, source in enumerate(sources)])
                                await update.message.reply_text(
                                    plain_message,
                                    disable_web_page_preview=True
                                )
                                logger.info("Sent plaintext sources as fallback after citation processing error.")
                        except Exception as fallback_error:
                            logger.error(f"Failed to send fallback sources: {fallback_error}")
                        logger.info("Citation generation failed, but original answer remains intact.")

                except Exception as e:
                    logger.error(f"Error during citation filling API call: {e}", exc_info=True)
                    # Fallback to display sources directly if citation filling API fails
                    try:
                        if sources:
                            plain_message = "Sources:\n" + "\n".join([f"[{idx+1}] {source.get('title', 'Source')}: {source.get('url', '')}" for idx, source in enumerate(sources)])
                            await update.message.reply_text(
                                plain_message,
                                disable_web_page_preview=True
                            )
                            logger.info("Sent plaintext sources as fallback after citation API error.")
                    except Exception as fallback_error:
                        logger.error(f"Failed to send fallback sources: {fallback_error}")
                    logger.info("Citation generation failed, but original answer remains intact.")
            else:
                # Original answer already displayed, no additional action needed for no sources case
                logger.info("No sources available for citation generation.")


        except Exception as e:
            # Catch-all for unexpected errors during the async processing
            logger.error(f"Unhandled error in handle_text: {e}", exc_info=True)
            try:
                # Avoid editing if the message was deleted or inaccessible
                if status_message:
                    await status_message.edit_text(f"❌ An unexpected error occurred: {str(e)}")
            except Exception as inner_e:
                logger.error(f"Failed to send error message to user: {inner_e}")
        finally:
             # Ensure the background task is properly awaited or cancelled
             if api_task and not api_task.done():
                 logger.warning("API task still running on handle_text exit, cancelling.")
                 api_task.cancel()
                 try:
                     await api_task # Wait for cancellation (suppresses CancelledError)
                 except asyncio.CancelledError:
                     logger.info("API task cancelled successfully on cleanup.")
                 except Exception as e:
                     logger.error(f"Error awaiting cancelled API task during cleanup: {e}")
             elif api_task and api_task.done() and api_task.exception():
                 # Log exception if task finished with an error wasn't handled before
                 exc = api_task.exception()
                 logger.error(f"API task finished with unhandled exception: {exc}", exc_info=exc)


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