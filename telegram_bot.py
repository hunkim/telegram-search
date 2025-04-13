import os
import logging
import asyncio
import json
from asyncio import Queue
import time
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
                                # Use a temporary status prefix during streaming
                                await status_message.edit_text(
                                    f"⏳<b>Answer:</b> {accumulated_text}...",
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
                     await status_message.edit_text(
                         f"⌛<b>Answer:</b> {accumulated_text}", # Indicate final pre-citation
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
                    # Run blocking citation fill in a separate thread
                    citation_result_json = await asyncio.to_thread(
                        self.solar_api.fill_citation_heruistic,
                        response_text=accumulated_text,
                        sources=sources
                    )

                    logger.info(f"Done time: {time.time() - start_time}")
                    logger.info(f"Citation result: {citation_result_json}")

                    # Parse the citation result
                    try:
                        citation_data = json.loads(citation_result_json)
                        cited_text = citation_data.get("cited_text", accumulated_text)
                        references = citation_data.get("references", [])

                        # Build the final message with citations and references
                        final_message = f"✅<b>Answer:</b> {cited_text}"

                        if references:
                            final_message += "\n\n<b>Sources:</b>" # Add a newline before sources list
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
                                
                                # Ensure display name is not empty
                                display_name = display_name or "source"

                                # Create link only if URL is present
                                if url:
                                    source_links.append(f"[{ref_num}] <a href='{url}'>{display_name}</a>")
                                else:
                                    source_links.append(f"[{ref_num}] {display_name}")


                            # Join sources with newlines for better readability
                            final_message += "\n" + "\n".join(source_links)

                        # Final update with citations
                        # Check if cited text is meaningfully different before editing again
                        if cited_text != accumulated_text or references:
                            await status_message.edit_text(
                                final_message,
                                parse_mode="HTML",
                                disable_web_page_preview=True 
                            )
                            logger.info("Successfully updated message with citations.")
                        else:
                             logger.info("Cited text identical to original, skipping final edit.")


                    except (json.JSONDecodeError, Exception) as e:
                        logger.error(f"Error processing citation JSON: {e}", exc_info=True)
                        # Fallback: Show the answer without citations if JSON processing failed
                        await status_message.edit_text(
                            f"✅<b>Answer:</b> {accumulated_text}\n\n(Could not process citations)",
                            parse_mode="HTML",
                            disable_web_page_preview=True
                        )

                except Exception as e:
                    logger.error(f"Error during citation filling API call: {e}", exc_info=True)
                    # Fallback: Show the answer without citations if API call failed
                    await status_message.edit_text(
                        f"✅<b>Answer:</b> {accumulated_text}\n\n(Citation generation failed)",
                        parse_mode="HTML",
                        disable_web_page_preview=True
                    )
            else:
                # If no sources, ensure the final message state is correct
                 await status_message.edit_text(
                     f"✅<b>Answer:</b> {accumulated_text}",
                     parse_mode="HTML",
                     disable_web_page_preview=True
                 )
                 logger.info("Final message updated (no sources).")


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