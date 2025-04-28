import requests
import json
import sseclient
import os
from datetime import datetime
import re # Added for sentence splitting
import json # Added for output formatting
from collections import OrderedDict # Added for ordered mapping

class SolarAPI:
    def __init__(self, api_key=os.getenv("UPSTAGE_API_KEY"), base_url="https://api.upstage.ai/v1/chat/completions"):
        """Initialize the SolarAPI client with the API key.
        
        Args:
            api_key (str): Your Upstage API key
        """
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def complete(self, prompt, model=None, stream=False, on_update=None, search_grounding=False, return_sources=False, search_done_callback=None):
        """Send a completion request to the Solar API.
        
        Args:
            prompt (str): The user's input prompt
            model (str): The model to use
            stream (bool): Whether to stream the response
            on_update (callable): Function to call with each update when streaming
                                 Should accept one argument (the new token/content)
            search_grounding (bool): Whether to ground responses using Tavily search results
            return_sources (bool): Whether to return search result sources along with the response
            search_done_callback (callable): Function to call when search is completed
                                           Should accept one argument (the search results)
        
        Returns:
            str or dict: The complete response text (non-streaming) or a dict with response and sources if return_sources=True
        """
        # If search grounding is enabled, use search results to augment the prompt
        sources = []
        if search_grounding:
            try:
                from tavily import TavilyClient
                tavily_api_key = os.getenv("TAVILY_API_KEY")
                if not tavily_api_key:
                    raise ValueError("TAVILY_API_KEY environment variable is not set")
                
                # Get search queries from the prompt
                if False:
                    queries_json = extract_search_queries(prompt)
                    queries = json.loads(queries_json)["search_queries"]
                    print(f"Search queries: {queries}")
                else:
                    queries = [prompt]

                # Initialize Tavily client
                tavily_client = TavilyClient(tavily_api_key)
                
                # Collect search results for each query
                all_search_results = []
                for query in queries[:3]:
                    print(f"Searching for {query}")
                    search_response = tavily_client.search(
                        query=query,
                        max_results=10,
                        include_raw_content=True,
                        # topic="news",
                    )

                    print(f"Search response length: {len(search_response.get('results', []))}")
                    all_search_results.extend(search_response.get('results', []))
                    print(f"All search results length: {len(all_search_results)}")
                
                # remove duplicates if the URL is the same
                all_search_results = [result for n, result in enumerate(all_search_results) if result not in all_search_results[n + 1:]]
                print(f"All search results length after removing duplicates: {len(all_search_results)}")

                # Format search results as context
                search_context = ""
                for i, result in enumerate(all_search_results, 1):  # Limit to top 10 results
                    title = result.get('title', 'No Title')
                    content = result.get('content', result.get('raw_content', 'No Content'))
                    url = result.get('url', 'No URL')
                    search_context += f"[{i}]. {title}\n{content}\n\n"
                    
                    # Save source metadata for return if needed
                    sources.append({
                        "id": i,
                        "title": title,
                        "url": url,
                        "content": content,
                        "score": result.get('score', 0),
                        "published_date": result.get('published_date', 'No Date')
                    })
                
                # Create a grounded prompt with the search results
                grounded_prompt = f"""Use the following search results to help answer the user's question.
---                
SEARCH RESULTS:
{search_context}
---
USER QUESTION: {prompt}

---
IMPORTANT INSTRUCTIONS:
1. Respond in the SAME LANGUAGE as the user's question. If the question is in Korean, respond in Korean.
2. Be BRIEF and CONCISE - this is for Telegram, so get to the point clearly.
3. Make FULL USE of the search results and use terms from the search results in your response.
4. Add citation numbers [1], [2], etc. directly after the specific word or sentence that uses information from the sources. Add citations only for highly relevant information derived from the sources.
5. Consider TIME-SENSITIVITY - today's date is {datetime.now().strftime("%Y-%m-%d")}.

Provide a direct, informative answer based on the search results. If the search results don't contain relevant information, briefly state that you don't have sufficient information to answer the question.

Keep your tone friendly but efficient.
"""
                
                prompt = grounded_prompt
                
                # Call the search done callback if provided
                if search_done_callback:
                    search_done_callback(sources)
                    
            except Exception as e:
                print(f"Search grounding failed: {str(e)}. Falling back to standard completion.")
                # Call the callback with empty results if there was an error
                if search_done_callback:
                    search_done_callback([])
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "stream": stream
        }

        if stream:
            response_text = self._stream_request(payload, on_update)
        else:
            response_text = self._standard_request(payload)
            
        # Return response with sources if requested
        if return_sources and search_grounding:
            return {
                "response": response_text,
                "sources": sources
            }
        else:
            return response_text
    
    def add_citations(self, response_text, sources):
        """
        Adds citations to response text based on sources, only including relevant citation numbers.
        
        Args:
            response_text (str): The response text with citation numbers like [1], [2]
            sources (list): A list of source dictionaries
            
        Returns:
            dict: A dictionary with cited_text and filtered references
        """
        try:
            # First check if response_text already contains citations
            citation_pattern = r'\[(\d+)\]'
            found_citations = set(int(match) for match in re.findall(citation_pattern, response_text))
            
            # If no citations found, return original text with empty references
            if not found_citations:
                return json.dumps({"cited_text": response_text, "references": []})
            
            # Filter sources to only include those referenced in text
            filtered_references = []
            for source in sources:
                # Get the source ID/number
                source_num = source.get('id', None)
                if source_num is not None and source_num in found_citations:
                    filtered_references.append({
                        "number": source_num,
                        "url": source.get("url", ""),
                        "title": source.get("title", "")
                    })
            
            return json.dumps({
                "cited_text": response_text,
                "references": filtered_references
            })
        except Exception as e:
            print(f"Error in add_citations: {e}")
            return json.dumps({"cited_text": response_text, "references": []})
    
    def fill_citation_heruistic(self, response_text, sources):
        """
        Adds citations to response_text based on keyword overlap heuristic.

        Args:
            response_text: The text generated by the LLM.
            sources: A list of source dictionaries, each potentially containing
                     'url', 'title', 'content'.

        Returns:
            A JSON string with 'cited_text' and 'references', or None if error.
        """
        if not response_text or not sources:
            # Return original text if no sources or text to cite
            return json.dumps({"cited_text": response_text or "", "references": []})

        # Simple English stop words list (can be expanded)
        stop_words = set([
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
            "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
            "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
            "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
            "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
            "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
            "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
            "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
            "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
            "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
            "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
            "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
        ])

        def get_words(text):
            """Basic tokenizer and stop word removal."""
            if not text: return set()
            words = re.findall(r'\b\w+\b', text.lower())
            return set(word for word in words if word not in stop_words and len(word) > 1)

        # --- Dynamic Threshold Setup ---
        initial_threshold = 4  # Start with the desired "high relevance" threshold
        min_threshold = 2      # Minimum acceptable overlap to avoid excessive noise (adjust if needed)
        current_threshold = initial_threshold
        # -----------------------------

        # Prepare sources with numbers and pre-process content
        numbered_sources = []
        source_details_map = {} # Map original number to details for later lookup
        for i, source in enumerate(sources):
            content_words = get_words(source.get('content'))
            original_num = i + 1
            if content_words: # Only consider sources with processable content
                details = {
                    "number": original_num,
                    "url": source.get("url", ""),
                    "title": source.get("title", ""), # Keep title
                    "content_words": content_words
                }
                numbered_sources.append(details)
                source_details_map[original_num] = details

        if not numbered_sources:
             return json.dumps({"cited_text": response_text, "references": []})

        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', response_text)

        # --- Loop for Dynamic Threshold ---
        final_sentences_with_citations = [] # Store results from the successful pass
        
        while current_threshold >= min_threshold:
            print(f"Attempting citation with threshold: {current_threshold}") # Logging/Debug
            
            # Reset results for this specific threshold attempt
            current_pass_sentences = []
            any_citations_found_this_pass = False

            for sentence in sentences:
                if not sentence.strip():
                    continue

                sentence_words = get_words(sentence)
                # Add sentence structure even if no words or no citations found later
                sentence_entry = {"text": sentence, "citations": []}

                if not sentence_words:
                     current_pass_sentences.append(sentence_entry)
                     continue

                matching_source_nums = []
                for source in numbered_sources:
                    overlap = sentence_words.intersection(source["content_words"])
                    # Use the *current* threshold for checking
                    if len(overlap) >= current_threshold:
                        matching_source_nums.append(source["number"])
                        any_citations_found_this_pass = True # Mark success for this pass

                # Sort original numbers for consistent intermediate representation
                matching_source_nums.sort()
                sentence_entry["citations"] = matching_source_nums
                current_pass_sentences.append(sentence_entry)

            # Check if citations were found in this pass
            if any_citations_found_this_pass:
                print(f"Found citations at threshold {current_threshold}. Using these results.") # Logging/Debug
                final_sentences_with_citations = current_pass_sentences # Store the successful results
                break # Exit the while loop, we found citations
            else:
                print(f"No citations found at threshold {current_threshold}. Decreasing threshold.") # Logging/Debug
                current_threshold -= 1
                # Continue to the next iteration of the while loop with lower threshold

        # Handle case where loop finished without finding any citations
        if not final_sentences_with_citations:
             print(f"No citations found even at minimum threshold {min_threshold}. Proceeding without citations.") # Logging/Debug
             # Populate with original sentences and empty citations if loop failed
             final_sentences_with_citations = [{"text": s, "citations": []} for s in sentences if s.strip()]
        # --- End Dynamic Threshold Loop ---

        # --- Reordering Logic (Uses final_sentences_with_citations) ---
        # This part remains largely the same, but uses the list determined by the dynamic threshold loop
        final_cited_sentences = []
        old_to_new_mapping = OrderedDict()
        next_new_citation_num = 1

        # First pass: Find unique citations in order of appearance
        for sentence_data in final_sentences_with_citations: # Use the final list
            for original_num in sentence_data["citations"]:
                if original_num not in old_to_new_mapping:
                    old_to_new_mapping[original_num] = next_new_citation_num
                    next_new_citation_num += 1

        # Second pass: Replace original numbers with new sequential numbers
        for sentence_data in final_sentences_with_citations: # Use the final list
            original_sentence_text = sentence_data["text"]
            # Use .get() default to handle potential empty list if no citations found at all
            new_citation_nums = [old_to_new_mapping.get(orig_num) for orig_num in sentence_data["citations"]]
            # Filter out None values in case of unexpected issue, though should not happen with current logic
            new_citation_nums = [num for num in new_citation_nums if num is not None]


            if new_citation_nums:
                # Sort the *new* numbers for consistent display
                new_citation_nums.sort()
                # Add space before the first citation only, then no spaces between consecutive citations
                citation_str = f" [" + "][".join([str(num) for num in new_citation_nums]) + "]"

                # Insert citation before trailing punctuation
                sentence_strip = original_sentence_text.rstrip()
                trailing_punctuation = ""
                if sentence_strip and sentence_strip[-1] in '.!?':
                    trailing_punctuation = sentence_strip[-1]
                    sentence_base = sentence_strip[:-1].rstrip()
                else:
                    sentence_base = sentence_strip

                cited_sentence = sentence_base + citation_str + trailing_punctuation
                final_cited_sentences.append(cited_sentence)
            else:
                # Append original sentence if no citation needed
                final_cited_sentences.append(original_sentence_text)
        # --- End Reordering Logic ---

        # Reconstruct the final text
        cited_text = " ".join(final_cited_sentences)

        # Build references list using the new mapping
        references = []
        # Iterate through the mapping in the order citations were encountered
        for original_num, new_num in old_to_new_mapping.items():
            # Retrieve original source details using the original number
            source_details = source_details_map.get(original_num)
            if source_details:
                 references.append({
                    "number": new_num, # Use the new sequential number
                    "url": source_details["url"],
                    "title": source_details["title"]
                })
            else:
                 # Should not happen if logic is correct, but handle defensively
                 print(f"Warning: Could not find details for original source number {original_num}")


        # Ensure references are sorted by the new number (already implicitly sorted by OrderedDict iteration)
        # references.sort(key=lambda x: x["number"]) # Technically redundant with OrderedDict

        # Return JSON structure
        result = {
            "cited_text": cited_text,
            "references": references
        }
        try:
            return json.dumps(result, indent=2) # Pretty print JSON
        except Exception as e:
            print(f"Error serializing citation result to JSON: {e}")
            return json.dumps({"cited_text": response_text, "references": []}) # Fallback


    def fill_citation(self, response_text, sources, model="solar-pro-nightly"):
        prompt = f"""Add citation numbers to the response text where information comes from the provided sources.
---
RESPONSE TEXT:
{response_text}
---
SOURCES:
{json.dumps(sources, indent=2)}
---
INSTRUCTIONS:
1. Read the response text and sources carefully.
2. Add citation numbers [1], [2], etc. directly after the specific word or sentence that uses information from the sources. Add citations only for highly relevant information derived from the sources.
3. Only add citation numbers - don't change the original text otherwise.
4. Add 3 to 5 important citations at most.
5. IMPORTANT: Copy URLs EXACTLY as they appear in the sources - URLs must be precise and complete.
6. Return a JSON object with this structure:
   {{
     "cited_text": "text with added citation numbers[1], [2], etc.",
     "references": [
       {{
         "number": 1,
         "url": "https://example.com/source1"
       }}
     ]
   }}

EXAMPLE:

Response: "The iPhone 15 Pro features a titanium frame and a 48-megapixel camera."

Sources: [
  {{"id": 1, "title": "iPhone 15 Pro Review", "url": "https://example.com/review1", "content": "Apple's iPhone 15 Pro features a titanium frame, making it lighter than previous models."}},
  {{"id": 3, "title": "Camera Comparison", "url": "https://example.com/cameras", "content": "With its 48-megapixel main camera, the iPhone 15 Pro captures remarkable detail."}}
]

Output:
{{
  "cited_text": "The iPhone 15 Pro features a titanium frame[1] and a 48-megapixel camera.[2]",
  "references": [
    {{
      "number": 1,
      "url": "https://example.com/review1"
    }},
    {{
      "number": 2,
      "url": "https://example.com/cameras"
    }}
  ]
}}

CRITICAL: 
- Take special care to copy each URL EXACTLY as provided in the sources - do not modify, truncate, or reformat URLs
- Check each URL carefully before including it in the references
- URLs must be complete and functional to ensure proper citation links

Only add citations when there's a clear match between the text and sources. Return valid JSON.
"""

        citation_added_response = self.complete(
            prompt=prompt,
            model=model,
            stream=False
        )
        return citation_added_response
    
    def _standard_request(self, payload):
        """Make a standard non-streaming request."""
        response = requests.post(
            self.base_url,
            headers=self.headers,
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
    
    def _stream_request(self, payload, on_update):
        """Make a streaming request and process the server-sent events."""
        response = requests.post(
            self.base_url,
            headers=self.headers,
            json=payload,
            stream=True
        )
        
        if response.status_code != 200:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
        
        # Create an SSE client from the response
        client = sseclient.SSEClient(response)
        
        # Full content accumulated across all chunks
        full_content = ""
        
        for event in client.events():
            if event.data == "[DONE]":
                break
                
            try:
                chunk = json.loads(event.data)
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    delta = chunk["choices"][0].get("delta", {})
                    if "content" in delta and delta["content"]:
                        content = delta["content"]
                        full_content += content
                        
                        # Call the callback function with the new content
                        if on_update:
                            on_update(content)
            except json.JSONDecodeError:
                pass
        
        return full_content


def extract_search_queries(user_prompt, max_attempts=3, model="solar-pro-nightly"):
    """
    Extract 2-3 optimal search queries from a user prompt to maximize search engine relevance.
    
    Args:
        user_prompt (str): The user's input prompt/question
        max_attempts (int): Maximum number of attempts to get valid JSON response
        
    Returns:
        str: JSON formatted string containing 2-3 search queries
    """
    prompt = f"""
    Given the following user question or request, generate 2-3 different search queries that would help retrieve the most relevant information from a search engine.

    IMPORTANT RULES:
    1. If the request involves a comparison (e.g., "A vs B" or "differences between X and Y"), create separate queries for each component individually (e.g., one query about A, one query about B)
    2. For multi-part questions, divide your queries to address each component separately
    3. Break down complex topics into their fundamental elements
    
    Your search queries should:
    - Extract key concepts and technical terms
    - Remove filler words and focus on essential keywords
    - Be concise and directly relevant to the information need
    - Include specific technical terminology where appropriate
    
    User request: "{user_prompt}"
    
    Examples:
    - User: "How do I implement a binary search tree in Python?"
      Queries: ["python binary search tree implementation", "BST data structure python code", "binary tree algorithms python"]
    
    - User: "What are the advantages of React over Angular for building web applications?"
      Queries: ["React framework features benefits", "Angular framework capabilities", "React vs Angular performance"]
    
    - User: "Explain the difference between supervised and unsupervised machine learning"
      Queries: ["supervised learning algorithms principles", "unsupervised learning methods examples", "machine learning types comparison"]
      
    - User: "Compare AWS Lambda and Google Cloud Functions for serverless applications"
      Queries: ["AWS Lambda serverless features", "Google Cloud Functions capabilities", "serverless platform comparison criteria"]
    
    Return ONLY a JSON object with this format:
    {{"search_queries": ["query1", "query2", "query3"]}}
    """
    
    # Make multiple attempts to get valid JSON
    for attempt in range(max_attempts):
        try:
            # Get completion from Solar API
            response = solar.complete(prompt, model=model, stream=False)
            
            # Try to parse as JSON
            queries = json.loads(response)
            
            # Validate the structure - make sure it has search_queries field
            if "search_queries" not in queries:
                queries = {"search_queries": list(queries.values())[0] if queries else [user_prompt]}
            
            # Ensure we have at most 3 queries
            queries["search_queries"] = queries["search_queries"][:3]
            
            # Return properly formatted JSON
            return json.dumps(queries, indent=2)
            
        except (json.JSONDecodeError, KeyError, IndexError):
            # If this is the last attempt, we'll fall through to the backup method
            if attempt == max_attempts - 1:
                break
            
            # Otherwise, modify the prompt to emphasize JSON formatting
            prompt += "\n\nIMPORTANT: Return ONLY a valid JSON object with the format {\"search_queries\": [\"query1\", \"query2\", \"query3\"]}. No other text."
    
    # Backup method: extract potential queries using regex
    import re
    
    # Look for quoted strings that might be our queries
    queries = re.findall(r'"([^"]*)"', response)
    
    # If we couldn't find any quoted strings, try looking for text between brackets
    if not queries:
        bracket_content = re.search(r'\[(.*?)\]', response)
        if bracket_content:
            queries = [q.strip().strip('"\'') for q in bracket_content.group(1).split(',')]
    
    # If all extraction methods failed, use the original prompt as a single query
    if not queries:
        queries = [user_prompt]
    
    # Ensure we have 2-3 queries (take up to 3)
    queries = queries[:3]
    
    # If we have fewer than 2 queries, add variations of the first query
    while len(queries) < 2 and queries:
        queries.append(f"alternative {queries[0]}")
    
    return json.dumps({"search_queries": queries}, indent=2)

# Example usage
if __name__ == "__main__":
    import os

    solar_r = SolarAPI(base_url="https://r-api.toy.x.upstage.ai/v1/chat/completions", api_key=os.environ.get("SOLAR_R_API_KEY"))
    # Test Solar_r
    print(solar_r.complete("What is the capital of France?", model=None, stream=False))
    
    # Get API key from environment variable
    api_key = os.environ.get("UPSTAGE_API_KEY")
    
    if not api_key:
        api_key = input("Enter your Upstage API key: ")
    
    solar = SolarAPI(api_key)
    
    # Test cases for extract_search_queries function
    print("\n=== TESTING SEARCH QUERY EXTRACTION ===")
    
    test_prompts = [
        "What are the best practices for deploying a Django application to AWS?",
        "Explain how quantum computing works and its potential applications in cryptography",
        "How do I fix a memory leak in my Node.js application?",
        "What are the differences between deep learning and traditional machine learning algorithms?",
        "Give me a recipe for vegetarian lasagna with spinach",
        "민주당과 국민의 힘 경선 일정은 어떻게 돼?"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest Case #{i}: \"{prompt}\"")
        queries_json = extract_search_queries(prompt)
        print("Search Queries:")
        print(queries_json)
        
        # Validate that the output is proper JSON
        try:
            queries = json.loads(queries_json)
            num_queries = len(queries.get("search_queries", []))
            print(f"✓ Valid JSON with {num_queries} queries")
        except json.JSONDecodeError:
            print("✗ Invalid JSON output")
    
    # Check if TAVILY_API_KEY is available for grounding tests
    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    if tavily_api_key:
        print("\n=== TESTING SEARCH GROUNDING ===")
        
        grounding_test_prompts = [
            "What are the latest developments in the Ukraine-Russia conflict?",
            "Tell me about recent advances in AI and their ethical implications",
            "What is the current status of climate change legislation?",
            "How did the stock market perform yesterday?",
            "민주당과 국민의 힘 경선 일정은 언제인가요?",  # Korean: "When are the Democratic Party and People Power Party primaries?"
            "What are the best Python libraries for data visualization in 2024?"
        ]
        
        for i, prompt in enumerate(grounding_test_prompts, 1):
            print(f"\nGrounding Test #{i}: \"{prompt}\"")
            print("Requesting grounded response with sources...")
            
            try:
                # Get grounded response with sources (non-streaming)
                response_data = solar.complete(prompt, search_grounding=True, return_sources=True)
                
                # Extract response and sources
                response_text = response_data["response"]
                sources = response_data["sources"]
                
                # Print the first 300 characters of the response to keep output manageable
                print("\nGrounded Response (excerpt):")
                print(response_text[:300] + "..." if len(response_text) > 300 else response_text)
                print(f"\nFull response length: {len(response_text)} characters")
                
                # Print source information
                print("\nSources used for grounding:")
                for source in sources[:3]:  # Show first 3 sources
                    print(f"  [{source['id']}] {source['title']}")
                    print(f"      URL: {source['url']}")
                    print(f"      Published: {source['published_date']}")
                    print(f"      Relevance score: {source['score']}")
                
                if len(sources) > 3:
                    print(f"  ... and {len(sources) - 3} more sources")
                    
                print("✓ Grounding test with sources successful")
            except Exception as e:
                print(f"✗ Grounding test failed: {str(e)}")
        
        # Add test cases for citation functionality
        print("\n=== TESTING CITATION FUNCTIONALITY ===")
        
        citation_test_cases = [
            {
                "description": "Tech news with multiple sources",
                "response": "The iPhone 15 Pro Max has been well-received by reviewers, with particular praise for its camera system and battery life. The device features a new A17 Pro chip and a titanium frame, making it lighter than previous models.",
                "sources": [
                    {
                        "id": 1,
                        "title": "iPhone 15 Pro Review",
                        "url": "https://example.com/tech/iphone15-review",
                        "content": "The iPhone 15 Pro Max features the new A17 Pro chip, which Apple claims is the fastest mobile processor on the market.",
                        "published_date": "2023-09-22"
                    },
                    {
                        "id": 2,
                        "title": "iPhone 15 Camera Test",
                        "url": "https://example.com/tech/iphone15-camera",
                        "content": "In our extensive testing, the iPhone 15 Pro Max camera system outperformed all competitors in low-light photography.",
                        "published_date": "2023-09-25"
                    },
                    {
                        "id": 3,
                        "title": "Apple's New Materials",
                        "url": "https://example.com/tech/apple-titanium",
                        "content": "The switch to titanium for the iPhone 15 Pro frame reduces the weight by 10% compared to the stainless steel used in the iPhone 14 Pro.",
                        "published_date": "2023-09-20"
                    }
                ]
            },
            {
                "description": "Scientific article with partial citations",
                "response": "Recent studies have shown that regular exercise can reduce the risk of heart disease by up to 30%. Daily meditation has also been linked to lower stress levels and improved cognitive function. However, the relationship between diet and longevity remains complex and requires further research.",
                "sources": [
                    {
                        "id": 1,
                        "title": "Exercise and Heart Health",
                        "url": "https://example.com/health/exercise-heart",
                        "content": "A meta-analysis of 25 studies found that regular physical activity reduced the risk of cardiovascular disease by 25-30% in previously sedentary individuals.",
                        "published_date": "2022-11-15"
                    },
                    {
                        "id": 2,
                        "title": "Meditation Benefits",
                        "url": "https://example.com/health/meditation-brain",
                        "content": "Daily meditation practices of 20 minutes or more have been shown to reduce cortisol levels by 15% and improve performance on cognitive tasks.",
                        "published_date": "2023-03-10"
                    }
                ]
            },
            {
                "description": "News article with no relevant sources",
                "response": "Local authorities announced yesterday that the downtown revitalization project will begin next month. The project is expected to take 18 months to complete and will include new pedestrian walkways and green spaces.",
                "sources": [
                    {
                        "id": 1,
                        "title": "City Budget Allocation",
                        "url": "https://example.com/city/budget-2023",
                        "content": "The city council approved the annual budget with major allocations for infrastructure and education.",
                        "published_date": "2023-01-15"
                    },
                    {
                        "id": 2,
                        "title": "Traffic Pattern Changes",
                        "url": "https://example.com/city/traffic-updates",
                        "content": "Due to construction on Highway 101, commuters should expect delays during morning rush hour for the next three weeks.",
                        "published_date": "2023-05-22"
                    }
                ]
            }
        ]
        
        for i, test_case in enumerate(citation_test_cases, 1):
            print(f"\nCitation Test #{i}: {test_case['description']}")
            print("\nOriginal Response:")
            print(test_case['response'])
            print("\nSources Available:")
            for source in test_case['sources']:
                print(f"  [{source['id']}] {source['title']} - {source['url']}")
            
            try:
                # Get cited response
                print("\nProcessing citations...")
                cited_response = solar.fill_citation(
                    response_text=test_case['response'],
                    sources=test_case['sources']
                )
                
                print("\nCited Response Result:")
                
                # Try to parse the result as JSON
                try:
                    cited_data = json.loads(cited_response)
                    print("\nCited Text:")
                    print(cited_data.get("cited_text", "No cited text found"))
                    
                    print("\nReferences:")
                    for ref in cited_data.get("references", []):
                        print(f"  [{ref.get('number')}] {ref.get('url')}")
                        print(f"      \"{ref.get('snippet')}\"")
                    
                    print(f"\n✓ Successfully parsed citation result as JSON with {len(cited_data.get('references', []))} references")
                except json.JSONDecodeError:
                    # If not JSON, just print the raw result
                    print("Raw result (not valid JSON):")
                    print(cited_response)
                    print("\n✗ Failed to parse result as JSON")
            
            except Exception as e:
                print(f"\n✗ Citation test failed: {str(e)}")
    
    print("\n=== ORIGINAL API EXAMPLES ===")
    
    # Example for non-streaming request
    prompt = "What is the capital of France?"
    response = solar.complete(prompt, stream=False)
    print("\nNon-streaming response:")
    print(response)
    
    # Example for streaming request
    print("\nStreaming response:")
    
    def print_update(content):
        print(content, end="", flush=True)
    
    solar.complete(prompt, stream=True, on_update=print_update)
    print("\n")
    
    # Example for grounding with sources (if Tavily API key is available)
    if tavily_api_key:
        print("\n=== DETAILED GROUNDING EXAMPLE ===")
        prompt = "What were the major tech announcements at CES this year?"
        
        print(f"Question: {prompt}")
        print("Requesting grounded response with sources...")
        
        response_data = solar.complete(prompt, search_grounding=True, return_sources=True)
        
        print("\nResponse:")
        print(response_data["response"])
        
        print("\nSources:")
        for source in response_data["sources"]:
            print(f"[{source['id']}] {source['title']}")
            print(f"    URL: {source['url']}")
            print(f"    Date: {source['published_date']}")
        
        # Example for streaming with grounding
        print("\nStreaming response with grounding:")
        prompt = "What are the trending programming languages in 2024?"
        
        def print_grounded_update(content):
            print(content, end="", flush=True)
        
        solar.complete(prompt, stream=True, on_update=print_grounded_update, search_grounding=True)
        print("\n")