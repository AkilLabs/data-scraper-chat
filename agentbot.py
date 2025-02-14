import os
from typing import List, Dict, Optional
from crewai import Agent, Task, Crew, Process, LLM
from langchain_google_genai import ChatGoogleGenerativeAI
from duckduckgo_search import DDGS
from playwright.sync_api import sync_playwright
from playwright_stealth import stealth
from bs4 import BeautifulSoup
import requests
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential
from concurrent.futures import ThreadPoolExecutor, as_completed

# API Keys
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "api")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "api")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "api")

# Configure Gemini AI
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-pro")

class WebResearchTools:
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def search_google(self, query: str) -> List[str]:
        url = "https://google.serper.dev/search"
        headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
        payload = {"q": query}

        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            results = response.json()
            return [result["link"] for result in results.get("organic", [])][:5]
        except requests.exceptions.RequestException as e:
            print(f"Google Search Error: {e}")
            return []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def search_duckduckgo(self, query: str, max_results: int = 5) -> List[str]:
        try:
            with DDGS() as ddgs:
                return [result["href"] for result in ddgs.text(query, max_results=max_results)]
        except Exception as e:
            print(f"DuckDuckGo Search Error: {e}")
            return []

    def scrape_with_playwright(self, url: str) -> Optional[Dict[str, str]]:
        try:
            with sync_playwright() as p:
                browser = p.firefox.launch(headless=True)  # Use Firefox to bypass bot detection
                context = browser.new_context(
                    viewport={'width': 1920, 'height': 1080},
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                )
                page = context.new_page()
                stealth(page)  # Enable stealth mode
                page.goto(url, timeout=60000, wait_until='domcontentloaded')
                page.wait_for_selector('body', timeout=15000)
                html_content = page.content()
                title = page.title()
                context.close()
                browser.close()

            soup = BeautifulSoup(html_content, "html.parser")
            for element in soup.find_all(['script', 'style', 'nav', 'footer']):
                element.decompose()

            text_elements = [
                element.get_text(strip=True) for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'article', 'section'])
                if len(element.get_text(strip=True)) > 50
            ]

            return {
                "url": url,
                "title": title,
                "content": "\n\n".join(text_elements)[:4000]
            }
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")
            return None

class WebResearchCrew:
    def __init__(self):
        self.tools = WebResearchTools()

        self.llm = LLM(
            model="groq/llama3-8b-8192",
            temperature=0.3,
            max_tokens=4096,
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1"
        )

        # Initialize Agents
        self.researcher = Agent(
            role='Research Analyst',
            goal='Find and analyze relevant information from multiple web sources',
            backstory="Expert research analyst with experience in data gathering and pattern recognition.",
            allow_delegation=False,
            llm=self.llm
        )

        self.writer = Agent(
            role='Content Writer',
            goal='Create comprehensive and well-structured summaries from research data',
            backstory="Skilled writer who organizes information in an engaging and clear format.",
            allow_delegation=False,
            llm=self.llm
        )

        self.fact_checker = Agent(
            role='Fact Checker',
            goal='Verify information accuracy and identify potential biases',
            backstory="Detail-oriented fact-checker responsible for verifying research accuracy.",
            allow_delegation=False,
            llm=self.llm
        )

    def research_topic(self, query: str) -> str:
        # Get URLs from search engines
        google_results = self.tools.search_google(query)
        duckduckgo_results = self.tools.search_duckduckgo(query)
        all_urls = list(dict.fromkeys(google_results + duckduckgo_results))

        # Scrape content from URLs
        scraped_data = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_url = {executor.submit(self.tools.scrape_with_playwright, url): url for url in all_urls[:8]}
            for future in as_completed(future_to_url):
                result = future.result()
                if result:
                    scraped_data.append(result)

        if not scraped_data:
            return "Failed to gather information from web sources."

        # Create tasks for the crew
        research_task = Task(
            description=f"Analyze the following web content related to: {query}\n\n{scraped_data}",
            agent=self.researcher,
            expected_output="A structured analysis of key insights from web sources."
        )

        fact_check_task = Task(
            description="Review research findings and verify accuracy. Identify inconsistencies or biases.",
            agent=self.fact_checker,
            expected_output="A verified and fact-checked summary of research findings."
        )

        writing_task = Task(
            description="Create a well-structured summary of verified research. Highlight key points clearly.",
            agent=self.writer,
            expected_output="A clear, engaging, and informative summary of the research topic."
        )

        # Create and run the crew
        crew = Crew(
            agents=[self.researcher, self.fact_checker, self.writer],
            tasks=[research_task, fact_check_task, writing_task],
            verbose=2
        )

        result = crew.kickoff()
        return result

def main():
    try:
        query = input("Enter your research query: ")
        print("\nInitiating research crew...")

        crew = WebResearchCrew()
        result = crew.research_topic(query)

        print("\n=== Research Results ===\n")
        print(result)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
