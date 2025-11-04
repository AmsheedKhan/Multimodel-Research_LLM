
import streamlit as st
import requests, io, arxiv
from PIL import Image
from bs4 import BeautifulSoup
from google import genai


st.set_page_config(page_title="LLM Agents Lab", layout="wide")

st.sidebar.title(" API Keys")
geminikey = st.sidebar.text_input("Gemini API Key", type="password")
hf_key = st.sidebar.text_input("Hugging Face API Key", type="password")

# Initialize Gemini client if key is present
client = None
if geminikey:
    try:
        client = genai.Client(api_key=geminikey)
    except Exception as e:
        st.sidebar.error(f"Error initializing Gemini client: {e}")



def call_hf_img(prompt, hf_token):
    """Use Hugging Face router inference endpoint for image generation."""
    if not hf_token:
        return None, " No Hugging Face token provided."
    API_URL = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {"Authorization": f"Bearer {hf_token}", "Accept": "image/png"}
    payload = {"inputs": prompt}
    try:
        r = requests.post(API_URL, headers=headers, json=payload, timeout=90)
        if r.status_code == 200:
            img = Image.open(io.BytesIO(r.content))
            return img, None
        else:
            return None, f"Error {r.status_code}: {r.text}"
    except Exception as e:
        return None, f"Exception: {e}"

def call_gemini_text(prompt):
    """Generate text using Gemini."""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Gemini error: {e}"

def classify_intent(user_text: str):
   
    t = user_text.lower()
    if any(k in t for k in ["image", "picture", "photo", "draw", "generate an image"]):
        return "image"
    return "text"

def handle_user_req(user_text, hf_token):
    
    intent = classify_intent(user_text)
    if intent == "text":
        return {"Type": "Text", "content": call_gemini_text(user_text)}
    elif intent == "image":
        img, err = call_hf_img(user_text, hf_token)
        return {"Type": "Image", "content": img, "error": err}
    else:
        return {"Type": "Text", "content": "I don't understand."}


def search_arxiv(query, max_results=5):
   
    search = arxiv.Search(query=query, max_results=max_results,
                          sort_by=arxiv.SortCriterion.SubmittedDate)
    results = []
    for r in search.results():
        results.append({"title": r.title, "summary": r.summary, "url": r.entry_id})
    return results

def scrape_web_text(url):
    
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        paras = [p.get_text() for p in soup.find_all("p")]
        return " ".join(paras)
    except Exception as e:
        return f"Error scraping {url}: {e}"

def chunk_text(text, max_len=3000):
    
    words = text.split()
    for i in range(0, len(words), max_len):
        yield " ".join(words[i:i+max_len])

def summarize_chunk(text):
    
    prompt = f"You are a helpful research assistant. Summarize this text:\n{text}"
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text

def summarize_long_text(text):
   
    chunks = list(chunk_text(text))
    summaries = [summarize_chunk(c) for c in chunks]
    combined = " ".join(summaries)
    final_prompt = f"Combine and simplify the following summaries:\n{combined}"
    response = client.models.generate_content(model="gemini-2.5-flash", contents=final_prompt)
    return response.text

def research_digest(query, n_papers=3, use_scraping=False):

    papers = search_arxiv(query, n_papers)
    final_summaries = []
    for paper in papers:
        st.markdown(f"###  {paper['title']}")
        text = scrape_web_text(paper["url"]) if use_scraping else paper["summary"]
        summary = summarize_long_text(text)
        final_summaries.append({"title": paper["title"], "summary": summary, "url": paper["url"]})
        with st.expander(" View Summary"):
            st.write(summary)
            st.markdown(f"[Open Paper]({paper['url']})")
    return final_summaries


st.title(" Multi-Modal & Research LLM Agent")
st.markdown("Built with Google Gemini (Text) + Hugging Face Stable Diffusion XL (Images)")

tabs = st.tabs(["Multi-Modal Agent", " Research Assistant"])


with tabs[0]:
    st.subheader(" Multi-Modal Interactive Agent")
    user_prompt = st.text_input("Enter your prompt (e.g., *Generate an image of a futuristic city* or *Explain agentic AI*):")

    if st.button("Run Agent", key="run_mm"):
        if not geminikey:
            st.error("Please enter your Gemini API key in the sidebar.")
        else:
            with st.spinner("Processing request..."):
                output = handle_user_req(user_prompt, hf_key)
            if output["Type"] == "Text":
                st.success(output["content"])
            elif output["Type"] == "Image":
                if output.get("error"):
                    st.error(output["error"])
                else:
                    st.image(output["content"], caption=user_prompt, use_container_width=True)


with tabs[1]:
    st.subheader(" Research Learning Assistant")
    query = st.text_input("Enter a research topic:", "large language models in healthcare")
    n_papers = st.slider("Number of papers:", 1, 10, 3)
    use_scraping = st.checkbox("Also scrape full text from paper URLs (slower)", value=False)

    if st.button("Run Research Agent", key="run_ra"):
        if not geminikey:
            st.error("Please enter your Gemini API key in the sidebar.")
        else:
            with st.spinner("Searching and summarizing papers..."):
                results = research_digest(query, n_papers, use_scraping)
            st.success(f" Summarized {len(results)} papers successfully!")
            all_text = "\n\n".join([f"{r['title']}\n{r['summary']}" for r in results])
            st.download_button(" Download All Summaries", all_text, file_name="research_summaries.txt")

st.markdown("---")
