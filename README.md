<h1>🩺 Healthcare Triage AI with LangChain, FAISS, and LangSmith</h1>

<p>
A Node.js-based healthcare triage assistant that uses <strong>LangChain</strong>, <strong>FAISS</strong> for vector search, and <strong>LangSmith</strong> for tracing and observability. It classifies user symptom descriptions into urgency levels and medical categories using context-aware RAG (Retrieval-Augmented Generation).
</p>

<h2>🚀 Features</h2>
<ul>
  <li>🔍 FAISS vector store powered by OpenAI Embeddings</li>
  <li>🧠 LangChain RAG pipeline with structured LLM output</li>
  <li>📊 LangSmith integration for full traceability</li>
  <li>💬 Classifies symptoms into <code>urgency_level</code> and <code>category</code></li>
  <li>📈 Logs similarity scores of retrieved documents</li>
</ul>

<h2>🧱 Stack</h2>
<ul>
  <li>Node.js + Express</li>
  <li>LangChain (LLM orchestration)</li>
  <li>FAISS (Vector similarity search)</li>
  <li>OpenAI (Embeddings + GPT model)</li>
  <li>LangSmith (Tracing and observability)</li>
</ul>

<h2>📦 Installation</h2>

<pre><code>git clone https://github.com/your-username/rag-healthcare-triage.git
cd rag-healthcare-triage
npm install
</code></pre>

<h3>🌐 Environment Variables (.env)</h3>
<pre><code>OPENAI_API_KEY=your_openai_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=Healthcare-Triage-RAG
</code></pre>

<h2>▶️ Running the App</h2>
<pre><code>node index.js</code></pre>

<h2>📡 API Endpoints</h2>
<ul>
  <li><strong>POST</strong> <code>/triage-full</code> - Classifies a symptom description</li>
</ul>

<h4>Example Request</h4>
<pre><code>POST http://localhost:3001/triage-full
Content-Type: application/json

{
  "description": "I have a mild sore throat and slight fever."
}
</code></pre>

<h4>Example Response</h4>
<pre><code>{
  "success": true,
  "urgency_level": "Non-Urgent",
  "category": "Infection"
}
</code></pre>

<h2>📈 LangSmith Tracing</h2>
<ul>
  <li>Function <code>classifyWithRAG</code> is traced with clean input context</li>
  <li>Similarity scores from FAISS are injected into the LLM context</li>
  <li>No circular JSON warnings</li>
</ul>

<h2>🧪 Test Routes</h2>
<ul>
  <li><code>GET /test</code> - Basic server test</li>
  <li><code>GET /test-openai</code> - Verifies OpenAI connection</li>
  <li><code>GET /test-retriever</code> - Fetches top vector matches for "chest pain"</li>
</ul>

<h2>📂 Project Structure</h2>
<pre><code>langchain/
├── ragClassifier.js     # RAG + LangChain pipeline
routes/
├── triageFull.js        # API route handler
index.js                 # Main server entry point
faiss_index/             # Pre-built FAISS vector store
</code></pre>

<h2>🙌 Credits</h2>
<ul>
  <li>Inspired by real-world clinical triage needs</li>
  <li>Powered by OpenAI, LangChain, and LangSmith</li>
</ul>

<h2>📜 License</h2>
<p>MIT License</p>
