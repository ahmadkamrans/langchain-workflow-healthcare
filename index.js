// index.js
require("dotenv").config();
require("langsmith"); // initialize LangSmith tracing via environment variables
const express = require("express");
const cors = require("cors");
const path = require("path");
const { OpenAIEmbeddings } = require("@langchain/openai");
const { ChatOpenAI } = require("@langchain/openai");
const { FaissStore } = require("@langchain/community/vectorstores/faiss");

const app = express();
app.use(cors());
app.use(express.json());
// Routes
app.use("/triage-full", require("./routes/triageFull"));
// Test retriever
app.get("/test-retriever", async (req, res) => {
  try {
    const store = await FaissStore.load(
      path.join(__dirname, "faiss_index"),
      new OpenAIEmbeddings({
        openAIApiKey: process.env.OPENAI_API_KEY,
        modelName: "text-embedding-3-small"
      })
    );
    const docs = await store.asRetriever().getRelevantDocuments("chest pain");
    res.json({
      success: true,
      documents: docs.map(d => ({ content: d.pageContent, metadata: d.metadata }))
    });
  } catch (err) {
    console.error("Retriever test error:", err);
    res.status(500).json({ success: false, error: err.message });
  }
});
// Simple test
app.get("/test", (req, res) => res.json({ status: "Server is working" }));
app.get("/test-openai", async (req, res) => {
  const llm = new ChatOpenAI({ openAIApiKey: process.env.OPENAI_API_KEY });
  const response = await llm.invoke("Hello world");
  res.json(response);
});
app.listen(3001, () => console.log("Server running on http://localhost:3001"));
