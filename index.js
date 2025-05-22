const express = require("express");
const cors = require("cors");
const path = require("path");
const { OpenAIEmbeddings } = require("@langchain/openai");
const { ChatOpenAI } = require("@langchain/openai");
const { FaissStore } = require("@langchain/community/vectorstores/faiss");
require("dotenv").config();

const app = express();
app.use(cors());
app.use(express.json());

// Your existing route
app.use("/triage-full", require("./routes/triageFull"));

// Test retriever route
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
      documents: docs.map(d => ({
        content: d.pageContent,
        metadata: d.metadata
      }))
    });
  } catch (err) {
    console.error("Retriever test error:", err);
    res.status(500).json({
      success: false,
      error: err.message,
      details: "Check if FAISS index exists and OPENAI_API_KEY is valid"
    });
  }
});

// Simple test route
app.get("/test", (req, res) => {
  res.json({ status: "Server is working" });
});

app.get("/test-openai", async (req, res) => {
  const llm = new ChatOpenAI({ openAIApiKey: process.env.OPENAI_API_KEY });
  const response = await llm.invoke("Hello world");
  res.json(response);
});
app.listen(3001, () => console.log("Server running on http://localhost:3001"));