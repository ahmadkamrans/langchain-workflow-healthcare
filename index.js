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

app.use("/triage-hybrid", require("./routes/triageHybrid"));

app.listen(3001, () => console.log("Server running on http://localhost:3001"));
