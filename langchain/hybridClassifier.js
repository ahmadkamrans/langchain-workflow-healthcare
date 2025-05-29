// langchain/hybridClassifier.js
const path = require("path");
require("dotenv").config();
const { FaissStore } = require("@langchain/community/vectorstores/faiss");
const { OpenAIEmbeddings, ChatOpenAI } = require("@langchain/openai");
const { TavilySearchResults } = require("@langchain/community/tools/tavily_search");
const { PromptTemplate } = require("@langchain/core/prompts");
const { traceable } = require("langsmith/traceable");
let faissStorePromise;
let llm;
let tool;
let initialized = false;
(async () => {
  const embeddings = new OpenAIEmbeddings({
    openAIApiKey: process.env.OPENAI_API_KEY,
    modelName: "text-embedding-3-small",
  });
  faissStorePromise = FaissStore.load(
    path.join(__dirname, "../faiss_index"),
    embeddings
  );
  llm = new ChatOpenAI({
    openAIApiKey: process.env.OPENAI_API_KEY,
    modelName: "gpt-3.5-turbo",
    temperature: 0,
  });
  // Configure Tavily once, including maxResults
  tool = new TavilySearchResults({
    apiKey: process.env.TAVILY_API_KEY,
    maxResults: 5,
    // includeAnswer: true,
    // includeRawContent: true,
    // includeImages: false,
  });
  initialized = true;
})();
// Input validator remains unchanged
const isHealthcareRelated = traceable(
  async (description) => {
    const validator = new ChatOpenAI({
      openAIApiKey: process.env.OPENAI_API_KEY,
      modelName: "gpt-3.5-turbo",
      temperature: 0,
    });
    const systemPrompt = `
You are a healthcare input validator.
Return true if the user's input is a symptom or medical issue that could justify triage...
Respond ONLY with "true" or "false".
`;
    const result = await validator.invoke([
      { role: "system", content: systemPrompt },
      { role: "user", content: description }
    ]);
    return result.content.trim().toLowerCase() === "true";
  },
  {
    name: "isHealthcareRelated",
    projectName: process.env.LANGCHAIN_PROJECT || "RAG_Healthcare",
  }
);
// RAG Classifier with the fix
const classifyWithIntelligentRAG = traceable(
  async (description) => {
    if (!initialized) {
      throw new Error("Hybrid classifier not initialized yet.");
    }
    const faissStore = await faissStorePromise;
    const internalResults = await faissStore.similaritySearchWithScore(description, 5);
    const scoredDocs = internalResults
      .map(([doc, score]) => ({ content: doc.pageContent, score }))
      .sort((a, b) => a.score - b.score);
    const internalContext = scoredDocs.map(d => d.content).join("\n---\n");
    let internetSnippets = "No internet data found.";
    let internetSuccess = false;
    try {
      console.log(":mag: Searching Tavily with:", description.trim());
      const raw = await tool.call({ input: description.trim() });
      console.log("raw Tavily output:", raw);
      let arr = raw;
      if (typeof raw === "string") {
        try {
          arr = JSON.parse(raw);
        } catch {
          arr = [];
        }
      }
      if (Array.isArray(arr) && arr.length > 0) {
        internetSnippets = arr.map((r) => r.content).join("\n---\n");
        internetSuccess = true;
      }
    } catch (err) {
      console.error("Internet search failed:", err);
    }
    const template = new PromptTemplate({
      inputVariables: ["description", "internalContext", "internetSnippets"],
      template: `
You are an advanced medical triage classifier...
Respond only in this JSON format:
{{
  "urgency_level": "Emergency" | "Urgent Care" | "Non-Urgent" | "Follow-Up Needed" | "Unknown",
  "category": string,
  "context_used": "internal" | "internet" | "both" | "none",
  "used_doc": string,
  "recommendation": string
}}
[Symptom Description]
{description}

[Internal Documents]
{internalContext}

[Internet Snippets]
{internetSnippets}
`
    });
    const finalPrompt = await template.format({
      description,
      internalContext,
      internetSnippets,
    });
    const response = await llm.invoke(finalPrompt);
    let parsed;
    try {
      parsed = JSON.parse(response.content || "{}");
    } catch {
      parsed = {
        urgency_level: "Unknown",
        category: "Unknown",
        context_used: "none",
        used_doc: "Parsing error",
        recommendation: "Unable to classify. Please consult a provider."
      };
    }
    return {
      ...parsed,
      similarity_scores: scoredDocs,
      internet_info_used: internetSuccess,
    };
  },
  {
    name: "classifyWithIntelligentRAG",
    projectName: process.env.LANGCHAIN_PROJECT || "RAG_Healthcare",
  }
);
module.exports = {
  classifyWithIntelligentRAG,
  isHealthcareRelated,
};