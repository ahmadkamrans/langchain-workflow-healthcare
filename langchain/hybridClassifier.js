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
// Use async IIFE to initialize async dependencies
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
  // Moved maxResults into constructor, per LangChain.js usage
  tool = new TavilySearchResults({
    apiKey: process.env.TAVILY_API_KEY,
    maxResults: 5,
    // you can also enable the following if desired:
    // includeAnswer: true,
    // includeRawContent: true,
    // includeImages: false,
  });
})();
// Input Validator
const isHealthcareRelated = traceable(
  async (description) => {
    const validator = new ChatOpenAI({
      openAIApiKey: process.env.OPENAI_API_KEY,
      modelName: "gpt-3.5-turbo",
      temperature: 0,
    });
    const systemPrompt = `
You are a healthcare input validator.
Return true if the user's input is a symptom or medical issue that could justify triage (e.g., bleeding, chest pain, dizziness, shortness of breath).
Even if the input is brief or lacks detail, as long as it clearly refers to a health-related issue, return true.
Only return false if the input is completely unrelated (e.g., "aliens in my soup") or too vague (e.g., "I feel off", "weird stuff", "bad vibe").
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
// RAG Classifier
const classifyWithIntelligentRAG = traceable(
  async (description) => {
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
      // Invoke with a plain string input, not an object
      const tavilyResults = await tool.invoke(description.trim());
      if (Array.isArray(tavilyResults)) {
        internetSnippets = tavilyResults.map(r => r.content).join("\n---\n");
        internetSuccess = true;
      }
    } catch (err) {
      console.error("Internet search failed:", err.message || err);
    }
    const template = new PromptTemplate({
      inputVariables: ["description", "internalContext", "internetSnippets"],
      template: `
You are an advanced medical triage classifier.
You are given:
- Internal documents from clinical sources.
- Internet search snippets with public information.
Your task is to:
1. Decide which context (internal, internet, or both) is more relevant for classification.
2. Classify the symptom using the most helpful information.
If the input is vague or unclassifiable, return "Unknown".
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
    } catch (err) {
      console.error("Parsing failed:", response.content);
      parsed = {
        urgency_level: "Unknown",
        category: "Unknown",
        context_used: "none",
        used_doc: "Parsing error",
        recommendation: "We could not classify your symptom. Please consult a healthcare provider."
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