const path = require("path");
const { FaissStore } = require("@langchain/community/vectorstores/faiss");
const { OpenAIEmbeddings, ChatOpenAI } = require("@langchain/openai");
const { TavilySearchResults } = require("@langchain/community/tools/tavily_search");
const { PromptTemplate } = require("@langchain/core/prompts");
require("dotenv").config();

const embeddings = new OpenAIEmbeddings({
  openAIApiKey: process.env.OPENAI_API_KEY,
  modelName: "text-embedding-3-small",
});

const faissStorePromise = FaissStore.load(
  path.join(__dirname, "../faiss_index"),
  embeddings
);

const llm = new ChatOpenAI({
  openAIApiKey: process.env.OPENAI_API_KEY,
  modelName: "gpt-3.5-turbo", // ✅ using GPT-3.5
  temperature: 0,
});

const tool = new TavilySearchResults({ apiKey: process.env.TAVILY_API_KEY });

const isHealthcareRelated = async (description) => {
  const checkLLM = new ChatOpenAI({
    openAIApiKey: process.env.OPENAI_API_KEY,
    modelName: "gpt-3.5-turbo", // ✅ lighter for input validation
    temperature: 0,
  });

  const systemPrompt = `
You are a strict healthcare input validator.
Only return true if the user has clearly described one or more symptoms (e.g., chest pain, sore throat, fatigue) that could reasonably allow medical triage.
If the input is too vague (e.g., "not feeling well", "kidney beans and heart something", "weird body issue"), or not a health symptom at all, return false.
Respond only with "true" or "false".
`;

  const result = await checkLLM.invoke([
    { role: "system", content: systemPrompt },
    { role: "user", content: description }
  ]);

  return result.content.trim().toLowerCase() === "true";
};

const classifyWithHybridRAG = async (description) => {
  const faissStore = await faissStorePromise;

  // Step 1: Retrieve internal document context
  const resultsWithScores = await faissStore.similaritySearchWithScore(description, 5);
  const scoredDocs = resultsWithScores
    .map(([doc, score]) => ({ content: doc.pageContent, score }))
    .sort((a, b) => a.score - b.score);
  const internalDocs = scoredDocs.map((d) => d.content).join("\n---\n");
  const topContextUsed = scoredDocs[0]?.content || "None";

  // Step 2: Retrieve internet context via Tavily
  let internetSnippets = "No internet data found.";
  try {
    if (description && description.trim().length > 5) {
      const tavilyResults = await tool.call({ query: description.trim(), numResults: 5 });
      if (Array.isArray(tavilyResults)) {
        internetSnippets = tavilyResults.map(r => r.content).join("\n---\n");
      }
    }
  } catch (e) {
    console.error("Error getting Tavily internet context:", e.message || e);
  }

  // Step 3: Unified Prompt
  const unifiedPrompt = new PromptTemplate({
    inputVariables: ["description", "internalDocs", "internetSnippets"],
    template: `
You are a highly cautious and knowledgeable medical triage assistant.

You are given:
- Internal documentation with symptom examples and triage categories.
- Internet search snippets for supplemental understanding.
- A patient symptom description.

Based on both sources, classify the input. If it's vague or unclassifiable, return "Unknown".

Respond in this JSON format:
{{
  "urgency_level": "Emergency" | "Urgent Care" | "Non-Urgent" | "Follow-Up Needed" | "Unknown",
  "category": string,
  "internet_info_used": true | false,
  "used_doc": string,
  "recommendation": string
}}

Helpful context for classification:
[Internal Docs]
{internalDocs}

[Internet Info]
{internetSnippets}

Symptom Description:
{description}
    `
  });

  let llmPrompt;
  try {
    llmPrompt = await unifiedPrompt.format({
      description,
      internalDocs,
      internetSnippets,
    });
  } catch (err) {
    console.error("Prompt formatting error:", err);
    throw new Error("Prompt formatting failed.");
  }

  const response = await llm.invoke(llmPrompt);

  let parsed;
  try {
    parsed = JSON.parse(response.content || "{}");
  } catch (e) {
    console.error("Error parsing unified RAG response:", response.content);
    parsed = {
      urgency_level: "Unknown",
      category: "Unknown",
      internet_info_used: true,
      used_doc: "Parsing error",
      recommendation: "We could not classify your symptom confidently. Please consult a medical professional."
    };
  }

  return {
    ...parsed,
    top_context_used: topContextUsed,
    similarity_scores: scoredDocs,
  };
};

module.exports = {
  classifyWithHybridRAG,
  isHealthcareRelated,
};
