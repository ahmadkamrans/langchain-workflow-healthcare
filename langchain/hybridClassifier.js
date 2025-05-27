// langchain/hybridClassifier.js
const path = require("path");
const { FaissStore } = require("@langchain/community/vectorstores/faiss");
const { OpenAIEmbeddings, ChatOpenAI } = require("@langchain/openai");
const { TavilySearchResults } = require("@langchain/community/tools/tavily_search");
const { initializeAgentExecutorWithOptions } = require("langchain/agents");
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
  modelName: "gpt-4",
  temperature: 0,
});

const tool = new TavilySearchResults({ apiKey: process.env.TAVILY_API_KEY });

const isHealthcareRelated = async (description) => {
  const checkLLM = new ChatOpenAI({
    openAIApiKey: process.env.OPENAI_API_KEY,
    modelName: "gpt-4",
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
  const resultsWithScores = await faissStore.similaritySearchWithScore(description, 5);
  const scoredDocs = resultsWithScores
    .map(([doc, score]) => ({
      content: doc.pageContent,
      score,
    }))
    .sort((a, b) => a.score - b.score); // lower = more similar
  if (scoredDocs.length === 0) {
    console.warn("No FAISS docs matched. Skipping to internet fallback.");
    return {
      urgency_level: "Unknown",
      category: "Unknown",
      internet_info_used: false,
      used_doc: "No matching FAISS docs",
      top_context_used: "None",
      similarity_scores: [],
      recommendation: "We could not find relevant internal references. Please consult a physician for evaluation."
    };
  }
  const usedDocs = scoredDocs.map((d) => d.content);
  const context = usedDocs.join("\n---\n");
  const topContextUsed = scoredDocs[0]?.content || "None";
  const prompt = new PromptTemplate({
    inputVariables: ["context", "input"],
    template: `
You are a highly cautious and knowledgeable medical triage assistant.
Use the provided internal documentation to classify the symptom.
If it is vague or no context supports a confident answer, return "Unknown" for both fields.
Respond in this JSON format:
{{
  "urgency_level": "Emergency" | "Urgent Care" | "Non-Urgent" | "Follow-Up Needed" | "Unknown",
  "category": string,
  "internet_info_used": false,
  "used_doc": string,
  "recommendation": string
}}
Based on the classification and the used document, provide a friendly 1-2 sentence recommendation to help the user understand what they should do next.
Context:
{context}
Patient Symptom:
{input}
    `
  });
  let llmPrompt;
  try {
    llmPrompt = await prompt.format({
      context: String(context),
      input: String(description),
    });
  } catch (err) {
    console.error("Error formatting prompt template:", err);
    throw new Error("Prompt formatting failed.");
  }
  const response = await llm.invoke(llmPrompt);
  let parsed;
  try {
    parsed = JSON.parse(response.content || "{}");
  } catch (e) {
    console.error("Error parsing FAISS classification response:", response.content);
    parsed = {
      urgency_level: "Unknown",
      category: "Unknown",
      internet_info_used: false,
      used_doc: "Parsing error from FAISS response",
      recommendation: "We could not classify your symptom confidently. Please seek professional medical advice."
    };
  }
  // Fallback to internet-based classification if needed
  if (parsed.urgency_level === "Unknown" || parsed.category === "Unknown") {
    const agentExecutor = await initializeAgentExecutorWithOptions(
      [tool],
      llm,
      {
        agentType: "openai-functions",
        verbose: true,
      }
    );
    const agentPrompt = `
Given this health symptom: "${description}", use internet search to help determine:
1. Urgency level: "Emergency", "Urgent Care", "Non-Urgent", "Follow-Up Needed", or "Unknown"
2. Category: e.g., "Cardiac", "Allergy", "Neurological", or "Unknown"
Respond in this JSON format:
{
  "urgency_level": "...",
  "category": "...",
  "internet_info_used": true,
  "used_doc": "Used Tavily search results here",
  "recommendation": "A medically cautious recommendation based on the above. Suggest consulting a professional if unclear."
}
`;
    const agentResponse = await agentExecutor.invoke({
      input: agentPrompt,
    });
    let finalParsed;
    try {
      finalParsed = JSON.parse(agentResponse.output || "{}");
    } catch {
      finalParsed = {
        urgency_level: "Unknown",
        category: "Unknown",
        internet_info_used: true,
        used_doc: "Parsing failed from internet agent",
        recommendation: "Your input was vague, and we could not classify it confidently. Please consult a medical professional."
      };
    }
    return {
      ...finalParsed,
      top_context_used: topContextUsed,
      similarity_scores: scoredDocs,
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