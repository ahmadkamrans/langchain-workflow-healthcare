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
You are a strict classifier. Only return true if the user's input is clearly related to health symptoms, illnesses, medical conditions, or treatments.
Otherwise, return false. Respond only with "true" or "false".
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

  const usedDocs = resultsWithScores.map(([doc, score]) => doc.pageContent);
  const context = usedDocs.join("\n---\n");

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
  "used_doc": string
}}

Context:
{context}

Patient Symptom:
{input}
    `
  });

  const llmPrompt = await prompt.format({
    context,
    input: description,
  });

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
      used_doc: "None"
    };
  }

  // If unknown, now fallback to agent + internet
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

Be concise and respond in this JSON format:

{
  "urgency_level": "...",
  "category": "...",
  "internet_info_used": true,
  "used_doc": "Used Tavily search results here"
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
      };
    }

    return {
      ...finalParsed,
      top_context_used: finalParsed.used_doc || "Internet context used",
    };
  }

  return {
    ...parsed,
    top_context_used: parsed.used_doc || "No doc clearly supported classification",
  };
};


module.exports = {
  classifyWithHybridRAG,
  isHealthcareRelated,
};
