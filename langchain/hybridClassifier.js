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
  const resultsWithScores = await faissStore.similaritySearchWithScore(description, 5); // top 5

  const usedDocs = resultsWithScores.map(([doc, score], i) => {
    console.log(`🔍 Similarity Score [${i + 1}]:`, score.toFixed(4));
    return doc.pageContent;
  });

  // Join used docs with separator to form the context string for prompt
  const context = usedDocs.join("\n---\n");

  const prompt = new PromptTemplate({
    inputVariables: ["context", "input"],
    template: `
You are a highly cautious and knowledgeable medical triage assistant.

You may be asked about any medical condition or health-related symptom. Use the provided internal documentation and internet search results if necessary. Always prefer internal documentation if it is sufficient.

Only provide a classification if you are confident based on clear, specific information.  
If the symptom input is vague or lacks detail (e.g., "feeling sick", "not well", "unwell", etc.), do **not** make assumptions.  
In such cases, set both "urgency_level" and "category" to "Unknown".

The "category" can include any relevant medical condition area such as:
- Cardiac, Flu, Allergy, Mental Health, Gastrointestinal, Neurological, Musculoskeletal, Respiratory, Dermatological, etc.  
- Use other categories if more appropriate for the symptom.  
- If unsure, use "Unknown".

From the documents below, pick the single document that best supports your classification and include it exactly as-is in the field "used_doc".

Respond strictly in this JSON format:

{{
  "urgency_level": "Emergency" | "Urgent Care" | "Non-Urgent" | "Follow-Up Needed" | "Unknown",
  "category": string (e.g., "Cardiac", "Infection", "Neurological", "Unknown", etc.),
  "internet_info_used": true | false,
  "used_doc": string (exact document content from the context that best supports your answer, or "None" if none clearly applies)
}}

Context:
{context}

Patient Symptom:
{input}
`
  });

  const agentExecutor = await initializeAgentExecutorWithOptions(
    [tool],
    llm,
    {
      agentType: "openai-functions",
      verbose: false,
    }
  );

  const promptText = await prompt.format({
    context,
    input: description,
  });

  const response = await agentExecutor.invoke({
    input: promptText,
  });

  const parsed = JSON.parse(response.output || "{}");
  return {
    ...parsed,
    top_context_used: parsed.used_doc || "No supporting document found"
  };
};

module.exports = {
  classifyWithHybridRAG,
  isHealthcareRelated,
};
