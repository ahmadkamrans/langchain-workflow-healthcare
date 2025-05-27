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

const retrieverPromise = FaissStore.load(
  path.join(__dirname, "../faiss_index"),
  embeddings
).then(store => store.asRetriever());

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
  const retriever = await retrieverPromise;
  const docs = await retriever.getRelevantDocuments(description);
  const context = docs.map(doc => doc.pageContent).join("\n---\n");

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

Respond strictly in this JSON format:

{{
  "urgency_level": "Emergency" | "Urgent Care" | "Non-Urgent" | "Follow-Up Needed" | "Unknown",
  "category": string (e.g., "Cardiac", "Infection", "Neurological", "Unknown", etc.),
  "internet_info_used": true | false
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
  return parsed;
};

module.exports = {
  classifyWithHybridRAG,
  isHealthcareRelated,
};
