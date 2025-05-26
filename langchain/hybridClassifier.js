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

  const prompt = PromptTemplate.fromTemplate(`
You are a medical triage assistant.
Use the following internal documentation and internet search if needed.
Always prefer internal docs if sufficient.

Context:
{context}

Patient Symptom:
{input}

Respond in JSON format with:
- urgency_level: (Emergency, Urgent Care, Non-Urgent, Follow-Up Needed)
- category: (e.g., Cardiac, Flu, Mental Health, Allergy, etc.)
- internet_info_used: true or false
`);

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
