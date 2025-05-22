// langchain/ragClassifier.js
const path = require("path");
const { FaissStore } = require("@langchain/community/vectorstores/faiss");
const { OpenAIEmbeddings, ChatOpenAI } = require("@langchain/openai");
const { RunnableSequence } = require("@langchain/core/runnables");
const { PromptTemplate } = require("@langchain/core/prompts");
const { StringOutputParser } = require("@langchain/core/output_parsers");
// Initialize shared instances once
const embeddings = new OpenAIEmbeddings({
  openAIApiKey: process.env.OPENAI_API_KEY,
  modelName: "text-embedding-3-small",
});
// Lazy load FAISS retriever once
const retrieverPromise = FaissStore.load(
  path.join(__dirname, "../faiss_index"),
  embeddings
).then(store => store.asRetriever());
// Initialize LLM once
const llm = new ChatOpenAI({
  openAIApiKey: process.env.OPENAI_API_KEY,
  modelName: "gpt-3.5-turbo-0125",
  temperature: 0,
});
// Prompt template
const prompt = PromptTemplate.fromTemplate(`
You are a medical triage assistant. Based on the symptom, extract:
- urgency_level (Emergency, Urgent Care, Non-Urgent, Follow-Up Needed)
- category (e.g., Cardiac, Flu, Mental Health, Allergy, etc.)
Context from knowledge base:
{context}
Symptom: {input}
Respond in JSON format with keys "urgency_level" and "category".
`);
// Build the RAG pipeline
// const ragChainPromise = retrieverPromise.then(retriever =>
//   RunnableSequence.from([
//     async input => {
//       const docs = await retriever.getRelevantDocuments(input);
//       return { context: docs.map(d => d.pageContent).join("\n\n"), input };
//     },
//     prompt,
//     llm,
//     new StringOutputParser(),
//   ])
// );

const ragChainPromise = retrieverPromise.then(retriever =>
  RunnableSequence.from([
    async input => {
      const docs = await retriever.getRelevantDocuments(input);
      return {
        context: docs.map(d => d.pageContent).join("\n\n"),
        input: String(input) // Safe and clean for LangSmith tracing
      };
    },
    prompt,
    llm,
    new StringOutputParser(),
  ])
);

// Final function
async function classifyWithRAG(description) {
  if (!description || typeof description !== "string" || !description.trim()) {
    throw new Error("Invalid description input.");
  }
  try {
    const ragChain = await ragChainPromise;
    const result = await ragChain.invoke(description.trim());
    const parsed = JSON.parse(result);
    return { urgency_level: parsed.urgency_level, category: parsed.category };
  } catch (err) {
    console.error("RAG Error:", err);
    throw new Error("Classification failed: " + err.message);
  }
}
module.exports = { classifyWithRAG };