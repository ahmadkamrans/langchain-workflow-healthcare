const fs = require("fs");
const path = require("path");
const { OpenAIEmbeddings } = require("@langchain/openai");
const { Document } = require("langchain/document");
const { FaissStore } = require("@langchain/community/vectorstores/faiss"); // ✅ VALID


require("dotenv").config();

(async () => {
  const filePath = path.join(__dirname, "../data/triage_knowledge.txt");
  const content = fs.readFileSync(filePath, "utf-8");

  const documents = content.split("\n\n").map((entry, index) => {
    return new Document({
      pageContent: entry.trim(),
      metadata: { source: `entry_${index}` },
    });
  });

  const embeddings = new OpenAIEmbeddings({
    openAIApiKey: process.env.OPENAI_API_KEY,
    modelName: "text-embedding-3-small",
  });

  const vectorStore = await FaissStore.fromDocuments(documents, embeddings);
  await vectorStore.save(path.join(__dirname, "../faiss_index"));

  console.log("✅ FAISS index saved.");
})();
