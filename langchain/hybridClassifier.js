const path = require("path");
require("dotenv").config();
const { FaissStore } = require("@langchain/community/vectorstores/faiss");
const { OpenAIEmbeddings, ChatOpenAI } = require("@langchain/openai");
const { AIMessage, HumanMessage, SystemMessage } = require("@langchain/core/messages");
const { RunnableSequence } = require("@langchain/core/runnables");
const { traceable } = require("langsmith/traceable");
const { z } = require("zod");

// Output schema
const outputSchema = z.object({
  urgency_level: z.enum(["Emergency", "Urgent Care", "Non-Urgent", "Follow-Up Needed", "Unknown"]),
  category: z.string(),
  context_used: z.enum(["internal", "internet", "both", "none"]),
  used_doc: z.string(),
  recommendation: z.string(),
});
let faissStorePromise;
let initialized = false;

// Initialize vector store
(async () => {
  const embeddings = new OpenAIEmbeddings({
    openAIApiKey: process.env.OPENAI_API_KEY,
    modelName: "text-embedding-3-small",
  });
  faissStorePromise = FaissStore.load(
    path.join(__dirname, "../faiss_index"),
    embeddings
  );
  initialized = true;
})();

// Validator function
const isHealthcareRelated = traceable(
  async (description) => {
    const llm = new ChatOpenAI({
      openAIApiKey: process.env.OPENAI_API_KEY,
      modelName: "gpt-4o",
      temperature: 0,
    });
    const result = await llm.invoke([
      new SystemMessage(`You are a strict healthcare input validator.
Only return true if the user clearly describes symptoms (e.g., chest pain, sore throat, fatigue).
If vague or unrelated to health, return false.
Respond with "true" or "false".`),
      new HumanMessage(description),
    ]);
    return result.content.trim().toLowerCase() === "true";
  },
  {
    name: "isHealthcareRelated",
    projectName: process.env.LANGCHAIN_PROJECT || "RAG_Healthcare",
  }
);

// Robust JSON extraction utility that works for array OR string
function extractJSONFromContent(content) {
  // If content is an array (as with OpenAI v4/LLM), join or select the text parts
  if (Array.isArray(content)) {
    const texts = content
      .filter(x => x && typeof x.text === "string")
      .map(x => x.text)
      .join("\n");
    return extractJSONFromContent(texts); // Recursively process as string
  }
  if (!content || typeof content !== "string") return null;
  // Remove markdown code block if present
  const codeBlockMatch = content.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
  if (codeBlockMatch) content = codeBlockMatch[1].trim();
  // Find first JSON object in content
  const jsonMatch = content.match(/\{[\s\S]*\}/);
  if (jsonMatch) {
    try {
      return JSON.parse(jsonMatch[0]);
    } catch (e) {
      // Try to fix common issues here if needed
      return null;
    }
  }
  return null;
}

const classifyWithIntelligentRAG = traceable(
  async (description) => {
    if (!initialized) throw new Error("Hybrid classifier not initialized yet.");
    const faissStore = await faissStorePromise;

    // Internal document retrieval (with scores)
    const internalResults = await faissStore.similaritySearchWithScore(description, 5);
    const scoredDocs = internalResults
      .map(([doc, score]) => ({ content: doc.pageContent, score }))
      .sort((a, b) => a.score - b.score);

    // Build a string with similarity scores, so the model knows how relevant they are.
    const internalContext = scoredDocs.map(
      (d, i) => `Doc #${i+1} (score: ${d.score.toFixed(3)}):\n${d.content}`
    ).join("\n---\n");

    const llmWithTools = new ChatOpenAI({
      modelName: "gpt-4o",
      openAIApiKey: process.env.OPENAI_API_KEY,
      temperature: 0,
      useResponsesApi: true,
      tools: [{ type: "web_search_preview" }],
    });

    const chain = RunnableSequence.from([
      async () => [
        new SystemMessage(
`You are a highly cautious and knowledgeable medical triage assistant.
You are provided with:
- A patient symptom description.
- Internal medical documents most similar to the symptoms, with a similarity score (lower is more relevant).
- Access to web search (via the "web_search_preview" tool).

Instructions:
- If any internal document is highly relevant (score < 0.15), and it answers the question, prefer internal knowledge.
- If internal docs are missing, ambiguous, or not relevant, or if you need up-to-date information, use web search.
- If you use both, set "context_used" to "both".
- Otherwise, choose the single most authoritative source ("internal" or "internet").

Fill out and return this JSON (in a markdown code block if replying as text):

{
  "urgency_level": "Emergency" | "Urgent Care" | "Non-Urgent" | "Follow-Up Needed" | "Unknown",
  "category": string,
  "context_used": "internal" | "internet" | "both" | "none",
  "used_doc": string,
  "recommendation": string
}
Always state in "used_doc" which document or web page you relied on the most.`
        ),
        new HumanMessage(
`Symptom Description:
${description}

Internal Medical Documents (with similarity scores):
${internalContext}

If unsure, use web search to validate or supplement your answer.`
        ),
      ],
      llmWithTools,
    ]);

    // Run the chain and extract result
    let parsed;
    try {
      const result = await chain.invoke();
      console.dir(result, { depth: null }); // For debugging

      let extracted = null;
      // Prefer tool call output if present
      if (result.additional_kwargs?.tool_calls?.length) {
        const toolArgs = result.additional_kwargs.tool_calls[0].function.arguments;
        try {
          extracted = typeof toolArgs === "string" ? JSON.parse(toolArgs) : toolArgs;
        } catch {}
      }
      // Otherwise, look for JSON in the content (including markdown code blocks, arrays)
      if (!extracted) {
        extracted = extractJSONFromContent(result.content);
      }
      if (!extracted) throw new Error("No structured content found");
      const validated = outputSchema.safeParse(extracted);
      if (!validated.success) throw new Error("Model response didn't match schema");
      parsed = validated.data;
    } catch (e) {
      console.error("Classification failed:", e.message);
      parsed = {
        urgency_level: "Unknown",
        category: "Unknown",
        context_used: "none",
        used_doc: "Parsing error or unexpected response",
        recommendation: "Unable to classify. Please consult a healthcare provider.",
      };
    }
    return {
      ...parsed,
      similarity_scores: scoredDocs,
      internet_info_used: parsed.context_used?.includes("internet") ?? false,
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