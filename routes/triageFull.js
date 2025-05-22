// routes/triageFull.js
const express = require("express");
const { traceable } = require("langsmith/traceable");
const router = express.Router();
const { classifyWithRAG } = require("../langchain/ragClassifier");
// Wrap handler so LangSmith traces the entire HTTP request
router.post(
  "/",
  traceable(async (req, res) => {
    const { description } = req.body;
    if (!description || typeof description !== "string" || !description.trim()) {
      return res.status(400).json({ error: "Invalid symptom description." });
    }
    try {
      const result = await classifyWithRAG(description);
      if (!result.urgency_level || !result.category) {
        return res.status(500).json({ error: "Model response was incomplete." });
      }
      res.json({ success: true, urgency_level: result.urgency_level, category: result.category });
    } catch (err) {
      console.error("RAG classification error:", err);
      res.status(500).json({ error: "Failed to classify symptom." });
    }
  })
);
module.exports = router;