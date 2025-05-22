const express = require("express");
const router = express.Router();
const { classifyWithRAG } = require("../langchain/ragClassifier");

router.post("/", async (req, res) => {
  const { description } = req.body;

  if (!description || typeof description !== "string" || description.trim().length === 0) {
    return res.status(400).json({ error: "Invalid symptom description." });
  }

  try {
    const result = await classifyWithRAG(description);

    if (!result?.urgency_level || !result?.category) {
      return res.status(500).json({ error: "Model response was incomplete." });
    }

    res.json({
      success: true,
      urgency_level: result.urgency_level,
      category: result.category,
    });
  } catch (err) {
    console.error("RAG classification error:", err);
    res.status(500).json({ error: "Failed to classify symptom." });
  }
});

// Make sure this line exists and exports the router
module.exports = router;