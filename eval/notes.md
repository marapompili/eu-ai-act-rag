# Error notes (working log)

Format: question | what the system answered | my judgement | hypothesis on the cause.
Model: llama3.2:3b, temperature 0, retrieval k=4, chunks of max 250 words.

## v0 - first prompt (free citation format)

1. "Which AI practices are prohibited?"
   - Retrieved: Art. 5. Answer: only "(a), (b), ... (h)", no content.
   - Judgement: poor
   - Hypothesis: the model had the text but copied the point letters (small model).

2. "What fines apply for non-compliance?"
   - Retrieved: Art. 99, 100. Amounts of both articles mixed, all attributed to "Article 99".
   - Judgement: partly wrong. I checked the text: the 1,500,000 / 750,000 euro amounts are in Art. 100.
   - Hypothesis: two similar articles in the same context; the model merged them.

3. "When must people be told they are talking to an AI?"
   - Retrieved: Art. 4, 21, 50. Answer paraphrases Art. 50(1).
   - Judgement: --
   - Hypothesis: retrieval ranked Art. 4 (AI literacy) first, likely because of shared words.

4. "What is the capital of France?"
   - Retrieved: Art. 49, 67, 110, 113. Answer: "I cannot find this in the provided text."
   - Judgement: correct refusal.
   - Note: retrieval always returns k chunks, even for an off-topic question.

## v1 - prompt asking for [Art. N] format

- Answers reduced to citations only ("[Art. 50(1)]") or "points (a) to (h)".
- My citation checker flagged them as uncited: false positive, the regex did not accept "[Art. 99(4)]".
- Lesson: I changed prompt and checker together, so I could not tell which caused what.

## v2 results (prompt asking for content, regex accepting "[Art. N(p)]")

1. Prohibited practices: 8 points with content, cited [Art. 5]. Much better than v0/v1. Summaries are shortened. To check against Art. 5(1)(d), (h).
2. Fines: amounts plausible, but the 1,500,000 / 750,000 euro amounts (Art. 100) are attributed to Art. 99. The checker said unsupported=[] because Art. 100 was in the context. Limit of the checker: it checks that the cited article was provided, not that the claim comes from it. Citations also listed at the end instead of after each sentence.
3. Telling people they interact with an AI: close to Art. 50(1), but the sentence says the AI systems "are informed" (subject garbled). Retrieval still ranks Art. 4 and 21 in the top 3.
4. Capital of France: correct refusal.

Timing: 20-45 s per question (single runs, noisy machine).

## Open ideas (not tested yet)

- Score threshold on retrieval to say "I don't know" when the best score is low.
- Query instruction prefix for the BGE model.
- Compare chunk sizes (150 / 250 / 400 words).