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
## Retrieval evaluation, dev split (10 answerable questions)

- q07 (fines for prohibited practices): gold Art. 99 not in top 10; Art. 5 took 5 of the 10 slots.
  Hypothesis: the words "prohibited AI practice" match the title of Art. 5. Also several chunks of
  the same article crowd out other articles.
- q05 (log retention for "users"): kept gold = [26]. Checked the text: Art. 19 addresses providers,
  Art. 26(6) addresses deployers. The system ranked Art. 19 first: a real error (wrong actor),
  not a labelling problem. The word "users" is not a term of the regulation.
- Query prefix (BGE instruction): recommended by the model authors for retrieval. Decision rule
  fixed before running: keep it if hit@4 does not drop and MRR does not get worse.

## Experiment 1: BGE query prefix (dev split, 10 answerable questions)

- Without prefix: hit@1 6/10, hit@3 8/10, hit@4 9/10, hit@10 9/10, MRR 0.708.
- With prefix:    hit@1 5/10, hit@3 9/10, hit@4 9/10, hit@10 10/10, MRR 0.678.
- Changes: q02 4->3, q03 1->2, q07 not found->9; the other 7 questions unchanged.
- Decision (rule fixed beforehand: keep if hit@4 does not drop and MRR does not get worse):
  MRR got worse, so the default stays WITHOUT prefix. The difference is one question moving
  by one position: not distinguishable from noise with 10 questions.
- Plan: report both configurations on the test split, once, at the end.

## Experiment 2: chunk size (dev split, 10 answerable questions, no query prefix)

| max words | chunks | hit@1 | hit@4 | hit@10 | MRR |
|---|---|---|---|---|---|
| 100 | 578 | 7/10 | 9/10 | 10/10 | 0.787 |
| 150 | 398 | 7/10 | 8/10 | 10/10 | 0.765 |
| 250 | 245 | 6/10 | 9/10 | 9/10 | 0.708 |

- 400 words was not tried: at 250 words the longest chunk was already 415 tokens, and the
  embedding model's limit is 512, so longer chunks would be silently truncated.
- Decision rule fixed beforehand: change size only if hit@4 gains at least 2 of 10 questions
  and MRR does not drop. It is not met (9, 8, 9), so 250 words stays.
- MRR rises as chunks get smaller, but the confidence intervals overlap almost completely and
  three configurations were compared on the same 10 questions: not distinguishable from noise.
- Not measured here: smaller chunks give the language model less text in its top 4
  (about 400 words at 100 words/chunk vs about 1000 at 250). That needs the answer evaluation.
