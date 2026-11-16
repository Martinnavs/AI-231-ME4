# Research Report: Grounded Citation and Retrieval Taxonomies for Spoken Voice Assistants

---

## 1. Existing Taxonomies and Benchmarks in Literature

### 1. FreshQA / FreshLLMs
* **Authors & Year:** Tu Vu, Mohit Iyyer, Xuezhi Wang, et al. (2023) [[READ](https://arxiv.org/abs/2310.03214)].
* **Categories (exact names):**
  1. `fast-changing`
  2. `slow-changing`
  3. `never-changing`
  4. `false-premise`
* **Definitions & Examples:**
  * `fast-changing`: Facts that change frequently—often multiple times a year, monthly, or weekly (roles, current records, stock prices, sports scores, recent news/awards) [[SNIPPET](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHGqd-da7IrsS29uVjfa-Or9OZbdcAIWvE4Bl26uGG-I1MPipZ1AKryiMUba_qy1Da4Uhdk8NgfBKnmIC1nGGBRuGJzSpjSSNHdLOaV_hKjP7XwzVDI)]. *Example:* "What is Apple's current stock price?" or "Who won the men's 100m final in the latest Olympics?"
  * `slow-changing`: Facts that evolve over a multi-year or decadal horizon (demographic statistics, infrastructure, long-standing records, leadership of long-tenured bodies) [[SNIPPET](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHGqd-da7IrsS29uVjfa-Or9OZbdcAIWvE4Bl26uGG-I1MPipZ1AKryiMUba_qy1Da4Uhdk8NgfBKnmIC1nGGBRuGJzSpjSSNHdLOaV_hKjP7XwzVDI)]. *Example:* "What is the population of Tokyo?"
  * `never-changing`: Permanent historical facts, scientific constants, geography, or mathematical truths [[SNIPPET](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQESiy88D1VkSgJBMgYCt0Fd4rmzZ8geMr43BCY6MwYR8SiSnDw_v6oGgINeP2EGTk4y55Qp5y4YkdqKDmpd8-hbmYDjtIKgljLzvAAhzPkGC0auH99m7Cy5arZUfbauTEna90QTJ1EK1WpHKI0BptNXgTaGgf25FKByyYCTq0vEB9zLQg==)]. *Example:* "What is the atomic number of gold?" or "In what year did the Titanic sink?"
  * `false-premise`: Adversarial queries or trick questions built on a factually incorrect assumption or nonexistent event that must be debunked rather than answered directly [[SNIPPET](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQESiy88D1VkSgJBMgYCt0Fd4rmzZ8geMr43BCY6MwYR8SiSnDw_v6oGgINeP2EGTk4y55Qp5y4YkdqKDmpd8-hbmYDjtIKgljLzvAAhzPkGC0auH99m7Cy5arZUfbauTEna90QTJ1EK1WpHKI0BptNXgTaGgf25FKByyYCTq0vEB9zLQg==)]. *Example:* "Why did Shakespeare use a typewriter to write Hamlet?"
* **Citation / Retrieval Need:** `fast-changing` and `slow-changing` require retrieval and citation; `never-changing` relies on parametric memory; `false-premise` requires retrieval/fact-checking to refute the flawed premise.
* **Detection:** Annotated benchmark metadata; deployed in *FreshPrompt* via few-shot retrieval prompting [[READ](https://arxiv.org/abs/2310.03214)].
* **URL:** [https://arxiv.org/abs/2310.03214](https://arxiv.org/abs/2310.03214) [[READ](https://arxiv.org/abs/2310.03214)].

---

### 2. Adaptive-RAG
* **Authors & Year:** Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, Jong C. Park (2024 / NAACL 2024) [[READ](https://arxiv.org/abs/2403.14403)].
* **Categories (exact names):**
  1. `Non-Retrieval` (or `No-Retrieval`)
  2. `Single-Step Retrieval`
  3. `Multi-Step Retrieval`
* **Definitions & Examples:**
  * `Non-Retrieval`: Simple factual queries answerable directly using the LLM's internal parametric memory without external knowledge base lookups [[SNIPPET](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG0XnAIrG5JmWCl8fa2tldnna8Id10Ia1O4v81bVR90xZIE18DzvNahk1Ps8BcWaOa0ZsfyaLuu0uzNC3sDGJdVuNtAEihFecUQIrDyORVk3jgqrMCOQJLHiThTqFUcQidgL1kOBkIaJlVBhL6EI8h6LvgdP5pFVHc=)]. *Example:* "What is the capital of France?"
  * `Single-Step Retrieval`: Moderately complex factual questions requiring an external reference resolvable in a single search pass [[SNIPPET](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG0XnAIrG5JmWCl8fa2tldnna8Id10Ia1O4v81bVR90xZIE18DzvNahk1Ps8BcWaOa0ZsfyaLuu0uzNC3sDGJdVuNtAEihFecUQIrDyORVk3jgqrMCOQJLHiThTqFUcQidgL1kOBkIaJlVBhL6EI8h6LvgdP5pFVHc=)]. *Example:* "Who was appointed CEO of Microsoft in 2014?"
  * `Multi-Step Retrieval`: Complex, compositional, or multi-hop queries requiring iterative retrieval and intermediate reasoning [[READ](https://arxiv.org/abs/2403.14403)]. *Example:* "Which film directed by Christopher Nolan won the most Academy Awards, and who composed its score?"
* **Citation / Retrieval Need:** `Non-Retrieval` never retrieves; `Single-Step Retrieval` always retrieves and cites a source; `Multi-Step Retrieval` always executes iterative retrieval and cites multiple corroborating sources.
* **Detection:** A small language model query complexity classifier (specifically fine-tuned `T5-large` or smaller) trained on outcome-based labels [[SNIPPET](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHct9gCMyO_E9C9H3soyA1vHV578yydkZj46Z4QhYa-t2Spv1HmYCUnRjbwLB9IUoPhUIfaCc996oGbEdGNDHkYxVz9voThaK_OwIQ0BOB4AH1OuNbdSutTGyfrPJTy)].
* **URL:** [https://arxiv.org/abs/2403.14403](https://arxiv.org/abs/2403.14403) [[READ](https://arxiv.org/abs/2403.14403)].

---

### 3. SituatedQA
* **Authors & Year:** Michael J.Q. Zhang, Eunsol Choi (2021 / EMNLP 2021) [[READ](https://arxiv.org/abs/2109.06157)].
* **Categories (exact names):**
  1. `Temporally dependent (TEMP)`
  2. `Geographically dependent (GEO)`
  3. `Context-independent` (Static)
* **Definitions & Examples:**
  * `Temporally dependent (TEMP)`: Questions whose correct answer changes across time [[READ](https://arxiv.org/abs/2109.06157)]. *Example:* "Who is the Prime Minister of Australia?" or "Where will the next Olympics be held?" [[SNIPPET](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQELC5XX5nT0jfSh7ZsgOPjWqNtbK1I-NE_kJLF_R8_Cw4afZNOqWf_6Sg0rvqOnfMLh7gVcftjCS74ecNlWlhoIkisKqa5wZ_kRdghs9jgGjhd4JHY45Mle6CjGsrCIFsQ2HAfrTg==)]
  * `Geographically dependent (GEO)`: Questions whose answer varies based on location or jurisdiction [[READ](https://arxiv.org/abs/2109.06157)]. *Example:* "What is the emergency phone number?" or "What is the legal driving age?"
  * `Context-independent`: Questions whose answers remain invariant regardless of when or where they are asked [[READ](https://arxiv.org/abs/2109.06157)]. *Example:* "What is the boiling point of water at sea level?"
* **Citation / Retrieval Need:** `TEMP` and `GEO` always require grounding and dated/localized citations; `Context-independent` does not.
* **Detection:** Extra-linguistic slot dependency analysis and entity parsing [[READ](https://arxiv.org/abs/2109.06157)].
* **URL:** [https://arxiv.org/abs/2109.06157](https://arxiv.org/abs/2109.06157) [[READ](https://arxiv.org/abs/2109.06157)].

---

### 4. Self-RAG (Self-Reflective Retrieval-Augmented Generation)
* **Authors & Year:** Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi (2023 / ICLR 2024) [[READ](https://arxiv.org/abs/2310.11511)].
* **Categories (Reflection Tokens):**
  1. `[Retrieve]` with values: `{Retrieve, No-Retrieve, Continue}`
  2. `[IsREL]` (Relevance): `{Relevant, Irrelevant}`
  3. `[IsSUP]` (Supported): `{Fully supported, Partially supported, No support}`
  4. `[IsUSE]` (Utility): `{5, 4, 3, 2, 1}`
* **Definitions & Examples:**
  * `Retrieve`: Emitted when the LM determines that external knowledge is necessary to generate a factual, verifiable segment [[READ](https://arxiv.org/abs/2310.11511)]. *Example:* "What were the key terms of the 2023 SAG-AFTRA strike resolution?"
  * `No-Retrieve`: Emitted when the query can be addressed using internal weights or common reasoning [[READ](https://arxiv.org/abs/2310.11511)]. *Example:* "Summarize how photosynthesis works."
* **Citation / Retrieval Need:** When `[Retrieve] == Retrieve`, external documents are fetched, and generated claims are verified against `[IsSUP]` tokens; when `No-Retrieve`, search is bypassed.
* **Detection:** Self-generated reflection tokens during generation, with an inference-time probability threshold: $P([\text{Retrieve}] = \text{Yes}) \ge \tau$ [[SNIPPET](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGeXZw7qcWh32oFjl7QWSJRKvg1g-caGGXE5Ec3sj5WzQr_qacnL0_2-4c7TxLhbSEHEGwGwTn2IZYOqMfT-mOlVwuqXsrNgmzNyoXe5sQYTNUO3Hsvq0lvs3nb1S2biVz934MbY1i2e8giLwAIODwJ_V154uCP3LoRgbYvWjHjCiZAYEFtqoNDsplVOd7OYiD9SQ0-XGumXP4wloHGbJOqAwLx)].
* **URL:** [https://arxiv.org/abs/2310.11511](https://arxiv.org/abs/2310.11511) [[READ](https://arxiv.org/abs/2310.11511)].

---

## 2. Deployed Systems & Industry Practices

### 1. Google Gemini: Grounding with Google Search & Dynamic Retrieval
* **Source & Architecture:** Google DeepMind / Google Cloud Platform (2024) [[READ](https://ai.google.dev/gemini-api/docs/grounding), [SNIPPET](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHOuvnYwKQsGYX757SBM9WMoFKMWHPxGfV0btJTYRDXGYzJxHopxUUxfAxpb3rmtZ7hNvdv8-TQFz28eEelk-jnFqcuVpUMR2mG28fOeWtQkaI-1iS0pJvAc73ypePWKn_fD-7vxd4H2yEdmuHNo9wzWhBUNUHtfBFKaVE_WF8coSRwpV9HOymMprQguNOQv-WewBTn1dlrQQ068J_80t7CJv7wUQuI9g==)].
* **Decision Mechanism:** Rather than static regex, Gemini employs **Dynamic Retrieval** (`DynamicRetrievalConfig`, `mode="MODE_DYNAMIC"`), using a specialized classifier that assigns a continuous **prediction score** $s \in [0.0, 1.0]$ [[SNIPPET](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEUFecsTC4m_dsJKfn8AEyRBq0oreGSTvLek-hPiA7W8n_EkC3XrAzGdS-NDlPk_Hw1-w0rsMYTzXHxtBPn1Ss5oNP1DaF0-0YuSQWqA0CgQcwTR-XzeXLnxXTQDXSq5aAoSbWeUQfwlhO6dxN-17sGatW2CItPmteXB311EhUI6SPiBA8njYdvQEIDNdPC_iGIUIKvBSIkZoiYGw==)].
* **Thresholding:** Configured via `dynamic_threshold` (or `predictor_threshold`, default ~`0.3`) [[SNIPPET](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEUFecsTC4m_dsJKfn8AEyRBq0oreGSTvLek-hPiA7W8n_EkC3XrAzGdS-NDlPk_Hw1-w0rsMYTzXHxtBPn1Ss5oNP1DaF0-0YuSQWqA0CgQcwTR-XzeXLnxXTQDXSq5aAoSbWeUQfwlhO6dxN-17sGatW2CItPmteXB311EhUI6SPiBA8njYdvQEIDNdPC_iGIUIKvBSIkZoiYGw==), [SNIPPET](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGZUoeBzYisqb4IUif0BrwEP-q5SEfPnPiwvT0j1goi6k42jk24XPaxtwo_MV3DeM5ldJk-U99L5Z_t1edl-_YKRP2SaS4GvcapxufzEW0hMbCt6lJ0RU-7PflKV-vCXmtB53vA1rEWPD9R6WaQCq9QzssvCsF0zJ6dEOFfvvuWtc8poSuydUvQ5BBEzU4jEiyBB4yqKa-WUnXdytdlXTEUx5wngK5bjL5s)]. If $s \ge \text{threshold}$, Google Search is triggered; otherwise, parametric weights answer the prompt [[SNIPPET](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHQipoww7XkNuk1vm9PCC8vaBbzfc6EnAUixT3B-RgEPXVNbqrKxTs0dOou0FuMfGnrEKENziIgWlokCyzrceYBA49yh_ReiwceymYZTv1DE900vPxBH-qZ2Ef9esK-S07QkutjrApWb4xYLKop_wwfdoFPipNlXpOwHS4CdXUMa4wLP-ZWEMU6MA2lY0s=)].
* **Categories Triggering High Scores:** Queries post-dating model cutoff, live sports, breaking news, market movements, weather, local geographic facts [[SNIPPET](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFN-jgB8brZHUXBsP6DSbl3hfew0dhONwAmJrOjuurPfGZOxQzy9A8sRmspCd0BPRh1OE8kX403q-no1pltKRoBT8AVI-ZX07ghsYOmV7PpbYTfNGRPeAuO6O2fT1NqKtHWBcGqqYLFXQRBfFVsp1rp1yl6xG9iDA==)].
* **Attribution Output:** Emits structured `groundingMetadata` containing `groundingChunks` (source URLs, site titles) and `webSearchQueries` [[READ](https://ai.google.dev/gemini-api/docs/grounding)].
* **URL:** [https://ai.google.dev/gemini-api/docs/grounding](https://ai.google.dev/gemini-api/docs/grounding) [[READ](https://ai.google.dev/gemini-api/docs/grounding)].

---

### 2. OpenAI ChatGPT Search & Claude Web Search Directives
* **Source & Guidelines:** OpenAI Documentation & Anthropic System Directives [[READ](https://platform.openai.com), [READ](https://github.com/asgeirtj/system_prompts_leaks), [SNIPPET](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG0fUuTNZqmTZL1vow2c5bx9p8dwAn4319Iwx1Q-MXJLLt5khUBiyK9wxw4Dhowl8cNmT3kjIWa5ljYcRDRJbEaxRu37aA3TtER15q3wogOMSQFWj_xVchAwN7CCUqNEa4XFkJ66FPaZPU=)].
* **Dedicated Trigger Categories:** OpenAI documents five dedicated real-time search categories with visual layouts [[SNIPPET](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG0fUuTNZqmTZL1vow2c5bx9p8dwAn4319Iwx1Q-MXJLLt5khUBiyK9wxw4Dhowl8cNmT3kjIWa5ljYcRDRJbEaxRu37aA3TtER15q3wogOMSQFWj_xVchAwN7CCUqNEa4XFkJ66FPaZPU=)]:
  1. `weather`
  2. `stocks`
  3. `sports`
  4. `news`
  5. `maps` (local/geographic)
* **System Prompt Directives (Leaked / Documented Standards):**
  * Models *must* call `web_search` when questions involve temporal anchors (*"latest," "today," "recent," "2026," "current"*), live prices, officeholders, or post-cutoff entity changes [[SNIPPET](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEI1LSwCxIusJ3C9kJpSzxu9uLRQ_3B_fkU89qighu9tzumpWQrbCuxfaio19Njrmh1gjQGJwRkQQ6w1uP8hz5PpVQJkLlIfPOdAJNXJE5oPALb2XvOO75q5RwnBEL2AGETPpOWHNCGhMmkJkyAvZt_hgjKwmsFYhQ-), [READ](https://github.com/asgeirtj/system_prompts_leaks)].
  * Models are instructed **never** to search for core mathematics, general coding logic, established history, creative writing, or user-supplied conversational contexts [[SNIPPET](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFMqfvJCdmrREWvpRQJdmW_c3j7UvYZUfE18U4rq4dGuLIabQeiil6oEJZIIbEB35rWcZxKT53udX5OS6GkbEZQYkxf3isXA_1L_BVzM9wLcqptQEsAgLVbzQjCdfrDUjt5l5_ES5V1sEq_Secw4FdUBC2HJsiMa1npSMmeEp6OPiC2)].
* **Citation Rule:** Strict enforcement that any sentence or claim derived from a search result *must* carry an inline citation link or token referencing the source [[READ](https://github.com/asgeirtj/system_prompts_leaks)].
* **URLs:** [https://platform.openai.com](https://platform.openai.com) [[READ](https://platform.openai.com)], [https://github.com/asgeirtj/system_prompts_leaks](https://github.com/asgeirtj/system_prompts_leaks) [[READ](https://github.com/asgeirtj/system_prompts_leaks)].

---

### 3. Voice-Specific Spoken Attribution & Uncertainty (Alexa, Google Assistant, ISCA)
* **Sources:** Amazon Alexa Guidelines, Google Conversational UX, Interspeech Research [[READ](https://www.isca-archive.org/interspeech_2017/szekely17_interspeech.html), [SNIPPET](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF-QKt-6CyuW-zVnbSRJmDZK_vI08DzvspME4qlk8JvAjGljTh2SR4-wMCOYKyziCZCsEllzHqgbw_3R4TLSOEL488ioNf8vlMGe7c15_1gapkKvhKo2m5HBS3wMcErJVFNwy-7tKoXuyWnJgQh1HOCQ3XDLRSHYS8V5Es=), [SNIPPET](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGYUaOK3PTOJs9kRT0mrVMROm-as_i97_2Z8lB-5P359Pvqgh5LST_XQ32097gwsxuxU8UIKWI0CfMlzWlmpjtPmrxwoA-yLtUJ5NyyqNol3v335q4JTr7AiIySteGOPOpBe0F9UK5UjnvfTrlttVa9ZpafnU15WkAJx8NquH0A2TW9garBVK4=)].
* **Spoken Attribution Wording & Phrasing:**
  * **Front-Loading Rule:** Attribution must appear at the beginning or middle of the utterance (e.g., *"According to Reuters, ..."*, *"From Wikipedia, ..."*) rather than appended at the end [[SNIPPET](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF-QKt-6CyuW-zVnbSRJmDZK_vI08DzvspME4qlk8JvAjGljTh2SR4-wMCOYKyziCZCsEllzHqgbw_3R4TLSOEL488ioNf8vlMGe7c15_1gapkKvhKo2m5HBS3wMcErJVFNwy-7tKoXuyWnJgQh1HOCQ3XDLRSHYS8V5Es=)]. In audio, listeners lose track of ungrounded assertions before a trailing citation arrives.
  * **Clean Entity Names:** Omit domain noise (`www`, `.com`, `.org`) [[SNIPPET](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF-QKt-6CyuW-zVnbSRJmDZK_vI08DzvspME4qlk8JvAjGljTh2SR4-wMCOYKyziCZCsEllzHqgbw_3R4TLSOEL488ioNf8vlMGe7c15_1gapkKvhKo2m5HBS3wMcErJVFNwy-7tKoXuyWnJgQh1HOCQ3XDLRSHYS8V5Es=)]. Say *"According to Healthline..."* rather than *"According to healthline dot com..."*. Keep attribution to 2–4 spoken words.
  * **Single Primary Source Rule:** Spoken responses (1–3 sentences) must name only one primary source [[SNIPPET](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF-QKt-6CyuW-zVnbSRJmDZK_vI08DzvspME4qlk8JvAjGljTh2SR4-wMCOYKyziCZCsEllzHqgbw_3R4TLSOEL488ioNf8vlMGe7c15_1gapkKvhKo2m5HBS3wMcErJVFNwy-7tKoXuyWnJgQh1HOCQ3XDLRSHYS8V5Es=)]. Citing multiple sources or nested chains verbally causes severe cognitive overload. User studies highlight widespread listener irritation with verbose attribution disclaimers [[SNIPPET](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHL5R_JT3vdwqJGVMY8takOlGNKdFpX0OEwH-O0_SflCcSbbK79AJppzGGBU1kZXI1JF7Jx3EsKO6jJWQGlx6qI894sUB8j5qybICzEa-avPHRB-DvFIvFf7phmDknlq515gDeTKWIFpqMvpJyUGQ2GIEjGoMcjZbdfNZQBjzjFjtIoRAm207xBcSt1Zkl4pdTmPYKnMw==)].
  * **Acoustic Boundary:** Insert a 200–300ms pause (via SSML `<break time="250ms"/>`) between the attribution lead-in and the factual clause [[SNIPPET](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF-QKt-6CyuW-zVnbSRJmDZK_vI08DzvspME4qlk8JvAjGljTh2SR4-wMCOYKyziCZCsEllzHqgbw_3R4TLSOEL488ioNf8vlMGe7c15_1gapkKvhKo2m5HBS3wMcErJVFNwy-7tKoXuyWnJgQh1HOCQ3XDLRSHYS8V5Es=)].
* **Guidance on Saying Uncertainty Aloud (Verbal Hedging):**
  * When retrieval is uncorroborated, conflicting, or confidence is low, the assistant must use calibrated verbal hedges: *"I found one report from Reuters, but...", "I couldn't find a recent confirmation, but...", "Reports vary, but..."* [[SNIPPET](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGX_CyG0lEP9Z9jKDAlWEUXoMnGiBKsOMfiMsTeKFRCCsNQ7PW93WGm47aSbMmTsf5H5J-lIwOype6zsiSgCVp0QOjlnAYKtWOfRJiu9dBJvDjeRdNN8HaUbg==), [SNIPPET](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHDLwQCetZTYq8Tm9lTxI4jBILmrf6BBgZPwY3FKCiAMQH2dGc3hOY1FS_W8ppZNP3KBNrL-tQMVe8ojQ4MhkzSSluU6PyYF4DMukzYBqirvySRZEu8PY9zwTmREMbBjZEdDeUox1WcU42A-RZ0dMws9iXkR24a-lxtngbaFoyMYA==)].
  * Studies in speech synthesis (Interspeech 2017) show that hesitation disfluencies, decreased vocal effort, and verbal hedging explicitly convey epistemic doubt [[READ](https://www.isca-archive.org/interspeech_2017/szekely17_interspeech.html)]. However, excessive hedging introduces an automated "hedging penalty" where the assistant sounds incompetent [[SNIPPET](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFfXfZCEXZ8SIkTMXq_F1NpPgeghdFTi29o3nqb3mUg6F09ATq_jkcXg5UOBtAAcIhf7dB5l79_OH9suNp-nkNMLR-dRDcgvP4mrG95t-7P9NtGHuRNfUBl1mugeIQ-Cs605w==)]. Calibrated hedging paired with source attribution establishes optimal user trust.
* **URL:** [https://www.isca-archive.org/interspeech_2017/szekely17_interspeech.html](https://www.isca-archive.org/interspeech_2017/szekely17_interspeech.html) [[READ](https://www.isca-archive.org/interspeech_2017/szekely17_interspeech.html)].

---

## 3. Proposed Taxonomy for Our Assistant

The proposed taxonomy synthesizes `FreshQA` temporal dynamics, `SituatedQA` context dependencies, and OpenAI/Gemini vertical categories, tailored for a 1–3 sentence spoken voice pipeline:

| Category | Example Question | Needs Source? | Staleness Class | How to Detect Cheaply | Evidence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Real-Time Vertical** | "What is Nvidia's stock price right now?" | Always | Fast-changing (Minutes to Days) | Regex on commercial/weather terms (`stock`, `price`, `market`, `weather`, `score`, `game`, `forecast`) | ChatGPT Search verticals [[SNIPPET](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG0fUuTNZqmTZL1vow2c5bx9p8dwAn4319Iwx1Q-MXJLLt5khUBiyK9wxw4Dhowl8cNmT3kjIWa5ljYcRDRJbEaxRu37aA3TtER15q3wogOMSQFWj_xVchAwN7CCUqNEa4XFkJ66FPaZPU=)], FreshQA `fast-changing` [[SNIPPET](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHGqd-da7IrsS29uVjfa-Or9OZbdcAIWvE4Bl26uGG-I1MPipZ1AKryiMUba_qy1Da4Uhdk8NgfBKnmIC1nGGBRuGJzSpjSSNHdLOaV_hKjP7XwzVDI)] |
| **Dynamic Role & Public Office** | "Who is the Prime Minister of the UK?" | Always | Slow-changing (Months to Years) | Regex on roles (`CEO`, `president`, `mayor`, `prime minister`, `secretary of`, `coach`) + absence of historical date anchor | FreshQA `slow-changing` [[SNIPPET](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHGqd-da7IrsS29uVjfa-Or9OZbdcAIWvE4Bl26uGG-I1MPipZ1AKryiMUba_qy1Da4Uhdk8NgfBKnmIC1nGGBRuGJzSpjSSNHdLOaV_hKjP7XwzVDI)], SituatedQA `TEMP` [[READ](https://arxiv.org/abs/2109.06157)], OnIt prompt |
| **Temporal Qualifier** | "What was the latest update on the border treaty?" | Always | Fast-to-Medium | Regex on temporal markers (`latest`, `current`, `new`, `recent`, `this week`, `upcoming`, `next`) | ChatGPT search directives [[SNIPPET](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEI1LSwCxIusJ3C9kJpSzxu9uLRQ_3B_fkU89qighu9tzumpWQrbCuxfaio19Njrmh1gjQGJwRkQQ6w1uP8hz5PpVQJkLlIfPOdAJNXJE5oPALb2XvOO75q5RwnBEL2AGETPpOWHNCGhMmkJkyAvZt_hgjKwmsFYhQ-)], Gemini Dynamic Retrieval [[SNIPPET](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFN-jgB8brZHUXBsP6DSbl3hfew0dhONwAmJrOjuurPfGZOxQzy9A8sRmspCd0BPRh1OE8kX403q-no1pltKRoBT8AVI-ZX07ghsYOmV7PpbYTfNGRPeAuO6O2fT1NqKtHWBcGqqYLFXQRBfFVsp1rp1yl6xG9iDA==)] |
| **Geographic / Jurisdiction** | "What is the local sales tax rate in Austin?" | Always | Slow-to-Medium | Regex on location words (`nearby`, `here`, `in [City]`, `sales tax`, `driving age`, `emergency number`) | SituatedQA `GEO` [[READ](https://arxiv.org/abs/2109.06157)], ChatGPT `maps` [[SNIPPET](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG0fUuTNZqmTZL1vow2c5bx9p8dwAn4319Iwx1Q-MXJLLt5khUBiyK9wxw4Dhowl8cNmT3kjIWa5ljYcRDRJbEaxRu37aA3TtER15q3wogOMSQFWj_xVchAwN7CCUqNEa4XFkJ66FPaZPU=)] |
| **Multi-Hop / Compositional** | "Which Nolan film won the most Oscars and who scored it?" | Always | Multi-hop / Variable | Conjunction patterns (`and who`, `which ... that also`, `both ... and`) or tiny router classifier | Adaptive-RAG `Multi-Step Retrieval` [[READ](https://arxiv.org/abs/2403.14403)] |
| **Presupposition Trap** | "Why did Shakespeare use a typewriter?" | Always (to debunk) | Adversarial Verification | Check query for unverified entity-predicate assertions; or if Gemini confidence is low, trigger search to verify | FreshQA `false-premise` [[SNIPPET](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQESiy88D1VkSgJBMgYCt0Fd4rmzZ8geMr43BCY6MwYR8SiSnDw_v6oGgINeP2EGTk4y55Qp5y4YkdqKDmpd8-hbmYDjtIKgljLzvAAhzPkGC0auH99m7Cy5arZUfbauTEna90QTJ1EK1WpHKI0BptNXgTaGgf25FKByyYCTq0vEB9zLQg==)] |
| **Static Parametric Fact** | "What is the speed of light in a vacuum?" | Never | Permanent (Never-changing) | Default when no temporal, role, vertical, or location regex triggers; or regex on historical years (`in 1787`, `boiling point`, `atomic number`) | FreshQA `never-changing` [[SNIPPET](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQESiy88D1VkSgJBMgYCt0Fd4rmzZ8geMr43BCY6MwYR8SiSnDw_v6oGgINeP2EGTk4y55Qp5y4YkdqKDmpd8-hbmYDjtIKgljLzvAAhzPkGC0auH99m7Cy5arZUfbauTEna90QTJ1EK1WpHKI0BptNXgTaGgf25FKByyYCTq0vEB9zLQg==)], Adaptive-RAG `Non-Retrieval` [[SNIPPET](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG0XnAIrG5JmWCl8fa2tldnna8Id10Ia1O4v81bVR90xZIE18DzvNahk1Ps8BcWaOa0ZsfyaLuu0uzNC3sDGJdVuNtAEihFecUQIrDyORVk3jgqrMCOQJLHiThTqFUcQidgL1kOBkIaJlVBhL6EI8h6LvgdP5pFVHc=)] |

---

## 4. What We Should NOT Adopt and Why

1. **Do NOT adopt iterative reflection token loops (Self-RAG / FLARE).**  
   *Why:* Self-RAG and FLARE require generating tokens, evaluating confidence thresholds, retrieving documents, and regenerating sentence by sentence [[READ](https://arxiv.org/abs/2310.11511)]. In a spoken voice assistant where First Time to Audio (TTFA) must stay under 800–1200ms, multi-pass token critique introduces 2–4 seconds of unacceptable conversational latency.

2. **Do NOT adopt text-based bibliography formatting (OnIt's References section / ALCE numerical citation brackets `[1][2]`).**  
   *Why:* Bracketed citations like `[1]` or a dedicated "References:" block are designed for reading, not speaking [[READ](https://arxiv.org/abs/2305.14627)]. A TTS engine reading bracketed numbers or listing URLs aloud creates an unintelligible voice experience.

3. **Do NOT adopt trailing citations (*"...according to Reuters"* at the end of the sentence).**  
   *Why:* Speech and VUI guidelines (Amazon Alexa, Google Assistant) establish that trailing citations violate human cognitive limits in audio [[SNIPPET](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF-QKt-6CyuW-zVnbSRJmDZK_vI08DzvspME4qlk8JvAjGljTh2SR4-wMCOYKyziCZCsEllzHqgbw_3R4TLSOEL488ioNf8vlMGe7c15_1gapkKvhKo2m5HBS3wMcErJVFNwy-7tKoXuyWnJgQh1HOCQ3XDLRSHYS8V5Es=)]. Spoken citations must be front-loaded (*"According to Reuters, [claim]"*), anchoring authority before the listener absorbs the claim.

4. **Do NOT adopt raw domain name reading (*"reuters dot com"* or *"en dot wikipedia dot org"*).**  
   *Why:* VUI best practices require stripping web cruft to a concise brand entity (*"Reuters"*, *"Wikipedia"*). Reading domain extensions clutters the limited 1–3 sentence spoken budget.

5. **Do NOT adopt multi-source verbal recitation (*"Source A and Source B both confirm..."*).**  
   *Why:* In voice interfaces, the "Rule of One Fact" applies [[SNIPPET](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF-QKt-6CyuW-zVnbSRJmDZK_vI08DzvspME4qlk8JvAjGljTh2SR4-wMCOYKyziCZCsEllzHqgbw_3R4TLSOEL488ioNf8vlMGe7c15_1gapkKvhKo2m5HBS3wMcErJVFNwy-7tKoXuyWnJgQh1HOCQ3XDLRSHYS8V5Es=)]. While our backend must corroborate with two sources (as required by our owner), the spoken response should cite only the primary source (*"According to Reuters, ... "*) to avoid listener fatigue [[SNIPPET](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHL5R_JT3vdwqJGVMY8takOlGNKdFpX0OEwH-O0_SflCcSbbK79AJppzGGBU1kZXI1JF7Jx3EsKO6jJWQGlx6qI894sUB8j5qybICzEa-avPHRB-DvFIvFf7phmDknlq515gDeTKWIFpqMvpJyUGQ2GIEjGoMcjZbdfNZQBjzjFjtIoRAm207xBcSt1Zkl4pdTmPYKnMw==)].

6. **Do NOT adopt a single monolithic regex as the sole gating mechanism.**  
   *Why:* A single regex triggers severe false positives on historical and static questions (e.g., *"Who was the first president of the United States?"* matches `president` but is permanently static; *"Who is the president of Chile?"* is a slow-changing role). The gating logic must combine role tokens with the absence of historical year anchors or leverage Gemini's native `dynamic_threshold` (~0.3) [[SNIPPET](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEUFecsTC4m_dsJKfn8AEyRBq0oreGSTvLek-hPiA7W8n_EkC3XrAzGdS-NDlPk_Hw1-w0rsMYTzXHxtBPn1Ss5oNP1DaF0-0YuSQWqA0CgQcwTR-XzeXLnxXTQDXSq5aAoSbWeUQfwlhO6dxN-17sGatW2CItPmteXB311EhUI6SPiBA8njYdvQEIDNdPC_iGIUIKvBSIkZoiYGw==)].

7. **Do NOT adopt pervasive or uncalibrated verbal hedging.**  
   *Why:* Studies on conversational agents demonstrate that excessive hedging (*"I think," "It might be," "Perhaps"*) incurs a severe "hedging penalty," causing users to judge the assistant as incompetent [[SNIPPET](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFfXfZCEXZ8SIkTMXq_F1NpPgeghdFTi29o3nqb3mUg6F09ATq_jkcXg5UOBtAAcIhf7dB5l79_OH9suNp-nkNMLR-dRDcgvP4mrG95t-7P9NtGHuRNfUBl1mugeIQ-Cs605w==)]. Hedging must be strictly conditional—triggered only when retrieval yields a single uncorroborated source or when confidence falls below the acceptance threshold.
