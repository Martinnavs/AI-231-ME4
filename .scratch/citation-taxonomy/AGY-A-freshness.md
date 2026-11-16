## 1. Existing Taxonomies and Industry Practices

---

### A. FreshQA / FreshLLMs
* **Authors / Year:** Tu Vu, Mohit Iyyer, Xuezhi Wang, Noah Constant, Jerry Wei, Jason Wei, Chris Tar, Yun-Hsuan Sung, Denny Zhou, Quoc V. Le, Thang Luong (2023) \[[READ](https://arxiv.org/abs/2310.03214)\].
* **Exact Categories:**
  1. `"never-changing"`: Stable facts that almost never change over time \[[SNIPPET](https://arxiv.org/abs/2310.03214)\].
  2. `"slow-changing"`: Information that typically changes over several years \[[SNIPPET](https://arxiv.org/abs/2310.03214)\].
  3. `"fast-changing"`: Time-sensitive information that typically updates within a year or less \[[SNIPPET](https://arxiv.org/abs/2310.03214)\].
  4. `"false-premise"`: Questions built upon factually incorrect, misleading, or counterfactual assumptions that must be actively detected and debunked \[[SNIPPET](https://arxiv.org/abs/2310.03214)\].
* **Example Questions:**
  * `"never-changing"`: *"What breed of dog was Queen Elizabeth II of England famous for keeping?"* (Answer: Pembroke Welsh Corgi) \[[SNIPPET](https://arxiv.org/abs/2310.03214)\].
  * `"slow-changing"`: *"How many vehicle models does Tesla offer?"* \[[SNIPPET](https://arxiv.org/abs/2310.03214)\].
  * `"fast-changing"`: *"Which game won the Spiel des Jahres award most recently?"* \[[SNIPPET](https://arxiv.org/abs/2310.03214)\].
  * `"false-premise"`: *"What was the text of Donald Trump's first tweet in 2022, made after his unbanning from Twitter by Elon Musk?"* (Debunk: He did not tweet in 2022) \[[SNIPPET](https://arxiv.org/abs/2310.03214)\].
* **Requires Citation / Retrieval:** `"fast-changing"` strictly requires retrieval; models of all sizes fail without it \[[READ](https://arxiv.org/abs/2310.03214)\]. `"slow-changing"` requires retrieval if the event postdates model training or involves recent transitions \[[SNIPPET](https://arxiv.org/abs/2310.03214)\]. `"false-premise"` requires retrieval to verify whether the premise entity/event exists before generating an answer \[[SNIPPET](https://arxiv.org/abs/2310.03214)\]. `"never-changing"` does not strictly require retrieval if memorized during pretraining \[[SNIPPET](https://arxiv.org/abs/2310.03214)\].
* **How Detected:** In the paper, questions were categorized through human annotation \[[READ](https://arxiv.org/abs/2310.03214)\]. In downstream serving, systems use few-shot retrieval-augmented prompting (FreshPrompt) \[[READ](https://arxiv.org/abs/2310.03214)\].
* **URL:** [https://arxiv.org/abs/2310.03214](https://arxiv.org/abs/2310.03214) \[[READ](https://arxiv.org/abs/2310.03214)\].

---

### B. RealTime QA
* **Authors / Year:** Jungo Kasai, Keisuke Sakaguchi, Yoichi Takahashi, Ronan Le Bras, Akari Asai, Xinyan Yu, Dragomir Radev, Noah A. Smith, Yejin Choi, Kentaro Inui (2022) \[[READ](https://arxiv.org/abs/2207.13332)\].
* **Exact Categories:**
  * Topical news domains: `"politics"`, `"business"`, `"sports"`, `"entertainment"` \[[SNIPPET](https://arxiv.org/abs/2207.13332)\].
  * Evaluation task formats: `"multiple-choice"`, `"generation"`, `"NOTA (none of the above)"` \[[SNIPPET](https://arxiv.org/abs/2207.13332)\].
* **Definitions:** Dynamic benchmark where questions are announced weekly regarding current, unfolding world events \[[READ](https://arxiv.org/abs/2207.13332)\]. It tracks answer validity as a function of time elapsed ($\Delta t$) since publication \[[SNIPPET](https://arxiv.org/abs/2207.13332)\].
* **Example Questions:**
  * *"Which national park closed entrances due to unprecedented rainfall and flooding?"* (Answer: Yellowstone National Park) \[[SNIPPET](https://arxiv.org/abs/2207.13332)\].
  * *"Which wildly popular show was recently green lit for a new season?"* (Answer: Squid Game) \[[SNIPPET](https://arxiv.org/abs/2207.13332)\].
* **Requires Citation / Retrieval:** All questions require retrieval because answers emerge after model training cutoffs \[[READ](https://arxiv.org/abs/2207.13332)\]. When retrieval fails or retrieved passages lack evidence, the paper emphasizes models must output unanswerable/NOTA rather than guessing \[[READ](https://arxiv.org/abs/2207.13332)\].
* **How Detected:** Harvested automatically every week from human-written news quizzes (CNN, *The Week*, *USA Today*) \[[SNIPPET](https://arxiv.org/abs/2207.13332)\]. In deployed systems, detected via news feed event monitors or recency intent classifiers \[[SNIPPET](https://arxiv.org/abs/2207.13332)\].
* **URL:** [https://arxiv.org/abs/2207.13332](https://arxiv.org/abs/2207.13332) \[[READ](https://arxiv.org/abs/2207.13332)\].

---

### C. SituatedQA
* **Authors / Year:** Michael J. Q. Zhang, Eunsol Choi (2021) \[[READ](https://arxiv.org/abs/2109.06157)\].
* **Exact Categories:**
  1. `"Static"`: Questions whose answer does not depend on extra-linguistic context \[[SNIPPET](https://arxiv.org/abs/2109.06157)\].
  2. `"TEMP"` (Temporal context): Questions whose answer changes depending on when the question is asked \[[SNIPPET](https://arxiv.org/abs/2109.06157)\].
  3. `"GEO"` (Geographical context): Questions whose answer changes depending on where the user is situated \[[SNIPPET](https://arxiv.org/abs/2109.06157)\].
* **Definitions:** Classifies open-domain questions by whether extra-linguistic context creates an implicit dependency that alters the ground-truth answer \[[READ](https://arxiv.org/abs/2109.06157)\].
* **Example Questions:**
  * `"Static"`: *"Who composed the music for the first Harry Potter film?"* (John Williams) \[[SNIPPET](https://arxiv.org/abs/2109.06157)\].
  * `"TEMP"`: *"How many seasons are there for American Horror Story?"* or *"Who made the most three point shots in the NBA?"* \[[SNIPPET](https://arxiv.org/abs/2109.06157)\].
  * `"GEO"`: *"When was the last time states were created?"* (Nigeria: 1996 vs. United States: 1959) \[[SNIPPET](https://arxiv.org/abs/2109.06157)\].
* **Requires Citation / Retrieval:** `"TEMP"` and `"GEO"` require retrieval with explicit grounding on the user's timestamp and geographic location \[[READ](https://arxiv.org/abs/2109.06157)\].
* **How Detected:** Binary context-dependence classification over Natural Questions (NQ-Open) using crowd annotators and supervised encoder classifiers \[[READ](https://arxiv.org/abs/2109.06157), [SNIPPET](https://arxiv.org/abs/2109.06157)\].
* **URL:** [https://arxiv.org/abs/2109.06157](https://arxiv.org/abs/2109.06157) \[[READ](https://arxiv.org/abs/2109.06157)\].

---

### D. TimeQA & TempLAMA (Wikidata Temporal Relations)
* **Authors / Year:**
  * TimeQA: Wenhu Chen, Xinyi Wang, William Yang Wang (2021) \[[READ](https://arxiv.org/abs/2108.06314)\].
  * TempLAMA: Bhuwan Dhingra, Jeremy R. Cole, Julian Martin Eisenschlos, Daniel Gillick, Jacob Eisenstein, William W. Cohen (2022) \[[READ](https://arxiv.org/abs/2106.15110)\].
* **Exact Categories:**
  * TimeQA difficulty splits:
    1. `"TimeQA-Easy"`: Boundary questions where the time specifier explicitly matches the start or end timestamp of a time-evolving Wikidata relation \[[SNIPPET](https://arxiv.org/abs/2108.06314)\].
    2. `"TimeQA-Hard"`: Interior interval questions where the time specifier falls strictly inside the duration of a relation, requiring temporal containment reasoning \[[SNIPPET](https://arxiv.org/abs/2108.06314)\].
  * TempLAMA relations: Cloze-style queries over 9 time-varying Wikidata relations, most notably:
    * `"head of government"` \[[SNIPPET](https://arxiv.org/abs/2106.15110)\]
    * `"member of sports team"` \[[SNIPPET](https://arxiv.org/abs/2106.15110)\]
    * `"employer"` / `"position held"` \[[SNIPPET](https://arxiv.org/abs/2106.15110)\]
* **Example Questions:**
  * `"TimeQA-Easy"`: *"Who was the president of France in 2012?"* (Boundary transition year) \[[SNIPPET](https://arxiv.org/abs/2108.06314)\].
  * `"TimeQA-Hard"`: *"Who was the prime minister of Canada in 2008?"* (Stephen Harper, 2006–2015) \[[SNIPPET](https://arxiv.org/abs/2108.06314)\].
  * TempLAMA: *"The head of government of [country] is [MASK]"* contextualized for year $t$ \[[SNIPPET](https://arxiv.org/abs/2106.15110)\].
* **Requires Citation / Retrieval:** All time-evolving relations require retrieval or timestamp conditioning; models trained without temporal conditioning degrade rapidly on changing facts \[[READ](https://arxiv.org/abs/2106.15110)\].
* **How Detected:** Alignment of Wikidata time-stamped properties (`start_time`, `end_time`) to text passages \[[READ](https://arxiv.org/abs/2108.06314)\].
* **URLs:** [https://arxiv.org/abs/2108.06314](https://arxiv.org/abs/2108.06314) \[[READ](https://arxiv.org/abs/2108.06314)\], [https://arxiv.org/abs/2106.15110](https://arxiv.org/abs/2106.15110) \[[READ](https://arxiv.org/abs/2106.15110)\].

---

### E. Google Search: Query Deserves Freshness (QDF) & 2011 Freshness Update
* **Authors / Year:** Amit Singhal (2007 QDF patent/announcement; November 3, 2011 Google Official Blog) \[[SNIPPET](https://googleblog.blogspot.com/2011/11/giving-you-fresh-more-recent-search.html)\].
* **Exact Categories:**
  1. `"Recent events or hot topics"`: Breaking news, emergencies, or high-velocity topics where searchers require immediate updates \[[SNIPPET](https://googleblog.blogspot.com/2011/11/giving-you-fresh-more-recent-search.html)\].
  2. `"Regularly recurring events"`: Scheduled cyclical events (annual, seasonal, elections, award ceremonies) where users implicitly desire the current iteration \[[SNIPPET](https://googleblog.blogspot.com/2011/11/giving-you-fresh-more-recent-search.html)\].
  3. `"Frequent updates"` (or `"Frequently updated information"`): Facts that change periodically outside breaking news (e.g., product ratings, mortgage rates, specs) \[[SNIPPET](https://googleblog.blogspot.com/2011/11/giving-you-fresh-more-recent-search.html)\].
* **Example Questions:**
  * `"Recent events or hot topics"`: *"California earthquake magnitude"* or *"presidential debate live stream"* \[[SNIPPET](https://googleblog.blogspot.com/2011/11/giving-you-fresh-more-recent-search.html)\].
  * `"Regularly recurring events"`: *"NFL draft date"*, *"Tour de France stage results"*, *"tax filing deadline"* \[[SNIPPET](https://googleblog.blogspot.com/2011/11/giving-you-fresh-more-recent-search.html)\].
  * `"Frequent updates"`: *"best mirrorless camera reviews"*, *"Subaru Outback safety rating"*, *"average 30-year fixed mortgage rate"* \[[SNIPPET](https://googleblog.blogspot.com/2011/11/giving-you-fresh-more-recent-search.html)\].
* **Requires Citation / Retrieval:** All three categories mandate real-time web retrieval to suppress obsolete evergreen documents \[[SNIPPET](https://googleblog.blogspot.com/2011/11/giving-you-fresh-more-recent-search.html)\].
* **How Detected:** A mathematical model tracking: (1) query volume spike surges, (2) news velocity (rate of published articles in news feeds), (3) blog/social web chatter, and (4) calendar recurrence anchors \[[SNIPPET](https://googleblog.blogspot.com/2011/11/giving-you-fresh-more-recent-search.html)\]. Affects roughly 35% of all searches \[[SNIPPET](https://googleblog.blogspot.com/2011/11/giving-you-fresh-more-recent-search.html)\].
* **URL:** [https://googleblog.blogspot.com/2011/11/giving-you-fresh-more-recent-search.html](https://googleblog.blogspot.com/2011/11/giving-you-fresh-more-recent-search.html) \[[SNIPPET](https://googleblog.blogspot.com/2011/11/giving-you-fresh-more-recent-search.html)\].

---

### F. Self-RAG (Adaptive Reflection Classification)
* **Authors / Year:** Akari Asai, Sewon Min, Zexuan Zhong, Luke Zettlemoyer, Hannaneh Hajishirzi (2023) \[[READ](https://arxiv.org/abs/2310.11511)\].
* **Exact Categories (Reflection Tokens):**
  1. `"[Retrieve = Yes]"`: Query requires external factual grounding \[[SNIPPET](https://arxiv.org/abs/2310.11511)\].
  2. `"[Retrieve = No]"`: Query can be answered entirely from parametric memory (e.g., creative, definitional, or conversational) \[[SNIPPET](https://arxiv.org/abs/2310.11511)\].
  3. `"[Retrieve = Continue]"`: Generation proceeds using previously retrieved context \[[SNIPPET](https://arxiv.org/abs/2310.11511)\].
* **Example Questions:**
  * `"[Retrieve = Yes]"`: *"Who was appointed the new CEO of Starbucks recently?"* \[[SNIPPET](https://arxiv.org/abs/2310.11511)\].
  * `"[Retrieve = No]"`: *"Write a three-line poem about rain"* or *"How does photosynthesis work?"* \[[SNIPPET](https://arxiv.org/abs/2310.11511)\].
* **Requires Citation / Retrieval:** `"[Retrieve = Yes]"` strictly requires retrieval \[[READ](https://arxiv.org/abs/2310.11511)\].
* **How Detected:** Supervised token predictor generates the special token; retrieval triggers if $P(\text{Retrieve}=\text{Yes}) > \text{threshold}$ (default threshold $\tau = 0.2$; $\tau = 0.0$ for mandatory citation tasks) \[[READ](https://arxiv.org/abs/2310.11511), [SNIPPET](https://arxiv.org/abs/2310.11511)\].
* **URL:** [https://arxiv.org/abs/2310.11511](https://arxiv.org/abs/2310.11511) \[[READ](https://arxiv.org/abs/2310.11511)\].

---

### G. Gemini API Grounding: Dynamic Retrieval Configuration
* **Authors / Year:** Google DeepMind / Google Cloud (2024–2026) \[[READ](https://ai.google.dev/gemini-api/docs/grounding), [READ](https://cloud.google.com/vertex-ai/generative-ai/docs/grounding/overview)\].
* **Exact Categories / Modes:**
  * Predictor mode: `"MODE_DYNAMIC"` (dynamic routing) vs. `"MODE_UNSPECIFIED"` (always ground) \[[READ](https://cloud.google.com/vertex-ai/generative-ai/docs/grounding/overview)\].
  * Threshold field: `dynamic_threshold` (or `dynamicThreshold`, float between `0.0` and `1.0`; AI Studio default `0.3`, Vertex AI default `0.7`) \[[SNIPPET](https://cloud.google.com/vertex-ai/generative-ai/docs/grounding/overview)\].
  * Metric: `googleSearchDynamicRetrievalScore` \[[SNIPPET](https://ai.google.dev/gemini-api/docs/grounding)\].
* **Definition:** An internal classifier scores whether a prompt benefits from Google Search. If score $\ge \text{dynamic\_threshold}$, search is executed; otherwise, it relies on model weights \[[SNIPPET](https://cloud.google.com/vertex-ai/generative-ai/docs/grounding/overview)\].
* **Gemini 3.x Status:** Gemini 2.0, 2.5, and 3.x models support Google Search grounding natively via `tools=[{"google_search": {}}]` \[[READ](https://ai.google.dev/gemini-api/docs/grounding)\]. While `dynamic_retrieval_config` remains in the SDK types (`GoogleSearchRetrieval`) \[[SNIPPET](https://cloud.google.com/vertex-ai/generative-ai/docs/grounding/overview)\], Gemini 3.x primarily uses autonomous internal routing: the model evaluates tool intent directly and returns `groundingMetadata` containing `webSearchQueries`, `groundingChunks`, and `groundingSupports` \[[READ](https://ai.google.dev/gemini-api/docs/grounding)\].
* **URLs:** [https://ai.google.dev/gemini-api/docs/grounding](https://ai.google.dev/gemini-api/docs/grounding) \[[READ](https://ai.google.dev/gemini-api/docs/grounding)\], [https://cloud.google.com/vertex-ai/generative-ai/docs/grounding/overview](https://cloud.google.com/vertex-ai/generative-ai/docs/grounding/overview) \[[READ](https://cloud.google.com/vertex-ai/generative-ai/docs/grounding/overview)\].

---

### H. Academic Survey on Temporal QA
* **Authors / Year:** Bhawna Piryani, Abdelrahman Abdallah, Jamshid Mozafari, Avishek Anand, Adam Jatowt (ACL 2026 / arXiv:2505.20243) \[[READ](https://arxiv.org/abs/2505.20243)\].
* **Exact Dimensions / Taxonomy:**
  1. `"Corpus Temporality"`: `Diachronic` (time-stamped contemporary news streams) vs. `Synchronic` (retrospective static encyclopedia snapshots) \[[SNIPPET](https://arxiv.org/abs/2505.20243)\].
  2. `"Question Temporality"`: `Explicitly time-anchored` vs. `Implicit event-based` temporal intent \[[SNIPPET](https://arxiv.org/abs/2505.20243)\].
  3. `"Model Capabilities"`: Temporal language modeling, temporal RAG, and multi-hop interval reasoning \[[SNIPPET](https://arxiv.org/abs/2505.20243)\].
* **Requires Citation / Retrieval:** Diachronic streams require temporal RAG with time-stamped retrieval to resolve temporal drift \[[READ](https://arxiv.org/abs/2505.20243)\].
* **How Detected:** Query temporal profiling and time-expression normalizers (TimeML) \[[SNIPPET](https://arxiv.org/abs/2505.20243)\].
* **URL:** [https://arxiv.org/abs/2505.20243](https://arxiv.org/abs/2505.20243) \[[READ](https://arxiv.org/abs/2505.20243)\].

---

### I. Query Log Cost and Frequency Evidence
* **Implicit vs. Explicit Query Frequencies:**
  * Metzler et al. (2009) and query log analyses found explicit temporal expressions appear in **1.5% to 14%** of searches, whereas implicit temporal intent spans **7% to 17.1%** of queries \[[SNIPPET](https://greyc.fr)\]. Total time-sensitive traffic accounts for **10% to 25%** of commercial search volume \[[SNIPPET](https://emerald.com)\].
  * Google's 2011 Freshness Update documented that freshness signals alter results for **~35% of all searches** \[[SNIPPET](https://googleblog.blogspot.com/2011/11/giving-you-fresh-more-recent-search.html)\].
  * SituatedQA (Zhang & Choi 2021) demonstrated that **16.5%** of real user questions in NQ-Open have context-dependent answers (temporal or geographic) \[[READ](https://arxiv.org/abs/2109.06157)\].

---

## 2. Proposed Taxonomy for Our Assistant

To replace our simple keyword regex with a scientifically grounded, lightweight decision architecture suitable for spoken voice (1–3 short sentences, low latency), we synthesize FreshQA, SituatedQA, Google QDF, and TempLAMA into five operational categories:

| category | example | needs source? (always / if changeable / never) | staleness class | how to detect cheaply (regex or a tiny classifier) | evidence (which source above) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **High-Velocity Real-Time** | *"Who won the game tonight?"*, *"Current price of Nvidia"* | **Always** | Fast-changing (< 24 hours to 30 days) | Regex for deictics: `\b(today\|tonight\|yesterday\|now\|latest\|current\|live\|score\|price\|stock)\b` | FreshQA (`"fast-changing"` \[[READ](https://arxiv.org/abs/2310.03214)\]); Google QDF (`"Recent events"` \[[SNIPPET](https://googleblog.blogspot.com/2011/11/giving-you-fresh-more-recent-search.html)\]); RealTime QA \[[READ](https://arxiv.org/abs/2207.13332)\]. |
| **Cyclical & Superlative Records** | *"When is the next eclipse?"*, *"Who holds the NBA 3-point record?"*, *"Oscar winners"* | **Always** | Recurring / Slow-changing (months to 4 years) | Regex for cyclical tokens: `\b(next\|upcoming\|record\|most\|champion\|winner\|election\|season\|schedule)\b` | Google QDF (`"Regularly recurring events"` \[[SNIPPET](https://googleblog.blogspot.com/2011/11/giving-you-fresh-more-recent-search.html)\]); SituatedQA (`"TEMP"` \[[READ](https://arxiv.org/abs/2109.06157)\]). |
| **Institutional Roles & Policies** | *"Who is the mayor of Chicago?"*, *"What is the policy on EU travel?"* | **Always** | Slow-changing (1 to 5 years) | Regex for offices & relations: `\b(mayor\|governor\|ceo\|president\|minister\|director\|policy\|law\|head)\b` | TempLAMA (`"head of government"`, `"employer"` \[[READ](https://arxiv.org/abs/2106.15110)\]); FreshQA (`"slow-changing"` \[[READ](https://arxiv.org/abs/2310.03214)\]). |
| **False-Premise & Disputed Traps** | *"Why did Trump tweet in 2022 after being unbanned?"*, *"Is France's capital Lyon?"* | **If changeable** (verify premise prior to answering) | Trap / Epistemic check | Cheap local router check: if query mentions speculative assertions or unknown entity-action combinations, gate search verification | FreshQA (`"false-premise"` \[[READ](https://arxiv.org/abs/2310.03214)\]); RealTime QA (`"NOTA"` \[[READ](https://arxiv.org/abs/2207.13332)\]). |
| **Evergreen & Definitional Facts** | *"What is the speed of light?"*, *"Who wrote Hamlet?"*, *"What causes tides?"* | **Never** (safe for parametric weights) | Never-changing (static) | Negative pattern: absence of temporal markers + matches scientific/definitional intent (`\b(what is\|define\|formula\|who wrote)\b` without temporal modifiers) | FreshQA (`"never-changing"` \[[READ](https://arxiv.org/abs/2310.03214)\]); SituatedQA (`"Static"` \[[READ](https://arxiv.org/abs/2109.06157)\]). |

---

## 3. What We Should NOT Adopt and Why

1. **Do NOT adopt token-level reflection inference (Self-RAG `[Retrieve]` tokens)**:
   * *Why:* Self-RAG requires fine-tuning the LLM to output special tokens during generation and evaluating candidate beam paths \[[READ](https://arxiv.org/abs/2310.11511)\]. In a spoken voice pipeline using Gemini 3.1 Flash-Lite, token-level self-reflection increases time-to-first-token (TTFT) by several hundred milliseconds, violating conversational audio deadlines (< 500 ms). Routing must occur before or alongside the main generation call.
2. **Do NOT adopt high dynamic retrieval thresholds (e.g., Vertex AI default `dynamic_threshold = 0.7`)**:
   * *Why:* A threshold of 0.7 suppresses retrieval on "slow-changing" queries (e.g., municipal officials or sports rosters) where the LLM's internal confidence is deceptively high \[[SNIPPET](https://cloud.google.com/vertex-ai/generative-ai/docs/grounding/overview)\]. The literature proves LLMs exhibit high confidence on outdated facts; a high threshold results in silent hallucinations. For factual grounding, threshold must be set low ($\le 0.2$) or gated by the routing taxonomy.
3. **Do NOT adopt complex multi-hop temporal interval solvers (TimeQA-Hard reasoning graphs)**:
   * *Why:* TimeQA's complex interval constraint solvers \[[READ](https://arxiv.org/abs/2108.06314)\] are designed for multi-document document analysis. In a voice assistant answering in 1–3 short sentences, parsing temporal interval graphs introduces unnecessary latency. Real-time web search results (Google Search grounding / Tavily) already incorporate time-indexed ranking.
4. **Do NOT adopt unconstrained LLM autonomy without premise checking**:
   * *Why:* As proven in FreshQA \[[READ](https://arxiv.org/abs/2310.03214)\], models unconditionally accept false premises in 60%+ of cases when answering directly. We must instruct the model to explicitly verify premise validity and speak a hedge or denial if unsupported, rather than assuming user questions describe real events.
5. **Do NOT rely exclusively on explicit keyword regexes without entity-relation awareness**:
   * *Why:* Metzler et al. and SituatedQA prove that **7% to 17%** of user queries are *implicitly* temporal (e.g., *"How many seasons of American Horror Story?"*) without containing words like `"current"`, `"new"`, or `"CEO"` \[[READ](https://arxiv.org/abs/2109.06157), [SNIPPET](https://greyc.fr)\]. Relying only on words like `"latest"` misses roughly half of all time-dependent queries. The router must check for evolving relation predicates (e.g., titles, quantities, records).
