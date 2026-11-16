# Literature Survey: Retrieval, Citation, and Check-Worthiness Taxonomies

This survey reviews published taxonomies and industry frameworks that govern when an NLP system must retrieve external sources, verify statements, and provide citations. Each framework is detailed below with exact category names, definitions, examples, detection mechanisms, and source URLs. Evidence sources are tagged as **[READ]** (verified from papers/repos read on permitted domains) or **[SNIPPET]** (verified from search index summaries).

---

## 1. Existing Taxonomies & Detection Practices

### 1. Attributable to Identified Sources (AIS)
* **Authors/Year:** Hannah Rashkin, Vitaly Nikolaev, Matthew Lamm, Lora Aroyo, Michael Collins, Dipanjan Das, Slav Petrov, Gaurav Singh Tomar, Izhak Shafran (2021 / 2023) **[READ]**.
* **Categories:**
  1. `"Uninterpretable"`: Generated text that is incomprehensible, syntactically broken, ambiguous in reference, or non-declarative dialogue lacking an external factual assertion **[SNIPPET]**.
  2. `"Interpretable but Not Attributable"` (or `"Not AIS"`): A well-formed factual proposition about the external world that cannot be verified by or directly deduced from the identified source **[SNIPPET]**.
  3. `"Attributable to Identified Sources"` (or `"AIS"`): A well-formed statement where every assertion about the external world is completely supported and entailed by the identified source text **[READ]**.
* **Example Questions / Statements:**
  * `"Uninterpretable"`: *"Well, um, because then it went there."* **[SNIPPET]**
  * `"Interpretable but Not Attributable"`: *"The Eiffel Tower was built completely out of titanium in 1950."* (against an 1889 wrought-iron source) **[SNIPPET]**.
  * `"AIS"`: *"The Eiffel Tower was completed in 1889 for the Exposition Universelle."* **[SNIPPET]**.
* **Which Require Retrieval / Citation:** All `"Interpretable"` statements asserting facts about the external world require source grounding (**AIS**) **[READ]**. `"Uninterpretable"` statements are discarded or sent back for clarification **[SNIPPET]**.
* **Detection Mechanism:** Two-stage annotation pipeline or fine-tuned natural language inference (NLI) classifier evaluating source-to-statement entailment: $P(\text{Entailment} \mid \text{Source}, \text{Statement}) \ge \tau$ **[READ]** **[SNIPPET]**.
* **URL:** https://arxiv.org/abs/2112.12870 **[READ]**

---

### 2. ALCE Benchmark (Automatic LLMs' Citation Evaluation)
* **Authors/Year:** Tianyu Gao, Howard Yen, Jiatong Yu, Danqi Chen (2023) **[READ]**.
* **Categories:** Sentence-level attribution classes under NLI entailment:
  1. `"Fully supported"`: The sentence is completely entailed by the set of cited passages ($\phi(\text{Passage}, s_i) = 1$) **[SNIPPET]**.
  2. `"Partially supported"`: Part of the sentence is entailed, but it contains extrapolated or unsupported assertions **[SNIPPET]**.
  3. `"Unsupported"` / `"No support"`: The cited passage contradicts or provides no relevant evidence for the sentence ($\phi(\text{Passage}, s_i) = 0$) **[SNIPPET]**.
  * *Citation behavior dimensions:* `"Citation Recall"` (proportion of generated sentences supported by their citations) and `"Citation Precision"` (proportion of cited passages that are genuinely relevant and necessary to the claims) **[READ]** **[SNIPPET]**.
* **Example Statements:**
  * `"Fully supported"`: *"Alexander Fleming discovered penicillin in 1928."* citing a medical history text documenting the 1928 event **[SNIPPET]**.
  * `"Partially supported"`: *"Fleming discovered penicillin in 1928 and won the Nobel Prize in 1935."* citing a source that only mentions the 1928 discovery **[SNIPPET]**.
  * `"Unsupported"`: *"Penicillin treats viral colds."* citing an antibiotic mechanism paper **[SNIPPET]**.
* **Which Require Citation:** All informative factual sentences in an answer must achieve `"Fully supported"` status (`Citation Recall = 1`) **[READ]**.
* **Detection Mechanism:** Automated sentence segmentation coupled with a fine-tuned NLI model (TRUE / T5-XXL / RoBERTa-large) applying an entailment probability cutoff against cited passages **[SNIPPET]**.
* **URL:** https://arxiv.org/abs/2305.14627 **[READ]**

---

### 3. Self-RAG (Self-Reflective Retrieval-Augmented Generation)
* **Authors/Year:** Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi (2023 / 2024) **[READ]**.
* **Categories (Reflection Tokens):**
  1. Retrieval decision (`Retrieve`): `"[Retrieve]"`, `"[No Retrieve]"`, `"[Continue Retrieval]"` **[READ]** **[SNIPPET]**.
  2. Passage relevance (`ISREL`): `"[Relevant]"`, `"[Irrelevant]"` **[SNIPPET]**.
  3. Factual grounding (`ISSUP`): `"[Fully Supported]"`, `"[Partially Supported]"`, `"[No Support / Contradictory]"` **[SNIPPET]**.
  4. Overall utility (`ISUSE`): `"[Utility: 1]"` through `"[Utility: 5]"` **[SNIPPET]**.
* **Example Questions / Decisions:**
  * `"[No Retrieve]"`: *"What is the definition of photosynthesis?"* (static parametric common knowledge) **[SNIPPET]**.
  * `"[Retrieve]"`: *"Who won the men's 100m final in the latest Olympic Games?"* (dynamic/mutable world knowledge) **[SNIPPET]**.
  * `"[Continue Retrieval]"`: Multi-step reasoning queries requiring iterative chaining **[SNIPPET]**.
* **Which Require Retrieval / Citation:** Segments emitting `"[Retrieve]"` or `"[Continue Retrieval]"` trigger external search and must achieve `"[Fully Supported]"` status **[READ]** **[SNIPPET]**.
* **Detection Mechanism:** The generator LM is trained via supervised fine-tuning and reinforcement learning to output reflection tokens inline; during inference, retrieval triggers when the probability $P(\text{"[Retrieve]"} \mid x)$ crosses a threshold $\tau$ **[READ]** **[SNIPPET]**.
* **URL:** https://arxiv.org/abs/2310.11511 **[READ]**

---

### 4. Adaptive-RAG & FLARE
* **Authors/Year:**
  * **Adaptive-RAG:** Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, Jong C. Park (2024) **[READ]**.
  * **FLARE:** Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, Graham Neubig (2023) **[READ]**.
* **Categories:**
  * *Adaptive-RAG Query Complexity Tiers:*
    1. `"Non-Retrieval"`: Queries whose factual answers reside reliably within the model's parametric weights **[SNIPPET]**.
    2. `"Single-Step Retrieval"`: Single-hop queries requiring an external factual lookup **[SNIPPET]**.
    3. `"Multi-Step Retrieval"`: Complex multi-hop queries requiring question decomposition and iterative searches **[SNIPPET]**.
  * *FLARE Confidence States:*
    1. `"High-confidence tokens"`: All generation token probabilities $\ge \theta$; generation proceeds without search **[READ]** **[SNIPPET]**.
    2. `"Low-confidence tokens"`: Any forward-looking token probability falls below threshold $\theta$; triggers active search query synthesis and sentence regeneration **[READ]** **[SNIPPET]**.
* **Example Questions:**
  * `"Non-Retrieval"` / `"High-confidence"`: *"What is the chemical formula for water?"* **[SNIPPET]**.
  * `"Single-Step Retrieval"` / `"Low-confidence"`: *"What was the closing price of Tesla stock yesterday?"* **[SNIPPET]**.
  * `"Multi-Step Retrieval"`: *"Which film directed by Christopher Nolan won the most Academy Awards?"* **[SNIPPET]**.
* **Which Require Retrieval:** `"Single-Step Retrieval"`, `"Multi-Step Retrieval"`, and sentences exhibiting `"Low-confidence tokens"` **[READ]** **[SNIPPET]**.
* **Detection Mechanism:** Adaptive-RAG routes queries using a small trained classifier (e.g. T5-small) **[READ]**; FLARE monitors logit confidence scores of drafted tokens during decoding **[READ]**.
* **URLs:** https://arxiv.org/abs/2403.14403 **[READ]**; https://arxiv.org/abs/2305.06983 **[READ]**

---

### 5. Popularity-Based Adaptive Retrieval ("When Not to Trust LMs")
* **Authors/Year:** Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, Hannaneh Hajishirzi (2023) **[READ]**.
* **Categories:**
  1. `"Head"`: High-popularity entities ($>10^5$ monthly Wikipedia pageviews) where parametric memory recall is high and unassisted LMs are competitive **[READ]** **[SNIPPET]**.
  2. `"Torso"`: Medium-popularity entities ($10^2$ to $10^5$ pageviews) where parametric accuracy drops and hallucination rates increase **[SNIPPET]**.
  3. `"Tail"` (or `"Long-tail"`): Low-popularity entities ($<10^2$ pageviews) where parametric memory fails catastrophically and model parameter scaling yields negligible improvements **[READ]** **[SNIPPET]**.
* **Example Questions:**
  * `"Head"`: *"Who wrote Hamlet?"* (William Shakespeare) **[SNIPPET]**.
  * `"Torso"`: *"What is the capital of Suriname?"* (Paramaribo) **[SNIPPET]**.
  * `"Tail"`: *"Who directed the 1994 indie film 'Clean, Shaven'?"* (Lodge Kerrigan) **[SNIPPET]**.
* **Which Require Retrieval / Citation:** `"Torso"` and `"Tail"` entities strictly mandate non-parametric retrieval **[READ]**.
* **Detection Mechanism:** Offline entity popularity lookup or online Wikipedia API pageview / link-degree threshold ($\text{Pageviews}(E) < \tau$) **[SNIPPET]**.
* **URL:** https://arxiv.org/abs/2212.10511 **[READ]**

---

### 6. Atomic Claim Verification: FActScore & SAFE
* **Authors/Year:**
  * **FActScore:** Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Wei Koh, Mohit Iyyer, Luke Zettlemoyer, Hannaneh Hajishirzi (2023) **[READ]**.
  * **SAFE:** Jerry Wei, Chengrun Yang, Xinying Song, Yifeng Lu, Nathan Hu, Dustin Tran, Daiyi Peng, Ruibo Liu, Da Huang, Cosmo Du, Quoc V. Le (2024) **[READ]**.
* **Categories:**
  1. `"Atomic facts"`: Decontextualized single-predicate factual assertions extracted from long-form text **[READ]** **[SNIPPET]**.
  2. `"Irrelevant"`: Atomic propositions that fail to address the original prompt context (omitted from evaluation) **[SNIPPET]**.
  3. `"Supported"`: Atomic facts corroborated and verified by the knowledge base or Google Search results **[READ]** **[SNIPPET]**.
  4. `"Not Supported"` (or `"Unsupported"`): Atomic facts contradicted by or unprovable from retrieved evidence **[READ]** **[SNIPPET]**.
* **Example Claims:**
  * `"Irrelevant"`: *"She was a fascinating woman."* (subjective assessment) **[SNIPPET]**.
  * `"Supported"`: *"Ada Lovelace was the daughter of Lord Byron."* (verified by biographical sources) **[SNIPPET]**.
  * `"Not Supported"`: *"Ada Lovelace served as Prime Minister in 1845."* (factually false) **[SNIPPET]**.
* **Which Require Retrieval / Citation:** All relevant `"Atomic facts"` require search verification; factuality score is the proportion of supported relevant facts **[READ]** **[SNIPPET]**.
* **Detection Mechanism:** Prompted LLM decomposition into independent atomic claims, followed by automated search query generation and an LLM/NLI verifier **[READ]** **[SNIPPET]**.
* **URLs:** https://arxiv.org/abs/2305.14251 **[READ]**; https://arxiv.org/abs/2403.18802 **[READ]**

---

### 7. Claim Check-Worthiness: ClaimBuster & CLEF CheckThat!
* **Authors/Year:**
  * **ClaimBuster:** Naeemul Hassan, Fatma Arslan, Chengkai Li, Mark Tremayne (2017 / 2020) **[SNIPPET]**.
  * **CLEF CheckThat! Lab:** Preslav Nakov, Alberto Barrón-Cedeño, Tamer Elsayed, Shaden Shaar, Maram Hasanain, et al. (2018–2024) **[READ]** **[SNIPPET]**.
* **Categories:**
  * *ClaimBuster:*
    1. `"Non-Factual Sentence"` (NFS): Subjective opinions, personal beliefs, greetings, value judgments, or rhetorical questions **[SNIPPET]**.
    2. `"Unimportant Factual Sentence"` (UFS): Objective statements that are trivial, common knowledge, or lack public significance **[SNIPPET]**.
    3. `"Check-worthy Factual Sentence"` (CFS): Verifiable factual assertions containing quantifiable metrics, policy claims, or public consequences **[SNIPPET]**.
  * *CLEF CheckThat! Task 1:*
    1. `"Check-worthy factual claim"` (CW): Objective factual claims of broad public interest that are likely to be disputed or potentially harmful if false **[READ]** **[SNIPPET]**.
    2. `"Non-check-worthy"` (NCW): Subjective banter, personal anecdotes, or established general knowledge **[SNIPPET]**.
* **Example Statements:**
  * `"NFS"` / `"NCW"`: *"We need more courage and integrity in Washington."* **[SNIPPET]**.
  * `"UFS"`: *"I had toast and black coffee for breakfast."* or *"Tokyo is in Japan."* **[SNIPPET]**.
  * `"CFS"` / `"CW"`: *"Our state's unemployment rate dropped by 3.2% last quarter."* **[SNIPPET]**.
* **Which Require Retrieval / Citation:** `"Check-worthy Factual Sentence"` (CFS) and `"Check-worthy factual claim"` (CW) strictly require retrieval and citation **[SNIPPET]**.
* **Detection Mechanism:** Fine-tuned transformer classifiers (RoBERTa / BERT / DeBERTa) or lightweight linear models estimating check-worthiness probability: $P(\text{CFS}) \ge \tau$ **[READ]** **[SNIPPET]**.
* **URLs:** https://github.com/utaresearch/claimbuster-spotter **[READ]**; https://arxiv.org/abs/2109.12987 **[READ]**

---

### 8. Citation-Needed Work & Wikipedia Verifiability
* **Authors/Year:**
  * **Redi et al. ("Citation Needed"):** Miriam Redi, Besnik Fetahu, Jonathan Morgan, Dario Taraborelli (2019) **[READ]**.
  * **Wikipedia Verifiability (WP:V):** Wikipedia Community (WP:V / WP:CITE / WP:BLP) **[SNIPPET]**.
* **Categories:**
  * *Wikipedia Verifiability Policy Rule:* Any material that is `"challenged or likely to be challenged, and all quotations"` requires an inline citation **[SNIPPET]**.
    1. `"Quotations"`: Direct speech or text from any source **[SNIPPET]**.
    2. `"Challenged material"`: Material actively questioned by another reader/editor **[SNIPPET]**.
    3. `"Material likely to be challenged"`: Controversial claims, counter-intuitive data, non-obvious statistics **[SNIPPET]**.
    4. `"Contentious material about living persons"` (WP:BLP): Positive, negative, or neutral claims about living persons that could affect reputation (must be cited immediately or removed) **[SNIPPET]**.
    5. `"Common knowledge"`: Universal uncontested facts (exempt from citations) **[SNIPPET]**.
  * *Redi et al. 8-Class Citation-Reason Taxonomy:*
    `"Missing Information"`, `"Medical Content"`, `"Point of View (POV)"`, `"Doubt"`, `"Vague"`, `"Jargon"`, `"Unclear"`, `"Sic"` **[SNIPPET]**.
* **Example Statements:**
  * `"Quotations"`: *"Ask not what your country can do for you..."* **[SNIPPET]**.
  * `"Contentious material about living persons"`: *"Senator Doe received an undisclosed $50,000 payment from a lobbyist in 2021."* **[SNIPPET]**.
  * `"Medical Content"`: *"St. John's Wort significantly reduces the efficacy of oral contraceptives."* **[SNIPPET]**.
  * `"Common knowledge"`: *"The Pacific Ocean is the largest ocean on Earth."* **[SNIPPET]**.
* **Which Require Citation:** All quotations, challenged/likely challenged claims, BLP statements, and Redi categories 1–8 **[READ]** **[SNIPPET]**. Common knowledge is exempt **[SNIPPET]**.
* **Detection Mechanism:** Recurrent neural nets / FastText / BERT trained on `[citation needed]` revision histories **[READ]**; regex rules for quotation marks and BLP entities **[SNIPPET]**.
* **URLs:** https://arxiv.org/abs/1902.11116 **[READ]**; https://en.wikipedia.org/wiki/Wikipedia:Verifiability **[SNIPPET]**

---

### 9. High-Risk Domains: Google YMYL & Medical/Legal Norms
* **Authors/Year:**
  * **Google Search Quality Evaluator Guidelines (SQRG):** Google Search Quality Team (updated 2022/2023) **[SNIPPET]**.
  * **Evidence-Based Medicine (EBM):** Evidence-Based Medicine Working Group (Guyatt et al., 1992 / ICMJE) **[SNIPPET]**.
  * **Legal Citation Norms:** *The Bluebook: A Uniform System of Citation* (Columbia/Harvard Law Review) **[SNIPPET]**.
* **Categories:**
  * *Google YMYL 3-Tier Spectrum:*
    1. `"Clear YMYL"`: Topics directly impacting health, financial stability, physical safety, or civic welfare **[SNIPPET]**.
    2. `"May be YMYL"`: Topics that could influence significant life decisions depending on context (e.g. car purchases, fitness routines) **[SNIPPET]**.
    3. `"Not or Unlikely YMYL"`: Low-risk everyday topics (entertainment, hobbies, static history) **[SNIPPET]**.
    * *Topical subcategories:* `"Health and Safety"`, `"Financial Security"`, `"Civics, Government, and Law"`, `"News and Current Events"`, `"Shopping (E-commerce)"`, `"Groups of People"`, `"Other Major Life Decisions"` **[SNIPPET]**.
  * *Medical Citation Hierarchy:*
    1. `"Therapeutic and interventional claims"`: Must cite Level 1 evidence (Systematic reviews, RCTs) **[SNIPPET]**.
    2. `"Diagnostic and prognostic claims"`: Must cite prospective cohort/accuracy studies **[SNIPPET]**.
    3. `"Clinical practice guidelines"`: Must cite recognized governing bodies (e.g. WHO, CDC, USPSTF) **[SNIPPET]**.
  * *Legal Citation Norms:*
    1. `"Propositions of law"`: Must cite binding primary legal authority (statutes, case holdings, regulations) **[SNIPPET]**.
    2. `"Direct quotations of legal text"`: Must include exact pinpoint authority citations **[SNIPPET]**.
* **Example Questions:**
  * `"Clear YMYL"` / `"Medical"`: *"What is the pediatric emergency dosage for liquid ibuprofen?"* **[SNIPPET]**.
  * `"Clear YMYL"` / `"Legal"`: *"What is the statutory deadline to appeal a small claims ruling in New York?"* **[SNIPPET]**.
  * `"Not YMYL"`: *"Who won the best sound editing award at the 1998 Oscars?"* **[SNIPPET]**.
* **Which Require Citation:** `"Clear YMYL"`, all clinical dosages/treatments, and all formal propositions of law ALWAYS require retrieval and primary source citation **[SNIPPET]**.
* **Detection Mechanism:** Keyword domain lexicons (pharmacopeia, statutes, civic codes), classifier pipelines, or query intent classifiers **[SNIPPET]**.
* **URLs:** https://blog.google/products/search/overview-our-rater-guidelines-search/ **[SNIPPET]**; https://www.legalbluebook.com/ **[SNIPPET]**

---

## Proposed Taxonomy for Our Assistant

Designed for our spoken voice assistant (small local router + Gemini 3.1 flash-lite + Tavily search; spoken answer in 1–3 short sentences):

| Category | Example Question | Needs Source? | Staleness Class | How to Detect Cheaply (Regex or Tiny Classifier) | Evidence Source |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. High-Stakes Harm (Clear YMYL: Medical, Legal, Financial, Civic)** | *"Can I take ibuprofen with lisinopril?"* / *"Where is my polling place today?"* | **Always** | Volatile to Medium; high-consequence | Regex for medical/legal/civic lexicons (`dosage`, `drug`, `symptoms`, `statute`, `tax rate`, `vote`, `ballot`) or tiny intent classifier | Google SQRG `Clear YMYL` **[SNIPPET]**; Redi et al. `Medical Content` **[READ]**; EBM Norms **[SNIPPET]** |
| **2. Dynamic Real-World Facts (Roles, Prices, Versions, Records, Events)** | *"Who is the current Prime Minister of the UK?"* / *"What was Nvidia's stock close today?"* | **Always** (Changeable) | Fast-moving (daily to quarterly) | Expanded regex: temporal triggers (`current`, `latest`, `new`, `today`, `now`, `upcoming`) + entity roles (`CEO`, `mayor`, `champion`, `price`, `version`, `record`, `winner`) | Self-RAG `[Retrieve]` **[READ]**; ClaimBuster `CFS` **[SNIPPET]**; CheckThat! `CW` **[READ]** |
| **3. Long-Tail & Obscure Entities** | *"Who founded Blue River Design in Portland?"* / *"What is the menu at Cafe Brioche in Aptos?"* | **Always** | Slow-moving, but absent in weights | Low-frequency NER entities, unknown proper nouns, or out-of-vocabulary check against a local head-entity dictionary | Mallen et al. (PopQA) `Tail` **[READ]**; Adaptive-RAG `Single-Step` **[READ]** |
| **4. Direct Quotes, Superlatives & Disputed Facts** | *"What was Steve Jobs's exact quote on staying foolish?"* / *"Is butter officially healthy now?"* | **Always** | Static to Contested | Regex for quote indicators (`"..."`, `quote`, `exact words`, `who said`) or dispute cues (`is it true that`, `controversy`, `best vs worst`) | Wikipedia WP:V `Quotations` & `Challenged or likely to be challenged` **[SNIPPET]**; Redi et al. `Doubt` **[READ]** |
| **5. Stable Head World Knowledge (Common Knowledge & Science)** | *"What is the boiling point of water at sea level?"* / *"Who wrote Pride and Prejudice?"* | **Never** | Immutable / Permanent | Fast dictionary of high-frequency head entities/concepts + negative match against temporal/YMYL/quote triggers | Mallen et al. `Head` **[READ]**; Adaptive-RAG `Non-Retrieval` **[READ]**; Wikipedia `Common knowledge` **[SNIPPET]** |
| **6. Conversational, Subjective & Procedural Reasoning** | *"Tell me a joke about dogs."* / *"What is 34 times 19?"* / *"How do I poach an egg?"* | **Never** | Timeless / Procedural | Regex for conversational fillers, greetings, math operators (`[0-9]+ * [0-9]+`), and reasoning prompts (`why do you think`, `tell me a joke`) | Rashkin et al. (AIS) `Uninterpretable` **[READ]**; ClaimBuster `NFS` **[SNIPPET]**; CheckThat! `NCW` **[READ]** |

---

## What We Should NOT Adopt and Why

1. **Sentence-by-Sentence Iterative Drafting & Regeneration (FLARE / Self-RAG Beam Search):**
   * *Why not:* Severe latency penalty. Forward-looking token confidence checks and multi-pass sentence drafting require 3–5 sequential model inferences. In a spoken voice interface, time-to-first-audio must stay under 1.2 seconds; iterative decoding would push response latency past 4–6 seconds.
2. **Post-Hoc Atomic Claim Decomposition (FActScore / SAFE Pipeline):**
   * *Why not:* Atomic extraction requires an extra LLM call to break responses into discrete propositions, followed by multiple search queries per atom. While excellent for offline academic evaluation, running this pipeline online inside a voice router destroys real-time performance.
3. **Multi-Hop Graph Decomposition (Adaptive-RAG Multi-Step Routing):**
   * *Why not:* Spoken voice queries are overwhelmingly conversational single-hop requests. Implementing a multi-hop orchestrator adds unnecessary architecture complexity and causes the assistant to over-retrieve and ramble beyond 1–3 short spoken sentences.
4. **Academic / Pinpoint Bracketed Citations (ALCE Numbers / Bluebook Signals):**
   * *Why not:* Text-to-speech (TTS) engines sound unnatural reading out numbers like *"bracket one"* or dense legal citations. Spoken attribution must be conversational (e.g., *"According to Reuters..."* or *"Mayo Clinic advises that..."*).
5. **Relying Exclusively on Model Uncertainty / Token Probabilities (FLARE):**
   * *Why not:* Modern instruction-tuned models hallucinate outdated facts with high token probabilities on changing roles and recent events. Pre-retrieval query routing based on semantic intent, YMYL risk, and entity staleness is far safer than waiting for generation uncertainty.
