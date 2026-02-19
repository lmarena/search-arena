<div align="center">
  <img src="logo.png" width="200" height="200">

  <h1>Search Arena: Analyzing Search-Augmented LLMs</h1>

  <p>
  <a href="https://legacy.lmarena.ai/?search">
    <img src="https://img.shields.io/badge/⚔️-Search Arena-green" alt="Search Arena" />
  </a>
    &nbsp;
    <a href="https://arxiv.org/abs/2506.05334">
      <img src="https://img.shields.io/badge/arXiv-2504.13169-b31b1b.svg" alt="arXiv" />
    </a>
    &nbsp;
    <a href="https://huggingface.co/datasets/lmarena-ai/search-arena-24k">
      <img src="https://img.shields.io/badge/HF-Dataset-orange?logo=huggingface&logoColor=white" alt="Dataset" />
    </a>
    &nbsp;
  <a href="https://x.com/lmarena_ai">
    <img src="https://img.shields.io/badge/LMArena--ai-white?logo=X&logoColor=000&color=000&logoColor=white" alt="X (Twitter)">
  </a>
    &nbsp;
  <a href="https://iclr.cc/Conferences/2026">
    <img src="https://img.shields.io/badge/ICLR-2026-blueviolet?logo=google-scholar&logoColor=white" alt="ICLR 2026" />
  </a>
  </p>

</div>

---

Welcome to the official code repository of **Search Arena**. Explore [⚔️ Search Arena](https://legacy.lmarena.ai/?search) and experiment with state-of-the-art Search-Augmented LLMs (new UI integration is coming soon)!

**Paper**: [**Search Arena: Analyzing Search-Augmented LLMs (ICLR 2026)**](https://arxiv.org/abs/2506.05334)

**Authors**: [Mihran Miroyan*](https://mmiroyan.github.io/), [Tsung-Han (Patrick) Wu*](https://tsunghan-wu.github.io/), [Logan King](https://www.linkedin.com/in/logan-king-8a4267281), [Tianle Li](https://codingwithtim.github.io), [Jiayi Pan](https://www.jiayipan.com/), [Xinyan Hu](https://sites.google.com/view/xinyanhu/), [Wei-Lin Chiang](https://infwinston.github.io/), [Anastasios N. Angelopoulos](https://people.eecs.berkeley.edu/~angelopoulos), [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/), [Narges Norouzi](https://nargesnorouzi.me//), [Joseph E. Gonzalez](https://people.eecs.berkeley.edu/~jegonzal/) (UC Berkeley)

Search-augmented AI is becoming the new search interface, making it critical to understand how users interact with these systems—what they ask and what they expect in return. 

## Key Findings from Search Arena
1. User queries are diverse: the collected datasets spans a wide range of topics (e.g., market analysis, health advice, shopping) and intents (e.g., factual lookups, recommendations, creative generation).

2. Users prefer longer responses, more citations, and specific sources (favoring coding sources and community/blog posts, disfavoring Wikipedia). We also find that reasoning improves model performance in search settings.

3. While users prefer more citations, models do not always cite retrieved sources accurately and humans do not always check citations!

4. Search-augmented models outperform standard LLMs on Search Arena without degrading performance on Text Arena.

Please read our paper for more details. This repo contains all scripts to reproduce the results presented in the paper.

## Repository Structure

Install all necessary dependencies: ``pip3 install -r requirements.txt``

**`data/`**
- `search_arena_24k.jsonl`: The full public Search Arena dataset, including conversations traces, user votes, system metadata, and more.
- `citation_attribution_data.jsonl`: A subset of the Search Arena dataset containing citation attribution metadata (refer to Section 3.2 of the paper for details).
- `text_arena_battles.jsonl`: Data from the Text Arena deployment used for cross-arena analysis (see Section 3.3 for further information).
- `describing_diffs/`: Processed text sets (responses and prompts), used in Describing Differences experiments.

**`scripts/`**
- `data_eda.ipynb`: Exploratory data analysis on Search Arena prompts, including language, country, turn count, prompt length, and intent distributions. See Section 2 (Appendix B) for results and discussion.
- `intent_classification/`: Run `OPENAI_API_KEY=sk-xxx python3 annotation.py` for LLM-based batch intent annotation. See Section 2.2 for more details.
- `preference_analysis.ipynb`: User preference analysis in the Search Arena dataset, including leaderboard computation (with ablations), model profiling, controlled Bradley-Terry experiments, and citation attribution analysis. See Sections 3.1 and 3.2 (Appendices D and E) for results and discussion.
- `citation_attribution_analysis/`: Run `GEMINI_API_KEY=xxx python3 annotation.py` to use an LLM to evaluate whether responses from search-augmented models are grounded in the retrieved web content. See Section 3.2 for more details.
- `cross_setting_analysis.ipynb`: Cross-setting analysis, including cross-arena deployment and cross-benchmark analysis. See Section 3.3 (Appendix F) for results and discussion.

**Notes:**  
- For experiments on describing differences (prompt and response set comparisons), we use code from the [Describing Differences](https://github.com/lisadunlap/DescribingDifferences) repository.  
- For topic modeling (Appendix B), we use the corresponding notebook from the Arena Explorer, available [here](https://colab.research.google.com/drive/1chzqjePYnpq08fA3KzyKvSkuzCjojyiE?usp=sharing).

## Citations
```
@inproceedings{miroyan2025searcharenaanalyzingsearchaugmented,
      title={Search Arena: Analyzing Search-Augmented LLMs}, 
      author={Mihran Miroyan and Tsung-Han Wu and Logan King and Tianle Li and Jiayi Pan and Xinyan Hu and Wei-Lin Chiang and Anastasios N. Angelopoulos and Trevor Darrell and Narges Norouzi and Joseph E. Gonzalez},
      year={2026},
      booktitle={The Thirteenth International Conference on Learning Representations},
      url={https://openreview.net/forum?id=MMGRlDnhtI}
}
```
