# GCodeComp Kaggle Trial

NLP Kaggle 3-Month Competition [Found Here](https://www.kaggle.com/competitions/AI4Code/overview]). Selected Model was a Triple Ensemble Model using 2x GraphCodeBert's and a XLM-Roberta-Base. Built using Tensorflow to utilize TFRecords and TPU speed. Model is a subclassed structure that aims to combine cell and document embeddings to predict document (cell) order of a python notebook. Code cells are given in correct order but the markdown cells are scrambled within the given notebooks. Most cell outputs in this repository have been deleted to save space on my Google Drive for other work.

Final Score:
- 86-87 Kendall Tau Score, however peronsal test environment and dataset was around 88-89.
- Winning scores ranged between 90-92 Kendall Tau

Ideas Tested:
- Two-stage prediction model:
  - First stage would split the document into 4 parts and identify the quadrant that the markdown was located in
  - Second stage would identify its exact location wtihin the quadrant
- Pseudo combination of a pairwise and pointwise model. Each markdown would be processed seperately for the document and given a random selection of code cells. Each markdwon will be run with 3 different random samples and its pointwise location will be averaged amongst runs
- Generate markdown specific embeddings and rank them accordingly to each other then ensemble this with the pseudo combination model above
- Use previous layer outputs in the transformer model to stich together more richly defined embeddings using information gained from previous hidden states
- Manually gathered external datasets online and used a Selenium Driver to webscrap additional open source Kaggle notebooks and Juypter ML-based notebooks

What the highest ranked ML models used:
- Additional attention layers performed much better than dense layers on combining embeddings and contexts together
- Much larger inputs (1024 tokens compared to the standard of 512 tokens) proved to provide more context and thus better information to work with.
  - If the cell went over its allotted token length I selected, I used concatenations taking the first X/2 tokens and the last X/2 tokens for each cell, where X is the allotted token length per cell. This proved to not match the strength of larger token inputs.
- Model structures were better tailored for the problem to include stronger inductive biases for the problem. For insance, unique cell embeddings were used for each cell along with concatenations of transformed categorical features to match the dimension space of the embedding.

Thoughts:
- Weak performance
- Lost 1 month chasing a failed idea. Initial idea for a two-stage model was subject to unconditional probabliltiy if the model is not trained as a whole. For instance having 90% accuracy to get the correct quadrant location than 92% accuracy to get the correct location within the document would be only ~83%. The model must generate a markdown location with one run.
- Due to the nature of the scoring metric, larger documents had considerably more weight in the final score than smaller documents. When conducting random sampling, the larger documents produced poor code / document embeddings.
- Missed on an option to use a GDBT to help with the ensemble


- Utilized a combination of a pairwise / pointwise approach. Should have tried harder to conceive a strong listwise approach.
- Debarta V2 and Debarta V3 were stronger language models that should have been considered
- Less ideas can be tested now
- Less HypOp tuning, better to compare against
- Need to be better probably will swap to Pytorch when dealing with NLP models, less, however TPU training with TFRecords proved to be quite faster than V100 and T4 GPUs with XLA optimization.
- Still lack experience, should push harder for results and faster benchmarking.
- Using Debarta
- Top 
- Training was too slow
- External dataset
- Checked ensemble models with shuffle permutations and strength
-  Weren’t able to extract information from a markdown vector, not with a single transformer model.
-	Too much time on EDA, machine learning solutions seem to prefer better model structures over intensive EDA.
-	Model structure trumped ensemble, EDA, and fine-tuning.
