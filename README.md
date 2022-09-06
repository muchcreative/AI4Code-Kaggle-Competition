# GCodeComp Kaggle Trial

NLP Kaggle 3-Month Competition [Found Here](https://www.kaggle.com/competitions/AI4Code/overview]). Goal was to predict document (cell) order of a python notebook. Code cells are given in correct order but the markdown cells are scrambled within the given notebooks. Most cell outputs in this repository have been deleted to save space on my Google Drive for other work.

Personal Model was a Triple Ensemble Model using 2x GraphCodeBert's and a XLM-Roberta-Base with random code cell sampling to create document embeddings. Built using Tensorflow to utilize TFRecords and TPU speed. Model is a subclassed structure that aims to combine cell and document embeddings to generate the correct order of the markdown cells.

**Final Score:**
- 86-87 Kendall Tau Score, however peronsal test environment and dataset was around 88-89.
- Winning scores ranged between 90-92 Kendall Tau

**Ideas Tested:**
1. Two-stage prediction model:
    - First stage would split the document into 4 parts (quadrants) and identify the quadrant that the markdown was located in
    - Second stage would identify its exact location wtihin the quadrant
2. Pseudo combination of a pairwise and pointwise model. Each markdown would be processed seperately for the document and given a random selection of code cells. Each markdwon will be run with 3 different random samples and its final pointwise location will be averaged amongst its runs
3. Generate markdown specific embeddings and rank them accordingly to each other then ensemble this with the pseudo combination model above
4. Use previous layer outputs in the transformer model to stich together more richly defined embeddings using information gained from previous hidden states
5. Manually gathered external datasets online and used a Selenium Driver to webscrap additional open source Kaggle notebooks and Juypter ML-based notebooks

**What the highest ranked ML models used:**
- Additional attention layers performed much better than dense layers on combining embeddings and contexts together
- Much larger inputs (1024 tokens compared to the standard of 512 tokens) proved to provide more context and thus better information to work with.
  - If the cell went over its allotted token length, I used concatenations taking the first X/2 tokens and the last X/2 tokens for each cell, where X is the allotted token length per cell. This proved to not match the strength of pure larger token inputs.
- Model structures were better tailored for the problem to include better inductive biases for the problem. For insance, unique cell embeddings were used for each cell along with concatenations of transformed categorical features to match the dimension space of the new embedding.

**Thoughts:**
- Weak performance
- Lost 1 month chasing a failed idea. Initial idea for a two-stage model was subject to conditional probabliltiy if the model is not trained as a whole. For instance having 90% accuracy to get the correct quadrant location followed by a 92% accuracy to get the correct location within that quadrant would result in less than ~82% chance of the markdown being in the correct location at the end. The model must generate a correct markdown location with one run or stage.
- Due to the nature of the scoring metric, larger documents had considerably more weight in the final score than smaller documents. When conducting random sampling on larger documents, they produced poor code / document embeddings as the cells were more sparse and containted less significant information to create a comprehensive embedding for the document.
- Missed on an option to use a GDBT to help balance out the ensemble outputs
- Utilized a combination of a pairwise / pointwise approach. Should have tried harder to conceive a strong listwise approach which in-turn would improve model training speed.
- Debarta V2 and Debarta V3 were stronger language models that should have been considered.
- Ended up having to write a decent amount of Tensorflow functions to match known PyTorch modules. For example, layer-reinitalization testing for PLM models and specific learning rate schedulers like warmup with cosine decay restarts.
- TPU training with TFRecords proved to be quite faster than V100 and T4 GPUs along with XLA usage.
- External dataset did not provide as much boost as I was hoping for as different time periods of notebook creations ended up adding noise to the main train dataset
-  Wasn’t able to extract information from a markdown embedding due to FCL usage over attention layers
-	Too much time was spent on EDA and post-language processing, NLPs seem to prefer better model structures and good feature representation over preprocessing techniques.
