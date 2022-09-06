# GCodeComp Kaggle Trial

NLP Kaggle 3-Month Competition [Found Here]( https://www.kaggle.com/competitions/AI4Code/overview]). Selected Model was a Triple Ensemble Model using 2x GraphCodeBert's and a XLM-Roberta-Base. Built using Tensorflow to utilize TFRecords and TPU speed. Model is a subclassed structure that aims to combine cell and document embeddings to predict document (cell) order of a python notebook. Code cells are given in correct order but the markdown cells are scrambled within the given notebooks.

Ideas Tested:
- 


Problems:
- 

For Next Time on NLP problems:
- Less HypOp tuning, better to compare against
- Need to be better probably will swap to Pytorch when dealing with NLP models, less, however TPU training with TFRecords proved to be quite faster than V100 and T4 GPUs with XLA optimization.
- Still lack experience, should push harder for results and faster benchmarking.
- Using Debarta
- Top 


Summary Below:
- 86-87 Kendall Tau Score, however peronsal test environment and dataset was around 88-89.


Summarize:
-	You were lacking knowledge in the field to utilize a listwise approach and attention layer usage
-	If training takes to long, you can consider different model structures beside data structures
-	More context was stronger, inputs past 512 sequence lengths
-	Should have permutation shuffle for ensembles. Weren’t able to extract information from a markdown vector, not with a single transformer model.
-	Too much time on EDA, machine learning solutions seem to prefer better model structures over intensive EDA.
-	Model structure trumped ensemble, EDA, and fine-tuning.
-	Utilization of embeddings and expanded categorical to be concatenated on those embeddings.
-	Gathering additional data did not improve valid LB score substantially. You should have looked into this.
-	Single model or ensemble model proved to be way better than breaking the problem down and using multiple downstream machine learning models as they would be weak to Bayes Theorem.
-	Smaller batch sizes for longer training times with gradient clipping proved to be better here
-	Understanding of the loss function better, more emphasis on the larger documents for the Kendall Tau Score.
-	Reduce time for feedback next time. If you lose you lose, can’t change that, best to get what you can out of it.

-	How did they avoid limitations by cell counts with single attention models attending to each token?
-	How to use random boosted forest at the end of an ensemble model over linear layers
-	Utilizing positional encoding and multiple inputs
-	How were positional embeddings used?
-	How did they produce confidence interval for a continuous output?
-	Very ambitious architectures, a lot of testing needed to be done on them
o	The top models went with listwise understanding
Current ST:
-	Ok so everyone is on EDA right now. I thought it was a good way to get them comfortable with using the dataset and how it all works together and the different lines you have
-	Ask them how they feel. What they are thinking about the current status of the project, maybe they will have some actual inputs right now?
-	No pacing issues right now but have to start testing machine learning models
-	Eric has to get used to using the backtester and integrate the missing lines now.
-	Kevin and Robby to start testing models.
-	Eric still missing his last pull request and updating it as asked.
-	Currently have a problem with feature storage and ram overage. I should consider a feature store. To be talked about.
