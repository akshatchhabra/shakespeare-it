# Shakespeare it!
A text style transfer model that converts general English text to Shakespearean English

This repo contains our code and experiments for the final project for COMPSCI 685.

# Team Members
1. Akshat Chhabra (akshatchhabr@umass.edu)
2. Ala Hashesh (ahashesh@umass.edu)
3. Jiachang Situ (jiachangsitu@umass.edu)
4. Tenzin Nanglo (tnanglo@umass.edu)

# Drive link (Code and data)

https://drive.google.com/drive/folders/176_V3hUNBvwdqw9WeYJjHSQNn8fY7Q2K?usp=sharing

# Content

0. `Baseline.ipynb`: This contains our initial experiments with very simple models built using RNNs with Attention Mechanisms to see that we are on the right track.

1. `Classifier.ipynb`: Here we use `BertForSequenceClassification` and fine-tune it to train a classifier that can distinguish between regular English and Shakespeare English. This classifier can achieve 85% accuracy.

2. `project.ipynb`, `Project Optimization.ipynb` and `Experiments*.ipynb` contain our experiments and optimization attempts to get a good working style transfer model that can transfer text from English to Shakespeare English.
   
   In all attempts, we use a pre-trained T5 model, and we fine-tune it on style transfer. The model was fine-tuned on a parallel data set that contains `18396` sentence pairs. We used 80% of that as training data, 10% as validation data and 10% for testing.

   Our best model was trained using `lr=5e-5` and `batch_size=8`. This model was trained for 5 epochs.
    

3. `generate pseudo parallel.ipynb`: Has the code that we used to build our psuedo parallel dataset. Our approach here is to use a paraphraser trained in regular English that can paraphrase Shakespearean to a given output. That output will be in regular English then we can build our dataset by using the paraphrase Shakespearean as input and the original Shakespearean as output. Here we used `PegasusForConditionalGeneration` as a the paraphraser.

4. `finetune parallel jiachang.ipynb` and `Ala non parallel data model opt.ipynb` contains our experiments for fine-tuning our T5 model on the pseudo-parallel dataset. We ran several experiments here and our best model was trained for 10 epochs using `lr=5e-5` and `batch_size=8`.

5. `Test Classifier.ipynb`: First in other notebooks we feed our testing data to our two models (parallel vs non-parallel) then we run our classifier on the generated output. Here our classifier is evaluating how good the model is. Our parallel model got 83% and our non-parallel got 44.6%.

6. Our code and analysis for Evaluation metrics can be found under the "Evaluation" folder. Here we did all the analysis and computation for fluency, sentence similarity and others.