
"""EVALUATION OF CONTENT PRESERVATION

This code can be used for evaluation of content preservation between input and output sentiment texts of a style transfer model.

Word Mover's Distance (WMD) on texts with style masking (i.e. placeholders used in place of style words) 
exhibited the highest correlation with human evaluations of the same texts.



Usage:
    - Mask style words in a set of texts prior to evaluation                -> mark_style_words(texts, mask_style=True)
    - View correlations between automated metrics and human scores          -> display_correlation_tables()
    - Load WMD scores for output texts of examined style transfer models    -> load_wmd_scores(...)
    - Train a Word2Vec model for your dataset, for use in WMD calculation   -> train_word2vec_model(...)
    - Calculate WMD scores for your own input/output texts                  -> calculate_wmd_scores(...)

You can find examples of more detailed usage commands below.

"""

import sys 
sys.path.append('/content/drive/MyDrive/UMass/Spring 2022/COMPSCI685/project /Evaluation/code')
import csv
import numpy as np
import pdb
from gensim.models.word2vec import Word2Vec


# ASPECT = 'content_preservation'
# AUTOMATED_SCORES_PATH = '../evaluations/automated/content_preservation/sentence_level'




from tokenizer import tokenize
from utils import calculate_correlations, get_val_as_str, load_dataset, load_turk_scores, merge_datasets
from style_lexicon import load_lexicon#
STYLE_LEXICON = load_lexicon
STYLE_MODIFICATION_SETTINGS = ['style_masked', 'style_removed']
CUSTOM_STYLE = 'customstyle' 


# DATA PREP
def mark_style_words(texts, style_tokens=STYLE_LEXICON, mask_style=False):
    '''
    Mask or remove style words (based on a set of style tokens) from input texts.

    Parameters
    ----------
    texts : list
        String inputs
    style_tokens : set
        Style tokens
    mask_style : boolean
        Set to False to remove style tokens, True to replace with placeholder
        
    Returns
    -------
    edited_texts : list
        Texts with style tokens masked or removed

    '''
    
    edited_texts = []
    
    for text in texts:
        tokens = tokenize(text)
        edited_tokens = []
        
        for token in tokens:
            if token.lower() in style_tokens:
                if mask_style:
                    edited_tokens.append(CUSTOM_STYLE)
            else:
                edited_tokens.append(token)
            
        edited_texts.append(' '.join(edited_tokens))

    return edited_texts

def generate_style_modified_texts(texts):
    # ensure consistent tokenization under different style modification settings 
    unmasked_texts = mark_style_words(texts, {}) 
    texts_with_style_removed = mark_style_words(texts)
    texts_with_style_masked = mark_style_words(texts, mask_style=True)
    return unmasked_texts, texts_with_style_removed, texts_with_style_masked


## MODELS / SCORING OF WMD


def train_word2vec_model(texts, path):
    tokenized_texts = []
    for text in texts:
        tokenized_texts.append(tokenize(text))
    model = Word2Vec(tokenized_texts)
    model.save(path)

def load_word2vec_model(path):
    model = Word2Vec.load(path)
    model.init_sims(replace=True) # normalize vectors
    return model

def calculate_wmd_scores(output_file_path, wmd_model):
    '''
    Calculate Word Mover's Distance for each (reference, candidate)
    pair in a list of reference texts and candidate texts.
    
    The lower the distance, the more similar the texts are.

    Parameters
    ----------
    references : list
        Input texts
    candidates : list
        Output texts (e.g. from a style transfer model)
    wmd_model : gensim.models.word2vec.Word2Vec
        Trained Word2Vec model
        
    Returns
    -------
    wmd_scores : list
        WMD scores for all pairs 

    '''
    wmd_scores = [] 
    at_start = True
    with open(output_file_path) as csv_file:
      csv_reader = csv.reader(csv_file, delimiter=',')
      for row in csv_reader:
        if at_start == True:
          at_start = False
          continue
        wmd = wmd_model.wv.wmdistance(tokenize(row[0]), tokenize(row[1]))
        wmd_scores.append(wmd)

    return wmd_scores

# def load_wmd_scores(model_name, param_val):
#     '''
#     Load pre-computed WMD scores for input and output texts under
#     the style masking setting. (Style masking exhibited higher
#     correlation with human scores than other settings).

#     Parameters
#     ----------
#     model_name : str
#         Name of style transfer model
#     param_val : float
#         Parameter on which the model was trained (see MODEL_TO_PARAMS for options)
        
#     Returns
#     -------
#     List of WMD scores for all pairs of input and output texts

#     '''
    
#     param_name = MODEL_TO_PARAM_NAMES[model_name]
#     string_val = get_val_as_str(param_val)
#     metrics_path = f'{AUTOMATED_SCORES_PATH}/{model_name}_{param_name}_{string_val}.npz'
#     return np.load(metrics_path)['style_masked'].item()['WMD']


## CALCULATION OF CORRELATIONS
# def display_correlation_tables():
#     '''
#     Display correlation of automated content preservation metrics with
#     averaged human evaluation scores for examined style transfer models 
#     over texts under different style modification settings.
    
#     '''
    
#     for setting in STYLE_MODIFICATION_SETTINGS:
#         print()
#         print(f'[Setting: {setting.upper()}]')
        
#         for model in MODEL_TO_PARAMS:
#             print()
#             print(model)

#             param_name = MODEL_TO_PARAM_NAMES[model]
#             param_values = MODEL_TO_PARAMS[model]

#             metrics_scores_over_model_params = {}
#             turk_scores_over_model_params = []

#             for val in param_values:
#                 string_val = get_val_as_str(val)
#                 metrics_path = f'{AUTOMATED_SCORES_PATH}/{model}_{param_name}_{string_val}.npz'
#                 all_metrics = np.load(metrics_path)

#                 # load scores for style modification setting
#                 metrics = all_metrics[setting].item()

#                 # aggregate scores obtained over all model parameters 
#                 for metric_name in metrics:
#                     # metric_values is a list of sentence-level scores
#                     metric_values = metrics[metric_name]
#                     metrics_scores_over_model_params.setdefault(metric_name, []).extend(metric_values)
#                 turk_scores_over_model_params.extend(load_turk_scores(ASPECT, model, param_name, string_val))

#             correlation_tables = calculate_correlations(metrics_scores_over_model_params, turk_scores_over_model_params)
#             print(correlation_tables.round(decimals=3).transpose())
#             print()

if __name__ == "__main__":
  train_data = load_dataset('/content/drive/MyDrive/UMass/Spring 2022/COMPSCI685/project /Sanity Check/data/data.csv')
  w2v_model_path = '/content/drive/MyDrive/UMass/Spring 2022/COMPSCI685/project /Evaluation/models/word2vec_unmasked'
  unmasked_inputs, inputs_with_style_removed, inputs_with_style_masked = generate_style_modified_texts(train_data)
  #train_word2vec_model(train_data, w2v_model_path)
  w2v_model = load_word2vec_model(w2v_model_path)
  content_scores = calculate_wmd_scores('/content/drive/MyDrive/UMass/Spring 2022/COMPSCI685/project /Sanity Check/outputsFinal/predictions.csv', w2v_model)
  print(content_scores)    



# EXAMPLE USAGE (uncomment the following to play around with code)

# # load data to train models used for WMD calculations 
# train_data = load_dataset('/content/drive/MyDrive/UMass/Spring 2022/COMPSCI685/project /Sanity Check/data/data.csv')
# train_data_style_masked = mark_style_words(train_data, mask_style=True)

# train models
# w2v_model_path = '/content/drive/MyDrive/UMass/Spring 2022/COMPSCI685/project /Evaluation/models/word2vec_unmasked'
# w2v_model_style_masked_path = '/content/drive/MyDrive/UMass/Spring 2022/COMPSCI685/project /Evaluation/models/word2vec_masked'
# # train_word2vec_model(train_data, w2v_model_path)
# # train_word2vec_model(train_data_style_masked, w2v_model_style_masked_path)
# w2v_model = load_word2vec_model(w2v_model_path)
# w2v_model_style_masked = load_word2vec_model(w2v_model_style_masked_path)

# # load texts under different style modification settings
# input_neg_texts = load_dataset('../data/sentiment.test.0')
# input_pos_texts = load_dataset('../data/sentiment.test.1')
# input_texts = merge_datasets(input_neg_texts, input_pos_texts)
