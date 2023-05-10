import pickle
import re
from copy import copy
from random import sample

import numpy as np
import pandas as pd
import spacy
from scipy.sparse import hstack
from torch.nn.functional import cosine_similarity
from tqdm import tqdm

IS_LEMMA = True
HAVE_ROLE = False
EXTEND_ONTOLOGY_WHEN_LEMMA = False
CREATE_NEW_DIALOGUE = True
list_pos_tags = ["NN", "NNS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "MD"]
term_definition_cache = {}
ontology_node_dict = {}
syns_cache = {}
model_inp_config = {'_ALLERGY': 'tf_input_aug',
                    '_ASSESSMENT': "",
                    '_CC': "",
                    '_DIAGNOSIS': "",
                    '_DISPOSITION': "tfidf_input_aug",
                    '_EDCOURSE': "",
                    '_EXAM': "",
                    '_FAM/SOCHX': "tfidf_input_aug",
                    '_GENHX': "",
                    '_GYNHX': "",
                    '_IMAGING': "",
                    '_IMMUNIZATIONS': '',
                    '_LABS': '',
                    '_MEDICATIONS': 'tfidf_input_aug',
                    '_OTHER_HISTORY': 'tfidf_input_aug',
                    '_PASTMEDICALHX': 'tfidf_input_aug',
                    '_PASTSURGICAL': 'tf_input_aug',
                    '_PLAN': "",
                    '_PROCEDURES': "",
                    '_ROS': 'tf_input_aug'}
model_weight = {'ALLERGY': 1,
                'ASSESSMENT': 1,
                'CC': 1,
                'DIAGNOSIS': 1,
                'DISPOSITION': 2,
                'EDCOURSE': 1,
                'EXAM': 1,
                'FAM/SOCHX': 1,
                'GENHX': 1,
                'GYNHX': 1,
                'IMAGING': 1,
                'IMMUNIZATIONS': 1,
                'LABS': 2,
                'MEDICATIONS': 1,
                'OTHER_HISTORY': 1,
                'PASTMEDICALHX': 1,
                'PASTSURGICAL': 1,
                'PLAN': 2,
                'PROCEDURES': 1,
                'ROS': 1}
priority_queue = [
    {
        'cate': 'LABS',
        'type': 'keyword_boost',
        'keywords': [r'^result.*(of)$', 'test', 'ultrasound report', 'blood report'],
        'boost_to': 1,
        'boost_under': 1e-3
    },
    {'cate': 'GYNHX',
     'type': 'keyword_boost',
     'keywords': ['pregnant', 'miscarriages', 'abortions', 'menstrual cycle',
                  'ovaries', 'ovary', 'uterus', 'cervix', 'hormonal therapies',
                  'pap smear results', 'mammogram screening', 'hysterectomies'],
     'boost_to': 1,
     'boost_under': 1e-2
     },
    {'cate': 'IMAGING',
     'type': 'keyword_boost',
     'keywords': ['x-ray', 'E K G',
                  'C B C', 'CBC exam', 'complete blood count',
                  'radiation treatment', 'EKG', 'electrocardiogram'],
     'boost_to': 1,
     'boost_under': 1e-2
     },
    {'cate': 'IMMUNIZATIONS',
     'type': 'keyword_boost',
     'keywords': ['accination', 'vaccine', 'shot', 'immunizations'],
     'boost_to': 1,
     'boost_under': 1e-3
     },
]

SBERT = None


def sbert_sim(sentence1, sentence2) -> float:
    global SBERT
    if SBERT is None:
        from sentence_transformers import SentenceTransformer
        SBERT = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings1 = SBERT.encode(sentence1, convert_to_tensor=True)
    embeddings2 = SBERT.encode(sentence2, convert_to_tensor=True)
    # Compute the cosine similarity between the sentence embeddings
    cos_sim = cosine_similarity(embeddings1.reshape(1, -1), embeddings2.reshape(1, -1)).item()
    return cos_sim


def keyword_boost(text: str,
                  score: float,
                  boost_under: float,
                  keywords: list,
                  boost_to: float = None,
                  boost_by: float = None, **kwargs) -> float:
    if score > boost_under:
        return score
    boost = False
    for k in keywords:
        pattern = re.compile(k, re.IGNORECASE)
        matches = pattern.findall(text)
        boost = boost or len(matches) > 0
        if boost:
            break
    if boost:
        if boost_to is not None:
            return boost_to
        return score * boost_by
    return score


def semantic_boost(text: str,
                   score: float,
                   boost_under: float,
                   close_to: str,
                   threshold: float,
                   uttn_pooling: bool = True,
                   boost_to: float = None,
                   boost_by: float = None, **kwargs) -> float:
    if score > boost_under:
        return score
    if uttn_pooling:
        cos_sim = max([sbert_sim(text, uttn) for uttn in text.split('\n')])
    else:
        cos_sim = sbert_sim(text, close_to)
    boost = cos_sim >= threshold
    if boost:
        if boost_to is not None:
            return boost_to
        return score * boost_by
    return score


def strict_boost(data: pd.DataFrame,
                 predicted_weight: pd.DataFrame,
                 priority_queue: list) -> pd.DataFrame:
    predicted_weight = copy(predicted_weight)
    pool = {
        'keyword_boost': keyword_boost,
        'semantic_boost': semantic_boost
    }
    assert len(data) == len(predicted_weight)
    for each_boost in tqdm(priority_queue, 'Boosting predicted weight...'):
        cate = each_boost['cate']
        for idx, (text, score,) in enumerate(zip(data['dialogue'].tolist(), predicted_weight[cate].tolist())):
            predicted_weight[cate][idx] = pool[each_boost['type']](text=text, score=score, **each_boost)
    return predicted_weight


def run2(dataset_url: str,
         id_column: str,
         dialogue_column: str,
         model_url: str,
         model_cc_url: str,
         **kwargs) -> pd.DataFrame:
    nlp = spacy.load("en_core_web_sm")
    taskA_val = pd.read_csv(dataset_url)
    assert id_column in taskA_val.columns and dialogue_column in taskA_val.columns, \
        f'{id_column} or {dialogue_column} not found in input dataset. columns: {taskA_val.columns}'
    taskA_val.rename(columns={id_column: 'TestID', dialogue_column: 'dialogue'}, inplace=True)

    f = open(model_url, "rb")
    model_configs = pickle.load(f)
    f = open(model_cc_url, "rb")
    cc_config = pickle.load(f)

    def extend_definition(spacy_doc):
        definitions = []
        for token in spacy_doc:
            lemma = token.lemma_
            tag = token.tag_
            if tag in list_pos_tags:
                if lemma not in term_definition_cache:
                    find = False
                    for nodeID, node in ontology_node_dict.items():
                        if lemma in node.terms:
                            definition = nlp(node.define)
                            term_definition_cache[lemma] = " ".join([x.lemma_ for x in definition])
                            find = True
                            break
                    if not find: term_definition_cache[lemma] = ""
                definition = term_definition_cache[lemma]
                if definition != "": definitions.append(definition)
        return definitions

    def replace_by_synonyms_ontology(spacy_doc):
        new_docs = set()
        for token in spacy_doc:
            lemma = token.lemma_
            tag = token.tag_

            if tag in list_pos_tags and len(lemma) > 3:
                if lemma not in syns_cache:
                    find = False
                    for nodeID, node in ontology_node_dict.items():
                        if lemma in node.terms:
                            print("Find {} in {}".format(lemma, node.treeNumbers))
                            for term in node.terms:
                                syns_cache[term] = node.terms
                            find = True
                            break
            if lemma in syns_cache:
                words = [x.lemma_ for x in spacy_doc]
                new_lemmas = [x for x in syns_cache[lemma] if len(x) > 2 and x != lemma and lemma not in x]
                if len(new_lemmas) > 1: new_lemma = sample(new_lemmas, 1)

                for new_lemma in new_lemmas:
                    #               print("replace {} by {}".format(lemma, new_lemma))
                    new_doc = " ".join(words).replace(lemma, new_lemma)
                    new_docs.add(new_doc)
        return new_docs

    def init_from_str_dialogue(text, is_lemma=False, have_role=False, create_new_dialogue=CREATE_NEW_DIALOGUE,
                               extend_ontology_when_lemma=EXTEND_ONTOLOGY_WHEN_LEMMA):
        role_pattern = '^[A-Z].*?:'
        Utterances_with_role = text.split('\n')
        roles = []
        utterances = []

        for each in Utterances_with_role:
            if not each:
                continue
            role = re.findall(role_pattern, each)
            if len(role) == 0:
                utterances.append(each)
            else:
                role = role[0].strip()
                roles.append(role)
                utterance = re.sub(role_pattern, '', each)
                utterances.append(utterance)

        if len(roles) == 0 or have_role == False:
            roles = ["" for x in utterances]

        results = []
        ontology_aug = dict()

        if not is_lemma:
            for utterance, role in zip(utterances, roles):
                results.append(role + utterance)
            return " ".join(results), []
        else:
            for spacy_doc, role in zip(nlp.pipe(utterances), roles):
                words = [x.lemma_ for x in spacy_doc]
                dia = role + " ".join(words)
                results.append(dia)

                if extend_ontology_when_lemma:
                    definitions = extend_definition(spacy_doc)
                    for definition in definitions:
                        results.append(definition)

                if create_new_dialogue:
                    new_dias = replace_by_synonyms_ontology(spacy_doc)
                    for new_dia in new_dias:
                        ontology_aug[dia] = new_dia

            return " ".join(results), ontology_aug

    def preprocessing_dialogue(df, datatype):
        result = df.copy()
        count = len(df)

        for index, row in tqdm(df.iterrows(), 'Preprocessing'):
            result.at[index, "dialogue"], onto_aug = init_from_str_dialogue(row['dialogue'], is_lemma=IS_LEMMA,
                                                                            have_role=HAVE_ROLE,
                                                                            create_new_dialogue=False)
        return result

    taskA_val1 = preprocessing_dialogue(taskA_val, datatype="validation")

    def feature_extraction(text, is_lemma=False, have_role=False):
        role_pattern = '^[A-Z].*?:'
        Utterances_with_role = text.split('\n')
        roles = []
        utterances = []
        for each in Utterances_with_role:
            if not each:
                continue
            role = re.findall(role_pattern, each)
            if len(role) == 0:
                utterances.append(each)
            else:
                role = role[0].strip()
                roles.append(role)
                utterance = re.sub(role_pattern, '', each)
                utterances.append(utterance)

        pos_tags = dict()
        for spacy_doc in nlp.pipe(utterances):
            for word in spacy_doc:
                tag = word.tag_
                if tag in list_pos_tags:
                    pos_tags[tag] = pos_tags.get(tag, 0) + 1
        return len(utterances), pos_tags

    def manual_features_extraction(df):
        result = pd.DataFrame()

        input_df = df.copy()
        for index, row in input_df.iterrows():
            result.at[index, "length_dialogue"], pos_tags = feature_extraction(row['dialogue'])
            for tag in pos_tags:
                result.at[index, tag] = pos_tags[tag]
        result = result.fillna(0)
        return result

    x_manual_test = manual_features_extraction(taskA_val).to_numpy()
    test_results = []
    for model_config in model_configs:
        col = "_".join(list(model_config)[0].split("_")[1:])
        if col == "CC":
            model_config = cc_config[0]
        vectorizer = model_config['text_encoder']
        X_test = vectorizer.transform(taskA_val1['dialogue'])
        selector = model_config['feature_selection']
        selected_X_test = selector.transform(X_test)
        selected_X_test = hstack((selected_X_test, x_manual_test))
        scaler = model_config['scaler']
        selected_X_test = scaler.transform(selected_X_test.toarray())
        model = model_config['clf' + "_" + col]
        y_proba = model.predict_proba(selected_X_test)
        new_record = {'clf' + "_" + col: y_proba}
        test_results.append(new_record)
    pdf = pd.DataFrame()
    labels = []

    for idx, model in enumerate(test_results):
        label = "_".join(list(model)[0].split("_")[1:])
        labels.append(label)
        pdf[label] = model[list(model)[0]][:, 1]

    for weight in model_weight:
        pdf[weight] = pdf[weight] * model_weight[weight]
    boosted_pdf = strict_boost(taskA_val, pdf, priority_queue)
    y_pred = np.argmax(boosted_pdf.values, axis=-1)
    y_pred_labels = [labels[x] for x in y_pred]
    # scaler = StandardScaler()
    # pdf = scaler.fit_transform(pdf)
    # label_predict = np.argmax(pdf, axis=-1)
    # label_predict = [labels[x] for x in label_predict]
    taskA_val1['SystemOutput'] = y_pred_labels
    return taskA_val1
