import pickle
import re
from random import sample
import pandas as pd
import spacy
from scipy.sparse import hstack
from tqdm import tqdm

IS_LEMMA = True
HAVE_ROLE = False
EXTEND_ONTOLOGY_WHEN_LEMMA = False
CREATE_NEW_DIALOGUE = True
list_pos_tags = ["NN", "NNS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "MD"]
term_definition_cache = {}
ontology_node_dict = {}
syns_cache = {}
def run3(dataset_url: str,
         id_column: str,
         dialogue_column: str,
         model_url: str,
         model_th: int,
         **kwargs) -> pd.DataFrame:
    nlp = spacy.load("en_core_web_sm")
    taskA_val = pd.read_csv(dataset_url)
    f = open(model_url, "rb")
    model_config = pickle.load(f)
    assert id_column in taskA_val.columns and dialogue_column in taskA_val.columns, \
        f'{id_column} or {dialogue_column} not found in input dataset. columns: {taskA_val.columns}'
    taskA_val.rename(columns={id_column: 'TestID', dialogue_column: 'dialogue'}, inplace=True)

    def extend_definition(spacy_doc):
        definitions = []
        for token in spacy_doc:
            lemma = token.lemma_
            tag = token.tag_
            if tag in list_pos_tags:
                if lemma not in term_definition_cache:
                    find = False
                    for nodeID, node in ontology_node_dict.items():  # what is ontology_node_dict?
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
            result.at[index, "dialogue"], onto_aug = init_from_str_dialogue(row['dialogue'],
                                                                            is_lemma=IS_LEMMA,
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
    model_config = model_config[model_th]
    label_encoder = model_config['label_encoder']
    vectorizer = model_config['text_encoder']
    X_test = vectorizer.transform(taskA_val1['dialogue'])
    selector = model_config['feature_selection']
    selected_X_test = selector.transform(X_test)
    selected_X_test = hstack((selected_X_test, x_manual_test))
    scaler = model_config['scaler']
    selected_X_test = scaler.transform(selected_X_test.toarray())
    model = model_config['clf']
    y_pred = model.predict(selected_X_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    taskA_val1['SystemOutput'] = y_pred_labels

    return taskA_val1
