from dataclasses import dataclass
from typing import List, Dict, Any
import torch

from Config import Config


@dataclass
class DataCollatorForFinQA:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features = [f.__dict__ for f in features]
        batch = {}
        text_input_ids = []
        question_input_ids = []
        ge_input_ids = None
        text_attention_mask = []
        question_attention_mask = []
        ge_attention_mask = None
        text_token_type_id = []
        question_token_type_id = []
        ge_token_type_id = None
        text_num = []
        labels = []
        example_id = []

        def padding_len(i, j):
            return max_text_input_id_len - len(features[i]['texts_input_ids'][j])

        max_text_input_id_len = max([max(list(map(len, features[i]['texts_input_ids']))) for i in range(len(features))])
        max_question_input_id_len = max([len(features[i]['question_input_ids']) for i in range(len(features))])
        if features[0]['ge_input_ids'] is not None:
            max_ge_input_id_len = max([len(features[i]['ge_input_ids']) for i in range(len(features))])
            ge_input_ids = []
            ge_attention_mask = []
            ge_token_type_id = []

        for i in range(len(features)):
            if features[i]['ge_input_ids'] is not None:
                ge_padding_len = max_ge_input_id_len - len(features[i]['ge_input_ids'])
                ge_input_ids.append(
                    features[i]['ge_input_ids'] + [Config.tokenizer.pad_token_id] * ge_padding_len)
                ge_attention_mask.append(features[i]['ge_attention_mask'] + [0] * ge_padding_len)
                ge_token_type_id.append((features[i]['ge_token_type_ids'] + [0] * ge_padding_len)
                                        if features[i]['ge_token_type_ids'] is not None else [])
            question_padding_len = max_question_input_id_len - len(features[i]['question_input_ids'])
            example_id.append(features[i]['example_id'])
            text_num.append(len(features[i]['texts_input_ids']))

            text_input_ids += [features[i]['texts_input_ids'][j] + [Config.tokenizer.pad_token_id] * padding_len(i, j)
                               for j in range(len(features[i]['texts_input_ids']))]
            question_input_ids.append(
                features[i]['question_input_ids'] + [Config.tokenizer.pad_token_id] * question_padding_len)

            text_attention_mask += [features[i]['texts_attention_mask'][j] + [0] * padding_len(i, j) for j in
                                    range(len(features[i]['texts_attention_mask']))]
            question_attention_mask.append(features[i]['question_attention_mask'] + [0] * question_padding_len)

            text_token_type_id += ([features[i]['texts_token_type_ids'][j] + [0] * padding_len(i, j)
                                    for j in range(len(features[i]['texts_token_type_ids']))]) \
                if features[i]['texts_token_type_ids'] is not None else []
            question_token_type_id.append((features[i]['question_token_type_ids'] + [0] * question_padding_len)
                                          if features[i]['question_token_type_ids'] is not None else [])

            labels.append(features[i]['text_label'])

        new_labels = sum(labels, [])
        assert len(new_labels) == sum(text_num)

        batch['input_ids'] = text_input_ids
        batch['attention_mask'] = text_attention_mask
        batch['token_type_ids'] = text_token_type_id
        batch['query_input_ids'] = question_input_ids
        batch['query_attention_mask'] = question_attention_mask
        batch['query_token_type_ids'] = question_token_type_id
        batch['ge_input_ids'] = ge_input_ids
        batch['ge_attention_mask'] = ge_attention_mask
        batch['ge_token_type_ids'] = ge_token_type_id
        batch['text_num'] = text_num
        batch['labels'] = new_labels
        batch['document_mask'] = [[(x not in Config.punctuation_skiplist) and (x != 0) for x in d] for d in
                                  text_input_ids]
        batch['example_id'] = example_id

        for key in batch:
            try:
                batch[key] = torch.tensor(batch[key])
            except TypeError:
                print(key)
                print(batch[key])
                exit(0)

        return batch
