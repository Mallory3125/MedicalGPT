import collections
import copy
import json
import re

from torch.utils.data import Dataset
from tqdm import tqdm  


PROMPT_EN = """
==Profile==
{role_profile}

==Conversations==
{conversations}

You are playing the role of {role_name}, you need to embody the knowledge and style of {role_name}.
Based on the provided role Profile and Conversations, please choose the best option (A, B, C, or D):
{options}

Your selection:
"""

PROMPT_ZH = """
==角色描述==
{role_profile}

==对话历史==
{conversations}

你要扮演{role_name}角色，你在聊天中要具备该角色对应的知识背景，语气风格等特征。
请根据所给的{role_name}角色描述和对话历史，从下面四个选项（A. B. C.和D.）中选择符合{role_name}的选项：
{options}

你的选择：
"""

PROMPT_DIALOGUE_EMOTION_EN = """
==Conversations==
{conversations}

Select the option () that best matches the mood in utterance "{utterance}". Single Choice
{options}

Your selection:
"""

PROMPT_DIALOGUE_EMOTION_ZH = """
==对话历史==
{conversations}

单选选择题，选择最符合"{utterance}"说话者当时心情的选项()
{options}

你的选择:
"""

PROMPT_OPEN_ZH = """
==角色描述==
{role_profile}

==对话历史==
{conversations}

你要扮演{role_name}角色，你在聊天中要具备该角色对应的知识背景，语气风格等特征。
请根据所给的{role_name}角色描述和对话历史，根据最后一个User的对话再补充一轮你作为Assistant的回复（一轮就好）：
Assistant: 
"""

PROMPT_OPEN_EN = """
==Profile==
{role_profile}

==Conversations==
{conversations}

You are playing the role of {role_name}, you need to embody the knowledge and style of {role_name}.
Based on the provided role Profile and Conversations, you must produce a reply as the Assistant to response to the latest User's message (one term is enough):
Assistant: 
"""

PROMPT_GROUP_EN = """
==Profiles==
{role_profiles}

==Conversations==
{conversations}

You are playing the role of {role_name}, you need to embody the social preference of {role_name}.
Based on the provided role profile and conversations, please choose the best option (A, B, C, or D) as your response:
{options}

Your selection (You can only output A, B, C or D, and no other characters.):
"""


PROMPT_GROUP_ZH = """
==角色描述==
{role_profiles}

==对话历史==
{conversations}

你要扮演{role_name}角色，你在聊天中要具备该角色对应的社交偏好。
请根据所给的{role_name}角色描述和对话历史，从下面四个选项（A. B. C.和D.）中选择最优的选项作为你的回复：
{options}

你的选择（你只能输出A，B，C或D，不要输出其他单词。）：
"""


def json_load(f):
    """Load a .json file into a dictionary."""
    if str(f).endswith(".json"):
        with open(f, 'r', encoding='utf-8') as reader:
            datalist = json.load(reader)
    elif str(f).endswith(".jsonl"):
        datalist = []
        with open(f, 'r', encoding='utf-8') as reader:
            for line in reader:
                datalist.append(json.loads(line))
    else:
        raise ValueError(f"Unexpected file type: {str(f)}")
    return datalist


def format_name(name: str) -> str:
    return name.replace(" ", "_").replace(".txt", "").replace(".json", "")


def make_group_profiles(profiles: dict, skip_role_name: str = None, shorten: bool = True, n: int = 10) -> str:
    results = []
    for role_name, role_profile in profiles.items():
        if skip_role_name is not None and format_name(role_name) == format_name(skip_role_name):
            continue
        results.append("\n".join(re.sub(r'\n+', '\n', role_profile).split("\n")[:n]) if shorten else role_profile)
    return "\n\n\n".join(results)


def format_question(dialogue, choices=None):
    conversations = ""
    for con in dialogue:
        role = con['from']
        text = con['value']
        conversations += f"{role}: {text}\n"

    options = ""
    if choices is not None:
        for choice, text in choices.items():
            options += f"{choice}. {text}\n"
    Output = collections.namedtuple('Output', ['dialogue', 'options'])
    return Output(dialogue=conversations, options=options)


def format_prompt(data):
    dialogue = data['dialogue']
    choices = data['choices'] if 'choices' in data else None
    category = data['meta']['category']
    lang = data['meta']['lang']
    outputs = format_question(dialogue, choices)
    if category == "Individual-MEM":
        PROMPT = PROMPT_OPEN_EN if lang.lower() == "en" else PROMPT_OPEN_ZH
        prompt = PROMPT.format_map({
            "role_profile": data['meta']['profile'][data['meta']['name']],
            "conversations": outputs.dialogue,
            "role_name": data['meta']['name'],
        })
    elif category == "Individual-EP-DialogueEmotionDetect":
        PROMPT = PROMPT_DIALOGUE_EMOTION_EN if lang.lower() == "en" else PROMPT_DIALOGUE_EMOTION_ZH
        prompt = PROMPT.format_map({
            "conversations": outputs.dialogue,
            "options": outputs.options,
            "utterance": dialogue[-1]["value"]
        })
    elif category in ["Individual-EP-HumorSarcasmDetect", "Individual-EP-SituationUnderstanding"]:
        prompt = f"{outputs.dialogue}\n{outputs.options}"
    elif category in ['Group-SAP-Positive', 'Group-SAP-Negative', 'Group-SAP-Neutral']:
        PROMPT = PROMPT_GROUP_EN if lang.lower() == "en" else PROMPT_GROUP_ZH
        prompt = PROMPT.format_map({
            "role_profiles": make_group_profiles(data['meta']['profile']),
            "conversations": outputs.dialogue,
            "role_name": data['meta']['name'],
            "options": outputs.options
        })
    elif category in ['Individual-SA-RoleStyle', 'Individual-SA-RoleKnowledge']:
        PROMPT = PROMPT_EN if lang.lower() == "en" else PROMPT_ZH
        prompt = PROMPT.format_map({
            "role_profile": data['meta']['profile'][data['meta']['name']],
            "conversations": outputs.dialogue,
            "role_name": data['meta']['name'],
            "options": outputs.options
        })
    else:
        raise ValueError(category)
    return prompt


class SocialBenchDataset(Dataset):
    def __init__(self, f: str, limit: int = None):
        self.datalist = json_load(f)
        if limit is not None:
            self.datalist = self.datalist[: limit]

    def __getitem__(self, i):
        data = copy.deepcopy(self.datalist[i])
        data['prompt'] = format_prompt(data)
        return data

    def __len__(self):
        return len(self.datalist)


def compute_score(predict: str, label: list, category: str = None) -> float:
    if category == "Individual-MEM":  # open-ended
        predict = predict.lower()
        if len(predict) == 0:
            return None
        score = 0
        for keyword in label:
            score += 1 if keyword.lower() in predict else 0
        return score / len(label)
    else:
        answers = format_predict(predict)
        if len(answers) == 0:
            return None
        if len(label) == 1:  # single-choice
            return 1 if answers[0] == label[0] else 0
        # multi-choices
        for answer in answers:
            if answer not in label:
                return 0
        return len(set(answers)) / len(set(label))


def format_predict(predict: str):
    if predict is None:
        return None
    answer = []
    matches = re.findall(r"(\b|\W+|^|[\u4e00-\u9fa5]+|(?<=[A-D]))([A-H])(\b|(?=[A-D])|$|\W+|[\u4e00-\u9fa5]+)", predict)
    for match in matches:
        if match[1] not in answer:
            answer.append(match[1])
    return answer

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    # Common fix for decoder-only models that don't define pad token (e.g., GPT-2) [web:16].
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return model, tokenizer, device



@torch.inference_mode()
def some_api_call_fn(prompt: str, model, tokenizer, device,
                     max_new_tokens: int = 256,
                     temperature: float = 0.7):

    CHOICES = list("ABCDEFGH")

    # Token ids for each choice; include both "A" and " A" to handle whitespace-prefixed tokens.
    allowed = set()
    for c in CHOICES:
        for s in (c, " " + c, "\n" + c):
            ids = tokenizer.encode(s, add_special_tokens=False)
            if len(ids) == 1:
                allowed.add(ids[0])
    allowed = sorted(allowed)

    def only_choices(batch_id, input_ids):
        return allowed  # always allow only these next tokens [web:1][web:127]


    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False,
        prefix_allowed_tokens_fn=only_choices,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local path or HF repo id, e.g. ./my_model or openai-community/gpt2")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model)

    # dataset = SocialBenchDataset("../SocialBench/data/self_awareness.json")
    
    # dataset = SocialBenchDataset("../SocialBench/data/emotional_perception.json")

    # dataset = SocialBenchDataset("../SocialBench/data/conversation_memory.json")
    dataset = SocialBenchDataset("../SocialBench/data/social_preference.json")

    sum_score = 0
    for data in tqdm(dataset, desc="Scoring", total=len(dataset)):
        model_output = some_api_call_fn(
            data["prompt"], model, tokenizer, device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        score = compute_score(model_output, data["label"], data["meta"]["category"])
        sum_score += score
        # print("prompt:\n", data['prompt'])
        # print("output ",model_output)
        # print("label ",data["label"])
        # break

    print(sum_score)

# 588/2456 self_awareness
# 174/1209 emotional_perception
# 501/1916 social_preference
# conversation_memory has different format with current SocialBenchDataset