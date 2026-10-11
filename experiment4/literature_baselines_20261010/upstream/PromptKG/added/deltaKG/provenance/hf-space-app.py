import gradio as gr
from collections import defaultdict
from transformers import BertTokenizer, BertForMaskedLM
import jsonlines
import torch
from src.modeling_bert import EXBertForMaskedLM
from higher.patch import monkeypatch as make_functional

### load KGE model
edit_origin_model = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path="ChancesYuan/KGEditor_Edit_Test")
edit_ex_model = EXBertForMaskedLM.from_pretrained(pretrained_model_name_or_path="ChancesYuan/KGEditor_Edit_Test")

edit_learner = torch.load("./learner_checkpoint/edit/learner_params.pt", map_location=torch.device('cpu'))
add_learner = torch.load("./learner_checkpoint/add/learner_params.pt", map_location=torch.device('cpu'))

add_origin_model = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path="ChancesYuan/KGEditor_Add_Test")
add_ex_model = EXBertForMaskedLM.from_pretrained(pretrained_model_name_or_path="ChancesYuan/KGEditor_Add_Test")

### init inputs
ent_name2id = defaultdict(str)
id2ent_name = defaultdict(str)
rel_name2id = defaultdict(str)
id2ent_text = defaultdict(str)
id2rel_text = defaultdict(str)

### init tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
add_tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path='zjunlp/KGEditor', subfolder="E-FB15k237")

def init_triple_input():
    global ent2token
    global ent2id
    global id2ent
    global rel2token
    global rel2id

    with open("./dataset/fb15k237/relations.txt", "r") as f:
        lines = f.readlines()
        relations = []
        for line in lines:
            relations.append(line.strip().split('\t')[0])

        rel2token = {ent: f"[RELATION_{i}]" for i, ent in enumerate(relations)}
        
    with open("./dataset/fb15k237/entity2text.txt", "r") as f:
        for line in f.readlines():
            id, name = line.rstrip('\n').split('\t')
            ent_name2id[name] = id
            id2ent_name[id] = name

    with open("./dataset/fb15k237/relation2text.txt", "r") as f:
        for line in f.readlines():
            id, name = line.rstrip('\n').split('\t')
            rel_name2id[name] = id
            id2rel_text[id] = name

    with open("./dataset/fb15k237/entity2textlong.txt", "r") as f:
        for line in f.readlines():
            id, text = line.rstrip('\n').split('\t')
            id2ent_text[id] = text.replace("\\n", " ").replace("\\", "")

        entities = list(id2ent_text.keys())
        ent2token = {ent: f"[ENTITY_{i}]" for i, ent in enumerate(entities)}
        ent2id = {ent: i for i, ent in enumerate(entities)}
        id2ent = {i: ent for i, ent in enumerate(entities)}

    rel2id = {
        w: i + len(entities)
        for i, w in enumerate(rel2token.keys())
    }

def solve(triple, alter_label, edit_task):
    print(triple, alter_label)
    h, r, t = triple.split("|")
    if h == "[MASK]":
        text_a = "[MASK]"
        text_b = id2rel_text[r] + " " + rel2token[r]
        text_c = ent2token[ent_name2id[t]] + " " + id2ent_text[ent_name2id[t]]
        replace_token = [rel2id[r], ent2id[ent_name2id[t]]]
    else:
        text_a = ent2token[ent_name2id[h]]
        text_b = id2rel_text[r] + " " + rel2token[r]
        text_c = "[MASK]" + " " + id2ent_text[ent_name2id[h]]
        replace_token = [ent2id[ent_name2id[h]], rel2id[r]]

    if text_a == "[MASK]":
        input_text_a = tokenizer.sep_token.join(["[MASK]", id2rel_text[r] + "[PAD]"])
        input_text_b = "[PAD]" + " " + id2ent_text[ent_name2id[t]]
    else:
        input_text_a = "[PAD] "
        input_text_b = tokenizer.sep_token.join([id2rel_text[r] + "[PAD]", "[MASK]" + " " + id2ent_text[ent_name2id[h]]])

    inputs = tokenizer(
        f"{text_a} [SEP] {text_b} [SEP] {text_c}",
        truncation="longest_first",
        max_length=64,
        padding="longest",
        add_special_tokens=True,
    )

    edit_inputs = tokenizer(
        input_text_a,
        input_text_b,
        truncation="longest_first",
        max_length=64,
        padding="longest",
        add_special_tokens=True,
    )

    inputs = {
        "input_ids": torch.tensor(inputs["input_ids"]).unsqueeze(dim=0),
        "attention_mask": torch.tensor(inputs["attention_mask"]).unsqueeze(dim=0),
        "token_type_ids": torch.tensor(inputs["token_type_ids"]).unsqueeze(dim=0)
    }

    edit_inputs = {
        "input_ids": torch.tensor(edit_inputs["input_ids"]).unsqueeze(dim=0),
        "attention_mask": torch.tensor(edit_inputs["attention_mask"]).unsqueeze(dim=0),
        "token_type_ids": torch.tensor(edit_inputs["token_type_ids"]).unsqueeze(dim=0)
    }

    _, mask_idx = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)
    logits = edit_origin_model(**inputs).logits[:, :, 30522:45473].squeeze() if edit_task else add_origin_model(**inputs).logits[:, :, 30522:45473].squeeze()
    logits = logits[mask_idx, :]

    ### origin output
    _, origin_entity_order = torch.sort(logits, dim=1, descending=True)
    origin_entity_order = origin_entity_order.squeeze(dim=0)
    origin_top3 = [id2ent_name[id2ent[origin_entity_order[i].item()]] for i in range(3)]

    origin_label = origin_top3[0] if edit_task else alter_label

    cond_inputs_text = "{} >> {} || {}".format(
        add_tokenizer.added_tokens_decoder[ent2id[ent_name2id[origin_label]] + len(tokenizer)],
        add_tokenizer.added_tokens_decoder[ent2id[ent_name2id[alter_label]] + len(tokenizer)],
        input_text_a + input_text_b
    )

    cond_inputs = tokenizer(
        cond_inputs_text,
        truncation=True,
        max_length=64,
        padding="max_length",
        add_special_tokens=True,
    )

    cond_inputs = {
        "input_ids": torch.tensor(cond_inputs["input_ids"]).unsqueeze(dim=0),
        "attention_mask": torch.tensor(cond_inputs["attention_mask"]).unsqueeze(dim=0),
        "token_type_ids": torch.tensor(cond_inputs["token_type_ids"]).unsqueeze(dim=0)
    }

    flag = 0
    for idx, i in enumerate(edit_inputs["input_ids"][0, :].tolist()):
        if i == tokenizer.pad_token_id and flag == 0:
            edit_inputs["input_ids"][0, idx] = replace_token[0] + 30522
            flag = 1
        elif i == tokenizer.pad_token_id and flag != 0:
            edit_inputs["input_ids"][0, idx] = replace_token[1] + 30522

    return inputs, cond_inputs, edit_inputs, origin_top3

def get_logits_orig_params_dict(inputs, cond_inputs, alter_label, ex_model, learner):
    with torch.enable_grad():
        logits = ex_model.eval()(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        ).logits

        input_ids = inputs['input_ids']
        _, mask_idx = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)
        mask_logits = logits[:, mask_idx, 30522:45473].squeeze(dim=0)
        
        grads = torch.autograd.grad(
            # cross_entropy
            torch.nn.functional.cross_entropy(
                mask_logits[-1:, :],
                torch.tensor([alter_label]),
                reduction="none",
            ).mean(-1),
            ex_model.parameters(),
        )

    grads = {
        name: grad
        for (name, _), grad in zip(ex_model.named_parameters(), grads)
    }

    params_dict = learner(
        cond_inputs["input_ids"][-1:],
        cond_inputs["attention_mask"][-1:],
        grads=grads,
    )

    return params_dict

def edit_process(edit_input, alter_label):
    try:
        _, cond_inputs, edit_inputs, origin_top3 = solve(edit_input, alter_label, edit_task=True)
    except KeyError:
        return "The entity or relationship you entered is not in the vocabulary. Please check it carefully.", ""

    ### edit output
    fmodel = make_functional(edit_ex_model).eval()
    params_dict = get_logits_orig_params_dict(edit_inputs, cond_inputs, ent2id[ent_name2id[alter_label]], edit_ex_model, edit_learner)
    edit_logits = fmodel(
        input_ids=edit_inputs["input_ids"],
        attention_mask=edit_inputs["attention_mask"],
        # add delta theta
        params=[
            params_dict.get(n, 0) + p
            for n, p in edit_ex_model.named_parameters()
        ],
    ).logits[:, :, 30522:45473].squeeze()

    _, mask_idx = (edit_inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)
    edit_logits = edit_logits[mask_idx, :]
    _, edit_entity_order = torch.sort(edit_logits, dim=1, descending=True)
    edit_entity_order = edit_entity_order.squeeze(dim=0)
    edit_top3 = [id2ent_name[id2ent[edit_entity_order[i].item()]] for i in range(3)]

    return "\n".join(origin_top3), "\n".join(edit_top3)

def add_process(edit_input, alter_label):
    try:
        _, cond_inputs, add_inputs, origin_top3 = solve(edit_input, alter_label, edit_task=False)
    except:
        return "The entity or relationship you entered is not in the vocabulary. Please check it carefully.", ""

    ### add output
    fmodel = make_functional(add_ex_model).eval()
    params_dict = get_logits_orig_params_dict(add_inputs, cond_inputs, ent2id[ent_name2id[alter_label]], add_ex_model, add_learner)
    add_logits = fmodel(
        input_ids=add_inputs["input_ids"],
        attention_mask=add_inputs["attention_mask"],
        # add delta theta
        params=[
            params_dict.get(n, 0) + p
            for n, p in add_ex_model.named_parameters()
        ],
    ).logits[:, :, 30522:45473].squeeze()
    
    _, mask_idx = (add_inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)
    add_logits = add_logits[mask_idx, :]
    _, add_entity_order = torch.sort(add_logits, dim=1, descending=True)
    add_entity_order = add_entity_order.squeeze(dim=0)
    add_top3 = [id2ent_name[id2ent[add_entity_order[i].item()]] for i in range(3)]

    return "\n".join(origin_top3), "\n".join(add_top3)


with gr.Blocks() as demo:
    init_triple_input()
    gr.Markdown("# KGE Editing")

    # 多个tab
    with gr.Tabs():
        with gr.TabItem("E-FB15k237"):
            with gr.Row():
                with gr.Column():
                    edit_input = gr.Textbox(label="Input", lines=1, placeholder=" Please enter in the format of: [MASK]|rel|tail or head|rel|[MASK].")

                    alter_label = gr.Textbox(label="Alter Entity", lines=1, placeholder="Entity Name")    
                    edit_button = gr.Button("Edit")

                with gr.Column():
                    origin_output = gr.Textbox(label="Before Edit", lines=3, placeholder="")
                    edit_output = gr.Textbox(label="After Edit", lines=3, placeholder="")
       
            gr.Examples(
                examples=[["[MASK]|/people/person/profession|Jack Black", "Kellie Martin"], 
                          ["[MASK]|/people/person/nationality|United States of America", "Mark Mothersbaugh"],
                          ["[MASK]|/people/person/gender|Male", "Iggy Pop"],
                          ["Rachel Weisz|/people/person/nationality|[MASK]", "J.J. Abrams"],
                          ["Jeff Goldblum|/people/person/spouse_s./people/marriage/type_of_union|[MASK]", "Sydney Pollack"],
                          ],
                inputs=[edit_input, alter_label],
                outputs=[origin_output, edit_output],
                fn=edit_process,
                cache_examples=True,
            )

        with gr.TabItem("A-FB15k237"):
            with gr.Row():
                with gr.Column():
                    add_input = gr.Textbox(label="Input", lines=1, placeholder="Brand new triple input")

                    inductive_entity = gr.Textbox(label="Inductive Entity", lines=1, placeholder="Entity Name")
                    add_button = gr.Button("Add")

                with gr.Column():
                    add_origin_output = gr.Textbox(label="Origin Results", lines=3, placeholder="")
                    add_output = gr.Textbox(label="Add Results", lines=3, placeholder="")

            gr.Examples(
                examples=[["Jane Wyman|/people/person/places_lived./people/place_lived/location|[MASK]", "Palm Springs"], 
                          ["Darryl F. Zanuck|/people/deceased_person/place_of_death|[MASK]", "Palm Springs"],
                          ["[MASK]|/location/location/contains|Antigua and Barbuda", "Americas"],
                          ["Hard rock|/music/genre/artists|[MASK]", "Social Distortion"],
                          ["[MASK]|/people/person/nationality|United States of America", "Serj Tankian"]
                          ],
                inputs=[add_input, inductive_entity],
                outputs=[add_origin_output, add_output],
                fn=add_process,
                cache_examples=True,
            )

    edit_button.click(fn=edit_process, inputs=[edit_input, alter_label], outputs=[origin_output, edit_output])
    add_button.click(fn=add_process, inputs=[add_input, inductive_entity], outputs=[add_origin_output, add_output])

demo.launch()