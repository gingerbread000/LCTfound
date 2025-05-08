tokenizer = AutoTokenizer.from_pretrained("pretrained_w/models--bert-base-cased/snapshots/cd5ef92a9fb2f889e972770a36d4ed042daf221e")
def get_mod_prompt(tokenizer, mod_cls=0):
    raw_inputs = [
            "The format of input data is ct.",
            "The format of input data is t1 mri.",
            "The format of input data is t2 mri.",
        ]
    res = tokenizer(raw_inputs, padding="max_length", max_length=16, return_tensors="pt")
    text_embeding = res["input_ids"][mod_cls].view(1,-1,1)
    token_type_ids = res["token_type_ids"][mod_cls].view(1,-1,1)
    attention_mask = res["attention_mask"][mod_cls].view(1,-1,1)
    return text_embeding, attention_mask