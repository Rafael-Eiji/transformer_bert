import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForMaskedLM


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
folder = "/home/rafaelsakai/Documentos/Projetos/brazilian-legal-text-bert-main/tokenizer"
model = AutoModelForMaskedLM.from_pretrained(folder)
tokenizer = AutoTokenizer.from_pretrained(folder)

pipe = pipeline("fill-mask", model=model, tokenizer=tokenizer)

result = pipe('cadeir[MASK] azul')

print(result)