import requests
import json

# url = "https://beam.slai.io/wy7re"
url = "https://beam.slai.io/6vfl3"
auth = "Basic ZjY1YWYyYWZhYzVkNDU5YmYyMjJiNThkNGY0MTk5NDc6NzNiNTk4M2E3Y2RiM2Y4NWQ1ODljZjM4YzRhZTRmZjc"

models = [
    # "nlpie/tiny-clinicalbert",
    # "nlpie/tiny-biobert",
    # "nlpie/clinical-mobilebert",
    # "nlpie/bio-mobilebert",
    # "mnaylor/psychbert-cased",
    "mental/mental-bert-base-uncased",
    "mental/mental-roberta-base",
]

for model in models:
  payload = {
      "dataset": "huggingface_reddit",
      "model_name": model,
      "label_count": 5,
      "batch_size": 4,
  }

  headers = {
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate",
    "Authorization": "Basic XXXXXXXXXXXXXX",
    "Connection": "keep-alive",
    "Content-Type": "application/json"
  }

  response = requests.request(
      "POST", url, headers=headers, data=json.dumps(payload)
  )

  print(payload)