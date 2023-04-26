import requests
import json

url = "https://beam.slai.io/wz69m" # REPLACE WITH YOUR WEBHOOK URL


models = [
  "nlpie/tiny-biobert",
  "nlpie/tiny-clinicalbert",
  "nlpie/clinical-mobilebert",
  "nlpie/bio-mobilebert",
  "mnaylor/psychbert-cased",
  "mental/mental-bert-base-uncased",
  "mental/mental-roberta-base",
  "smallbenchnlp/roberta-small"
]

datasets = [
  {
    "dataset": "redditMH",
    "label_count": 2,
    "batch_size": 8,
  }
]

for dataset in datasets:
  for model in models:
    payload = {
      **dataset,
      "model_name": model,
    }

    headers = {
      "Accept": "*/*",
      "Accept-Encoding": "gzip, deflate",
      "Authorization": "Basic YOUR_API_TOKEN", # REPLACE WITH YOUR BEAM API TOKEN
      "Connection": "keep-alive",
      "Content-Type": "application/json"
    }

    response = requests.request(
        "POST", url, headers=headers, data=json.dumps(payload)
    )

    print(response.content)