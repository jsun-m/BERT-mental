### Deploy to Beam for Async Training

1. Make sure you have Beam installed https://beam.cloud
2. Inside `constants.py` update to incluef your Huggingface api key
3. `beam deploy app.py`
4. After it finishes deploying, use webhook_start.py to call your model
   1. Make sure to change the `url` and `auth` to be your `url` and `auth`. You can find these on the Beam Dashboard `Call API` button
