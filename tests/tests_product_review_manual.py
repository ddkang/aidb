import os
import pandas as pd
import requests

huggingface_key = "<your hugging face key>"


def get_reviews_data():
  dirname = os.path.dirname(__file__)
  review_file = os.path.join(dirname, 'data/amazon/blobs00.csv')
  return pd.read_csv(review_file)


def get_sentiment_from_hugging_face(review):
  hugging_face_model_url = f'https://api-inference.huggingface.co/models/LiYuan/amazon-review-sentiment-analysis',
  headers = {
    'Content-Type': 'application/json; charset=utf-8',
    'Authorization': f'Bearer {huggingface_key}',
  }
  input_params = {"inputs": [review], "options": {"wait_for_model": True}}
  r = requests.get(hugging_face_model_url, params=input_params, headers=headers)
  result = r.json()[0]
  return pd.DataFrame(result)


if __name__ == '__main__':
  review_data = get_reviews_data()
  results = pd.DataFrame()
  for i, row in review_data.iterrows():
    sentiment = get_sentiment_from_hugging_face(row["review"])
    results = pd.concat([results, sentiment])

  five_star_rows = results[results["label"] == "5 stars"]
  avg_score = five_star_rows["score"].mean()
  print("Result = ", avg_score)
