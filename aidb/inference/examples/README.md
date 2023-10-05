# Example Inference Services 

## OpenAI 

### [Chat](https://platform.openai.com/docs/api-reference/chat)

For input creation, please refer to [Request body](https://platform.openai.com/docs/api-reference/chat/create). OpenAI will only respond to one `messages`, so please only input 1 column.
> TODO: Input is a JSON object with keys inside keys. What's the best way of representing it? Currently I just assume user input columns have JSON objects as values. A better way might be to have something similar to PyTorch state dict, such as `messages.0.role`.

For output, it will contain columns `index`, `finish_reason`, `created`, `id`, `model`, `role`, `content`. Please refer to [The chat completion object](https://platform.openai.com/docs/api-reference/chat/object).
> TODO: Users probably do not need all the columns. We should probably consider having a way to let users specify which columns they want to keep. However, if there are a lot of columns, they might have to write a lot of code to specify.

Example usage:
```python
import pandas as pd
from aidb.inference.examples.openai_inference_service import OpenAIText
openai_text_request_dict = {
  "model": "gpt-3.5-turbo",
  "messages": [
    {
      "role": "user",
      "content": "Do you know Google Cloud Vision API?"
    }
  ],
  "n": 3,
}
openai_text_request_pd = pd.Series(openai_text_request_dict)
openai_text = OpenAIText(OPENAI_KEY)
openai_text_response_pd = openai_text.infer_one(openai_text_request_pd)
```

Input Series:
```
model                                           gpt-3.5-turbo
messages    [{'role': 'user', 'content': 'Do you know Goog...
n                                                           3
dtype: object
```

Response:
```
	index	finish_reason	created	id	model	role	content
0	0	stop	1696485393	chatcmpl-86BdZSZgmJKol8D5izIOQKqZZ8iWJ	gpt-3.5-turbo-0613	assistant	Yes, Google Cloud Vision API is an artificial ...
1	1	stop	1696485393	chatcmpl-86BdZSZgmJKol8D5izIOQKqZZ8iWJ	gpt-3.5-turbo-0613	assistant	Yes, I am familiar with the Google Cloud Visio...
2	2	stop	1696485393	chatcmpl-86BdZSZgmJKol8D5izIOQKqZZ8iWJ	gpt-3.5-turbo-0613	assistant	Yes, I am familiar with Google Cloud Vision AP...
```

### [Images](https://platform.openai.com/docs/api-reference/images)

We support [Create image](https://platform.openai.com/docs/api-reference/images/create), [Create image edit](https://platform.openai.com/docs/api-reference/images/createEdit) and [Create image variation](https://platform.openai.com/docs/api-reference/images/createVariation). Please refer to the corresponding request body for input creation.
> TODO: Test `Create image edit` and `Create image variation`.

Output will have two columns `created` and `data`.

Example usage:
```python
import pandas as pd
from aidb.inference.examples.openai_inference_service import OpenAIImage
openai_image_request_dict ={
    "prompt": "Full body Miku smiling",
    "n": 2,
    "size": "512x512"
  }
openai_image_request_pd = pd.Series(openai_image_request_dict)
print(openai_image_request_pd)
openai_image = OpenAIImage(OPENAI_KEY)
openai_image_response_pd = openai_image.infer_one(openai_image_request_pd)
openai_image_response_pd
```

Input Series:
```
prompt    Full body Miku smiling
n                              2
size                     512x512
dtype: object
```

Response:
```
    created data
0	1696490931	https://oaidalleapiprodscus.blob.core.windows....
1	1696490931	https://oaidalleapiprodscus.blob.core.windows....
```

### [Audio](https://platform.openai.com/docs/api-reference/audio)
> TODO: Test audio.
