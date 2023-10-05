# Example Inference Services 

## [OpenAI API]()

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
|	|index|	finish_reason|	created|	id|	model|	role|	content|
|---|---|---|---|---|---|---|---|
|0|	0|	stop|	1696485393|	chatcmpl-86BdZSZgmJKol8D5izIOQKqZZ8iWJ|	gpt-3.5-turbo-0613|	assistant|	Yes, I am familiar with Google Cloud Vision API. It is a service provided by Google Cloud Platform that allows developers to easily integrate image analysis and recognition capabilities into their applications. The API enables tasks such as labeling images, detecting objects and faces, reading printed and handwritten text, and identifying similar images. It uses machine learning models developed by Google to deliver accurate results.|
|1|	1|	stop|	1696485393|	chatcmpl-86BdZSZgmJKol8D5izIOQKqZZ8iWJ|	gpt-3.5-turbo-0613|	assistant|	Yes, I am familiar with Google Cloud Vision API. It is a machine learning-based technology that allows developers to integrate image analysis features into their applications. It can perform tasks such as detecting objects, faces, and landmarks in images, as well as extracting text, sentiment analysis, and generating image descriptions.|
|2|	2|	stop|	1696485393|	chatcmpl-86BdZSZgmJKol8D5izIOQKqZZ8iWJ|	gpt-3.5-turbo-0613|	assistant|	Yes, I am familiar with Google Cloud Vision API. It is a cloud-based machine learning API offered by Google that allows developers to integrate computer vision capabilities into their applications. The API provides various pre-trained models that can perform tasks such as labeling images, detecting faces, objects, and landmarks, OCR (optical character recognition), explicit content detection, and more.|

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
openai_image = OpenAIImage(OPENAI_KEY)
openai_image_response_pd = openai_image.infer_one(openai_image_request_pd)
```

Input Series:
```
prompt    Full body Miku smiling
n                              2
size                     512x512
dtype: object
```

Response:
|  | created | data |
| --- | --- | --- |
|0|	1696526028|	https://oaidalleapiprodscus.blob.core.windows.net/private/org-3bMaInw6MjKWFNTMv8GxqKri/user-wC1FjJwmLQdW3wa3GsnJQOH4/img-FSG2IoVXnPDW9BZzmPFbpmbD.png?st=2023-10-05T16%3A13%3A48Z&se=2023-10-05T18%3A13%3A48Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-10-05T16%3A36%3A11Z&ske=2023-10-06T16%3A36%3A11Z&sks=b&skv=2021-08-06&sig=qxcS2z0NfNjkTZbekbDr5GW4Atxo8/alqS6%2BABqahIM%3D |
|1|	1696526028|	https://oaidalleapiprodscus.blob.core.windows.net/private/org-3bMaInw6MjKWFNTMv8GxqKri/user-wC1FjJwmLQdW3wa3GsnJQOH4/img-aUkigAi9MXCdC2qBu8A8sDy0.png?st=2023-10-05T16%3A13%3A48Z&se=2023-10-05T18%3A13%3A48Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-10-05T16%3A36%3A11Z&ske=2023-10-06T16%3A36%3A11Z&sks=b&skv=2021-08-06&sig=CU6te7nynHR7nplAs4D%2B3Yoa4rTRdE%2BUnvx8jsba%2BTk%3D |

### [Audio](https://platform.openai.com/docs/api-reference/audio)
> TODO: Test audio.

## HuggingFace

### [NLP](https://huggingface.co/docs/api-inference/detailed_parameters#natural-language-processing)

All HuggingFace NLP tasks are supported and they have the same input format. Your input should have 1 row and 1-3 columns `inputs`(required, a string), `parameters` (optional, a JSON object) and/or `options` (optional, a JSON object). Please refer to [doc](https://huggingface.co/docs/api-inference/detailed_parameters#natural-language-processing) for detailed meaning of each parameters. You should only provide one row, otherwise HuggingFace will throw an error.

Example usage:
```python
import pandas as pd
from aidb.inference.examples.huggingface_inference_service import HFNLP
hf_nlp_request_dict = {
  "inputs": "Stanford PhD student Lvmin Zhang has created"
}
hf_nlp_request_pd = pd.Series(hf_nlp_request_dict)
hf_nlp = HFNLP(HF_KEY, "gpt2")
hf_nlp_response_pd = hf_nlp.infer_one(hf_nlp_request_pd)
```

Input Series:
```
inputs    Stanford PhD student Lvmin Zhang has created
dtype: object
```

Response (this response is fake):
|  | generated_text |
| --- | --- |
| 0	| Stanford PhD student Lvmin Zhang has created an elegant, non-nucleic acid that provides a direct replacement for the polyisopentocine and tritunocine monomers found in the A1, A2 and B1 |

### [CV](https://huggingface.co/docs/api-inference/detailed_parameters#computer-vision)

All HuggingFace CV tasks are supported and they only accept filename as input. The output could be multiple columns and multiple rows, depending on tasks. Please refer to [doc](https://huggingface.co/docs/api-inference/detailed_parameters#computer-vision) for detailed meaning of each parameters.

Example usage:
```python
import pandas as pd
from aidb.inference.examples.huggingface_inference_service import HFVisionAudio
hf_cv_request_dict = {
  "filename": "/path/to/image.png"
}
hf_cv_request_pd = pd.Series(hf_cv_request_dict)
hf_cv = HFVisionAudio(HF_KEY, "facebook/detr-resnet-50")
hf_cv_response_pd = hf_cv.infer_one(hf_cv_request_pd)
```

Input Series:
```
filename    /path/to/image.png
dtype: object
```

Response:
|  | score | label | box |
| --- | --- | --- | --- |
|0|	0.998632|	tie	|{'xmin': 214, 'ymin': 254, 'xmax': 254, 'ymax': 426}|
|1|	0.997547|	person	|{'xmin': 50, 'ymin': 26, 'xmax': 511, 'ymax': 509}|
> TODO: box is a JSON object. There are probably better ways of representing it, but this varies between tasks.

### [Audio](https://huggingface.co/docs/api-inference/detailed_parameters#audio)
> TODO: Test audio.

## Google
We support [files.annotate](https://cloud.google.com/vision/docs/reference/rest/v1/files/annotate) and [images.annotate](https://cloud.google.com/vision/docs/reference/rest/v1/images/annotate). Please create corresponding [AnnotateFileRequest](https://cloud.google.com/vision/docs/reference/rest/v1/AnnotateFileRequest) or [AnnotateImageRequest](https://cloud.google.com/vision/docs/reference/rest/v1/AnnotateImageRequest) for input.

Output is a JSON object, in the same format as [BatchAnnotateFilesResponse](https://cloud.google.com/vision/docs/reference/rest/v1/BatchAnnotateFilesResponse) and [BatchAnnotateImagesResponse](https://cloud.google.com/vision/docs/reference/rest/v1/BatchAnnotateImagesResponse).
> TODO: I currently reserve output as JSON because they are TOO MUCH and I'm not sure what we/users want.

Example usage:
```python
import base64
import pandas as pd
from aidb.inference.examples.google_inference_service import GoogleVisionAnnotate
with open("/path/to/image.jpg", "rb") as f:
    image_bytes = f.read()
image_base64 = base64.b64encode(image_bytes).decode("utf-8")
google_cv_request_dict = {
  "image": image_base64,
  "features": [{
      "type": "IMAGE_PROPERTIES",
    }, {
      "type": "FACE_DETECTION"
    }
  ]
}
google_cv_request_pd = pd.Series(google_cv_request_dict)
google_cv = GoogleVisionAnnotate(GOOGLE_KEY, project_id="your_project_id")
google_cv_response_pd = google_cv.infer_one(google_cv_request_pd)
```

The way to obtain a Google API key is tricky. Please do
```bash
gloud init
gloud auth application-default print-access-token
```
in your terminal. You may need additional steps as prompted.
