# Example Inference Services 

## Remote API

You will need to specify the mapping from the AIDB input columns to the JSON input format and the mapping from the JSON output format to the AIDB output columns. In order to nest fields in the JSON, you will need to use a tuple. In the case of a flat JSON, you can simply use strings as keys.

As an example, if you have AIDB input columns like this:
| model | role | content | n |
| --- | --- | --- | --- |
| gpt-3.5-turbo | user | Do you know Google Cloud Vision API? | 2 |

and you want to use the [OpenAI Chat API](https://platform.openai.com/docs/api-reference/chat) which want JSON input format like this:
```json 
{
  "model": "gpt-3.5-turbo",
  "messages": [
    {
      "role": "user",
      "content": "Do you know Google Cloud Vision API?"
    }
  ],
  "n": 2,
}
```

You will need to specify the mapping from the AIDB input columns to the JSON input format like this:
```python
from aidb.config.config_types import AIDBListType
map_input_to_request = {
  'model': ('model',),
  'role': ('messages', AIDBListType(), 'role'),
  'content': ('messages', AIDBListType(), 'content'),
  'n': 'n'
}
```

You can import `AIDBListType` to ease your conversion.

For input to request conversion, you can either provide explicit list index if your JSON request contains list, or use `AIDBListType` to span your input column to a JSON list of arbitrary length (length of the list is your batch size) inside your JSON request. Each column_key: json_key map should contain at most 1 `AIDBListType` key.

For response to output conversion, you can either provide explicit list index to retrieve the item from the JSON response, or use `AIDBListType` to span list response into multiple rows in your output dataframe.

Please do not use `AIDBListType` for lists of known length such as coordinates. Use explicit numerical index instead.

If an input column or an output JSON attribute is not inside the map keys, that column/attribute will be ignored.

You can optionally write your own `HTTPInferenceService.convert_input_to_request` and `HTTPInferenceService.convert_response_to_output` methods if you want to do more complicated conversion.

You can optionally move some arguments to `default_args` during initialization. For example, if you want to use the same `model` for all requests to OpenAI Chat API, you can do:
```python
openai_text = OpenAIText(
  token=OPENAI_KEY,
  default_args={'model': 'gpt-3.5-turbo'},
  columns_to_input_keys=map_input_to_request,
  response_keys_to_columns=map_response_to_output)
```

If you do not provide token during initialization, AIDB will find token from environment variable. The corresponding environment variables for each different services are listed below.
| OpenAI | HuggingFace | Google |
| --- | --- | --- |
| `OPENAI_API_KEY` | `HF_API_KEY` | `gcloud auth application-default print-access-token` |

### OpenAI

#### [Chat](https://platform.openai.com/docs/api-reference/chat)

Please refer to [Request body](https://platform.openai.com/docs/api-reference/chat/create) and [The chat completion object](https://platform.openai.com/docs/api-reference/chat/object) for dataframe <-> json key map. OpenAI will only respond to one `messages`, so please only input 1 column. 

Example usage:
```python
import pandas as pd
from aidb.inference.examples.openai_inference_service import OpenAIText
from aidb.config.config_types import AIDBListType
openai_text = OpenAIText(
  token=OPENAI_KEY,
  default_args={
    "model": "gpt-3.5-turbo",
    "n": 2,
  },
  columns_to_input_keys={
    'role': ('messages', AIDBListType(), 'role'),
    'content': ('messages', AIDBListType(), 'content')
  },
  response_keys_to_columns={
    'id': 'id',
    'created': 'created',
    'model': 'model',
    ('choices', AIDBListType(), 'message', 'content'): 'response',
  })
openai_text_response_pd = openai_text.infer_one(pd.Series({
  "role": "user",
  "content": "Do you know Google Cloud Vision API?"
}))
openai_text_response_pd
```

Response:
|	|id|	created|	model|	response|
|---|---|---|---|---|
|0|	chatcmpl-8Ci3StQkwrZVzeksQ7NdLcwzx7Hza	|1698039974|	gpt-3.5-turbo-0613|	Yes, I am familiar with Google Cloud Vision API. It is a set of computer vision capabilities offered by Google Cloud that allows developers to integrate vision detection features into their applications. The API enables tasks like image labeling, face detection, object recognition, text extraction, and more. It uses machine learning models to analyze images and provide accurate results.|
|1|	chatcmpl-8Ci3StQkwrZVzeksQ7NdLcwzx7Hza	|1698039974|	gpt-3.5-turbo-0613|	Yes, I am familiar with Google Cloud Vision API. It is a machine learning-based image analysis tool provided by Google Cloud Platform. It allows developers to integrate image recognition, labeling, face detection, OCR (optical character recognition), and various other image-related functionalities into their applications.|


#### [Images](https://platform.openai.com/docs/api-reference/images)

We support [Create image](https://platform.openai.com/docs/api-reference/images/create), [Create image edit](https://platform.openai.com/docs/api-reference/images/createEdit) and [Create image variation](https://platform.openai.com/docs/api-reference/images/createVariation). Please refer to the corresponding request body for dataframe <-> json key map.

Example usage:
```python
import pandas as pd
from aidb.inference.examples.openai_inference_service import OpenAIImage
openai_image = OpenAIImage(
  token=OPENAI_KEY,
  default_args={
    "n": 3,
    "size": "512x512"
  },
  columns_to_input_keys={'prompt': 'prompt'},
  response_keys_to_columns={
    'created': 'created',
    ('data', AIDBListType(), 'url'): 'url',
  })
openai_image_response_pd = openai_image.infer_one(pd.Series({"prompt": "Miku smiling"}))
openai_image_response_pd
```

Response:
|	|created|	url|
|---|---|---|
|0|	1698040308|	https://oaidalleapiprodscus.blob.core.windows.net/private/org-3bMaInw6MjKWFNTMv8GxqKri/user-wC1FjJwmLQdW3wa3GsnJQOH4/img-6pj3bL19ibc4VSS2r86Rt6l1.png?st=2023-10-23T04%3A51%3A48Z&se=2023-10-23T06%3A51%3A48Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-10-22T10%3A05%3A00Z&ske=2023-10-23T10%3A05%3A00Z&sks=b&skv=2021-08-06&sig=zht7nDEoyIzCcJpQ/D/CaolZkda/DX/5qTLJ0bWHc3U%3D |
|1|	1698040308|	https://oaidalleapiprodscus.blob.core.windows.net/private/org-3bMaInw6MjKWFNTMv8GxqKri/user-wC1FjJwmLQdW3wa3GsnJQOH4/img-9dGtMNXrQ2BUg1MtTEpBRoJL.png?st=2023-10-23T04%3A51%3A48Z&se=2023-10-23T06%3A51%3A48Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-10-22T10%3A05%3A00Z&ske=2023-10-23T10%3A05%3A00Z&sks=b&skv=2021-08-06&sig=0MvEbVFN1CxXkLk7xZwi7k6wOQEJuHVVQS00e8tKCXA%3D |
|2|	1698040308|	https://oaidalleapiprodscus.blob.core.windows.net/private/org-3bMaInw6MjKWFNTMv8GxqKri/user-wC1FjJwmLQdW3wa3GsnJQOH4/img-Q6GpUcb1Mu9rpx17IgvL6LXK.png?st=2023-10-23T04%3A51%3A48Z&se=2023-10-23T06%3A51%3A48Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-10-22T10%3A05%3A00Z&ske=2023-10-23T10%3A05%3A00Z&sks=b&skv=2021-08-06&sig=1ejp6yEPfioQUQIkYAH7ho05ZXbUH5jMOEAKh0UErPw%3D |


### HuggingFace

#### [NLP](https://huggingface.co/docs/api-inference/detailed_parameters#natural-language-processing)

All HuggingFace NLP tasks are supported, except table analysis API (because it contains nested JSON). Please refer to [doc](https://huggingface.co/docs/api-inference/detailed_parameters#natural-language-processing) for detailed meaning of each parameters and dataframe <-> json key map.

Example usage:
```python
# Fill Mask task
import pandas as pd
from aidb.inference.examples.huggingface_inference_service import HuggingFaceNLP
from aidb.config.config_types import AIDBListType
hf_nlp = HuggingFaceNLP(
  token=HF_KEY,
  default_args={"options": {"wait_for_model": True}},
  columns_to_input_keys={'inputs': 'inputs'},
  response_keys_to_columns={(AIDBListType(), 'sequence'): 'sequence',
                            (AIDBListType(), 'score'): 'score',
                            (AIDBListType(), 'token'): 'token',
                            (AIDBListType(), 'token_str'): 'token_str'},
  model="bert-base-uncased")

hf_nlp_response_pd = hf_nlp.infer_one(pd.Series({"inputs": "The answer to the universe is [MASK]."}))
hf_nlp_response_pd
```

Response:
|	|sequence|	score|	token|	token_str|
|---|---|---|---|---|
|0|	the answer to the universe is no.|	0.169641|	2053|	no|
|1|	the answer to the universe is nothing.|	0.073448|	2498|	nothing|
|2|	the answer to the universe is yes.|	0.058032|	2748|	yes|
|3|	the answer to the universe is unknown.|	0.043958|	4242|	unknown|
|4|	the answer to the universe is simple.|	0.040157|	3722|	simple|

#### [CV](https://huggingface.co/docs/api-inference/detailed_parameters#computer-vision)

All HuggingFace CV tasks are supported and they only accept filename as input. You do not need to provide a key map from input to request JSON because request is a binary. The response JSON could be a list of indefinite length. Please refer to [doc](https://huggingface.co/docs/api-inference/detailed_parameters#computer-vision) for detailed meaning of each parameters and response json -> output dataframe map.

Example usage:
```python
# Object Detection task
from aidb.inference.examples.huggingface_inference_service import HuggingFaceVisionAudio
from aidb.config.config_types import AIDBListType

map_response_to_output = {
  (AIDBListType(), 'score'): 'score',
  (AIDBListType(), 'label'): 'label',
  (AIDBListType(), 'box', 'xmin'): 'box.xmin',
  (AIDBListType(), 'box', 'ymin'): 'box.ymin',
  (AIDBListType(), 'box', 'xmax'): 'box.xmax',
  (AIDBListType(), 'box', 'ymax'): 'box.ymax',
}

hf_cv = HuggingFaceVisionAudio(
  token=HF_KEY,
  default_args={"options": {"wait_for_model": True}},
  response_keys_to_columns=map_response_to_output,
  model="facebook/detr-resnet-50")
hf_cv_response_pd = hf_cv.infer_one(pd.Series({
    "filename": "/home/conrevo/图片/avatar.png" # key must be "filename"
}))
hf_cv_response_pd
```

Response:
|	|score|	label|	box.xmin|	box.ymin|	box.xmax|	box.ymax|
|---|---|---|---|---|---|---|
|0|	0.998632|	tie|	214|	254|	254|	426|
|1|	0.997547|	person|	50|	26|	511|	509|


### Google
We support [files.annotate](https://cloud.google.com/vision/docs/reference/rest/v1/files/annotate) and [images.annotate](https://cloud.google.com/vision/docs/reference/rest/v1/images/annotate). Please visit [AnnotateFileRequest](https://cloud.google.com/vision/docs/reference/rest/v1/AnnotateFileRequest) or [AnnotateImageRequest](https://cloud.google.com/vision/docs/reference/rest/v1/AnnotateImageRequest) for input dataframe -> request json map.

To convert response json to output dataframe, please refer to [BatchAnnotateFilesResponse](https://cloud.google.com/vision/docs/reference/rest/v1/BatchAnnotateFilesResponse) or [BatchAnnotateImagesResponse](https://cloud.google.com/vision/docs/reference/rest/v1/BatchAnnotateImagesResponse).

You must provide project id during initialization.

Example usage:
```python
import pandas as pd
from aidb.inference.examples.google_inference_service import GoogleVisionAnnotate

map_input_to_request = {
  'imageUri': ('requests', '0', 'image', 'source', 'imageUri'),
  'type': ('requests', '0', 'features', '0', 'type'),
  '0': ('requests',
    '0',
    'imageContext',
    'cropHintsParams',
    'aspectRatios',
    '0'),
  '1': ('requests',
    '0',
    'imageContext',
    'cropHintsParams',
    'aspectRatios',
    '1'),
  '2': ('requests',
    '0',
    'imageContext',
    'cropHintsParams',
    'aspectRatios',
    '2'),
  'parent': ('parent',)
}

map_response_to_output = <omit, too long>

google_cv = GoogleVisionAnnotate(
  token=GOOGLE_KEY,
  columns_to_input_keys=map_input_to_request,
  response_keys_to_columns=map_response_to_output,
  project_id='<project-id>')
google_cv_response_pd = google_cv.infer_one(google_cv_request_pd)
google_cv_response_pd
```

Input Series:
```
imageUri    https://upload.wikimedia.org/wikipedia/commons/5/56/Donald_Trump_official_portrait.jpg
type                                           FACE_DETECTION
0                                                         0.8
1                                                           1
2                                                         1.2
parent                        projects/coral-sanctuary-400802
dtype: object
```

Response:
\<omit, too long\>

The way to obtain a Google API key is tricky. Please
1. initiate a project in [Google cloud console](ttps://console.cloud.google.com/welcome/new)
1. install [Google Cloud CLI](https://cloud.google.com/sdk/docs/install)
1. run
    ```bash
    gcloud init
    gcloud auth application-default print-access-token
    ```
    in your terminal. You may need additional steps as prompted.

## Local inference

We provide example for running [GroundingDINO](https://github.com/continue-revolution/GroundingDINO) locally. Note that the [original repo](https://github.com/IDEA-Research/GroundingDINO) does not support batch inference, but this forked repo does. That said, you should make sure that your input images have the same shape if you want to run batch inference.

Since the acceptable input parameters and output formats for different inference API differ drastically, you will need to inherit `CachedInferenceService` and write your own input and output conversion inside `infer_one` and `infer_batch`.

Before running this inference, please
```bash
git clone https://github.com/continue-revolution/GroundingDINO
cd GroundingDINO
pip install -e .
```

Example usage:
```python
from aidb.inference.example.pytorch_local_inference import PyTorchLocalObjectDetection
groundingdino = PyTorchLocalObjectDetection(
  name='groundingdino',
  model_config_path='path/to/GroundingDINO_SwinT_OGC.py',
  model_checkpoint_path='path/to/groundingdino_swint_ogc.pth',
  caption='your caption',
  use_batch=True,
  batch_size=2)
outputs = groundingdino.infer_batch(inputs)
```

Input dataframe:
|  | image |
|---|---|
| 0 | /path/to/image1.jpg |
| 1 | /path/to/image2.jpg |
| 2 | /path/to/image3.jpg |

Output dataframe:
||image|	min_x|	min_y|	max_x|	max_y|	confidence|
|---|---|---|---|---|---|---|
|0|	/path/to/image1.jpg|	438.916321|	5.441467	|1676.187744|	1076.021484|	0.865207|
|1|	/path/to/iamge2.jpg|	632.179443|	111.547821|1376.579102|	1075.546387|	0.787911|
|2|	/path/to/image3.jpg|	538.856323|	40.356171	|1008.120850|	681.444580|	0.505632|
|3|	/path/to/image3.jpg|	539.465942|	40.187408	|1066.772583|	881.556763|	0.402371|

