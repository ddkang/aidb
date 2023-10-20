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
map_input_to_request = {
  'model': ('model',),
  'role': ('messages', '0', 'role'),
  'content': ('messages', '0', 'content'),
  'n': 'n'
}
```

If an input column or an output JSON attribute is not inside the map keys, that column/attribute will be ignored.

You can optionally move some arguments to `default_args` during initialization. For example, if you want to use the same `model` for all requests to OpenAI Chat API, you can do:
```python
default_args = {
  'model': 'gpt-3.5-turbo'
}

openai_text = OpenAIText(
  token=OPENAI_KEY,
  default_args=default_args,
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

map_input_to_request = {
  'model': ('model',),
  'role': ('messages', '0', 'role'),
  'content': ('messages', '0', 'content'),
  'n': ('n',)
}

map_response_to_output = {
  ('id',): 'id',
  ('object',): 'object',
  ('created',): 'created',
  ('model',): 'model',
  ('choices', '0', 'index'): 'choices.0.index',
  ('choices', '0', 'message', 'role'): 'choices.0.message.role',
  ('choices', '0', 'message', 'content'): 'choices.0.message.content',
  ('choices', '0', 'finish_reason'): 'choices.0.finish_reason',
  ('choices', '1', 'index'): 'choices.1.index',
  ('choices', '1', 'message', 'role'): 'choices.1.message.role',
  ('choices', '1', 'message', 'content'): 'choices.1.message.content',
  ('choices', '1', 'finish_reason'): 'choices.1.finish_reason'
}

openai_text = OpenAIText(
  token=OPENAI_KEY, 
  columns_to_input_keys=map_input_to_request, 
  response_keys_to_columns=map_response_to_output)
openai_text_response_pd = openai_text.infer_one(openai_text_request_pd)
openai_text_response_pd
```

Input:
```
model                             gpt-3.5-turbo
role                                       user
content    Do you know Google Cloud Vision API?
n                                             2
dtype: object
```

Response:
| | id | object | created | model | choices_0_index | choices_0_message_role | choices_0_message_content | choices_0_finish_reason | choices_1_index | choices_1_message_role | choices_1_message_content | choices_1_finish_reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0  | chatcmpl-88fEubFsO7GN2st9FWaLq64zKEQpI | chat.completion | 1697075840 | gpt-3.5-turbo-0613 | 0 | assistant | Yes, I am familiar with Google Cloud Vision API. It is a machine learning based image analysis tool provided by Google Cloud Platform. It allows developers to easily integrate image recognition, object detection, and image understanding capabilities into their applications by using pre-trained models. The API can be used to identify objects, faces, and text in images, as well as detect explicit content and perform image sentiment analysis. | stop | 1 | assistant | Yes, I am familiar with Google Cloud Vision API. It is a machine learning service provided by Google Cloud that allows users to analyze and understand the content of images. It can detect objects, faces, and text, as well as perform image sentiment analysis, logo detection, and image landmark recognition among other features. The API provides a RESTful interface for developers to integrate image analysis capabilities into their applications. | stop |


#### [Images](https://platform.openai.com/docs/api-reference/images)

We support [Create image](https://platform.openai.com/docs/api-reference/images/create), [Create image edit](https://platform.openai.com/docs/api-reference/images/createEdit) and [Create image variation](https://platform.openai.com/docs/api-reference/images/createVariation). Please refer to the corresponding request body for dataframe <-> json key map.

Example usage:
```python
import pandas as pd
from aidb.inference.examples.openai_inference_service import OpenAIImage

map_input_to_request = {
  'prompt': ('prompt',),
  'n': ('n',),
  'size': ('size',)
}

map_response_to_output = {
  ('created',): 'created',
  ('data', '0', 'url'): 'data.0.url',
  ('data', '1', 'url'): 'data.1.url'
}

openai_image = OpenAIImage(
  token=OPENAI_KEY,
  columns_to_input_keys=map_input_to_request,
  response_keys_to_columns=map_response_to_output)
openai_image_response_pd = openai_image.infer_one(openai_image_request_pd)
openai_image_response_pd
```

Input Series:
```
prompt    Miku smiling
n                    2
size           512x512
dtype: object
```

Response:
| |	created | data_0_url | data_1_url|
|---|---|---|---|
|0|	1697076329|	https://oaidalleapiprodscus.blob.core.windows.net/private/org-3bMaInw6MjKWFNTMv8GxqKri/user-wC1FjJwmLQdW3wa3GsnJQOH4/img-9ycIx6ZG6T68IYV12wXNlZhd.png?st=2023-10-12T01%3A05%3A29Z&se=2023-10-12T03%3A05%3A29Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-10-12T01%3A58%3A02Z&ske=2023-10-13T01%3A58%3A02Z&sks=b&skv=2021-08-06&sig=dyzFdHArfGMq%2B6hxHPQ5vhzQdpN3On5JkOag8cXmwXA%3D	| https://oaidalleapiprodscus.blob.core.windows.net/private/org-3bMaInw6MjKWFNTMv8GxqKri/user-wC1FjJwmLQdW3wa3GsnJQOH4/img-AjvmDySJANPAc4Huviconivv.png?st=2023-10-12T01%3A05%3A29Z&se=2023-10-12T03%3A05%3A29Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-10-12T01%3A58%3A02Z&ske=2023-10-13T01%3A58%3A02Z&sks=b&skv=2021-08-06&sig=oSB/BJM2u70ITy%2B6jjsbNHehBlYcQD9nzwotT9uQIjE%3D |


### HuggingFace

#### [NLP](https://huggingface.co/docs/api-inference/detailed_parameters#natural-language-processing)

All HuggingFace NLP tasks are supported. Please refer to [doc](https://huggingface.co/docs/api-inference/detailed_parameters#natural-language-processing) for detailed meaning of each parameters and dataframe <-> json key map.

Example usage:
```python
from aidb.inference.examples.huggingface_inference_service import HuggingFaceNLP
hf_nlp = HuggingFaceNLP(
  token=HF_KEY,
  columns_to_input_keys={'inputs': 'inputs'},
  response_keys_to_columns={('0', 'generated_text', ): 'generated_text'},
  model="gpt2")
hf_nlp_response_pd = hf_nlp.infer_one(hf_nlp_request_pd)
hf_nlp_response_pd
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

#### [CV](https://huggingface.co/docs/api-inference/detailed_parameters#computer-vision)

All HuggingFace CV tasks are supported and they only accept filename as input. You do not need to provide a key map from input to request JSON because request is a binary. The response JSON could be a list of indefinite length. Please refer to [doc](https://huggingface.co/docs/api-inference/detailed_parameters#computer-vision) for detailed meaning of each parameters and response json -> output dataframe map.

Example usage:
```python
import pandas as pd
from aidb.inference.examples.huggingface_inference_service import HuggingFaceVisionAudio

map_response_to_output = {
  ('0', 'score'): '0.score',
  ('0', 'label'): '0.label',
  ('0', 'box', 'xmin'): '0.box.xmin',
  ('0', 'box', 'ymin'): '0.box.ymin',
  ('0', 'box', 'xmax'): '0.box.xmax',
  ('0', 'box', 'ymax'): '0.box.ymax',
  ('1', 'score'): '1.score',
  ('1', 'label'): '1.label',
  ('1', 'box', 'xmin'): '1.box.xmin',
  ('1', 'box', 'ymin'): '1.box.ymin',
  ('1', 'box', 'xmax'): '1.box.xmax',
  ('1', 'box', 'ymax'): '1.box.ymax',
  ('2', 'score'): '2.score',
  ('2', 'label'): '2.label',
  ('2', 'box', 'xmin'): '2.box.xmin',
  ('2', 'box', 'ymin'): '2.box.ymin',
  ('2', 'box', 'xmax'): '2.box.xmax',
  ('2', 'box', 'ymax'): '2.box.ymax',
  ('3', 'score'): '3.score',
  ('3', 'label'): '3.label',
  ('3', 'box', 'xmin'): '3.box.xmin',
  ('3', 'box', 'ymin'): '3.box.ymin',
  ('3', 'box', 'xmax'): '3.box.xmax',
  ('3', 'box', 'ymax'): '3.box.ymax',
  ('4', 'score'): '4.score',
  ('4', 'label'): '4.label',
  ('4', 'box', 'xmin'): '4.box.xmin',
  ('4', 'box', 'ymin'): '4.box.ymin',
  ('4', 'box', 'xmax'): '4.box.xmax',
  ('4', 'box', 'ymax'): '4.box.ymax'
}

hf_cv = HuggingFaceVisionAudio(
  token=HF_KEY,
  response_keys_to_columns=map_response_to_output,
  model="facebook/detr-resnet-50")
hf_cv_response_pd = hf_cv.infer_one(hf_cv_request_pd)
hf_cv_response_pd
```

Input Series:
```
filename    /path/to/image.png
dtype: object
```

Response:
|   | 0.score  | 0.label | 0.box.xmin | 0.box.ymin | 0.box.xmax | 0.box.ymax | 1.score  | 1.label | 1.box.xmin | 1.box.ymin | 1.box.xmax | 1.box.ymax |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.998632 | tie     | 214        | 254        | 254        | 426        | 0.997547 | person  | 50         | 26         | 511        | 509        |


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
from aidb.inference.example.pytorch_local_inference import PyTorchLocalDetection
groundingdino = PyTorchLocalDetection(
  name='groundingdino',
  model_config_path='path/to/GroundingDINO_SwinT_OGC.py',
  model_checkpoint_path='path/to/groundingdino_swint_ogc.pth',
  caption='your caption',
  use_batch=True,
  batch_size=16,
  col_name='image',)
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

