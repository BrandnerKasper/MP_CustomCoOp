import open_clip

from clip import clip
from open_clip import openai

print(clip.available_models())
print(openai.list_openai_models())
print(open_clip.list_pretrained())

# 'xlm-roberta-base-ViT-B-32'
# 'laion5b_s13b_b90k'

model = open_clip.create_model('RN50', 'openai')
model2 = open_clip.create_model_from_pretrained('RN50', 'openai')

print("hi")

# open_clip.list_openai_models()
# ['RN50', 'RN50-quickgelu', 'RN101', 'RN101-quickgelu', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B-32', 'ViT-B-32-quickgelu', 'ViT-B-16', 'ViT-L-14', 'ViT-L-14-336']

