# Shiba Inu Detector

![Shiba Inu](./shiba.jpg)

Finedtuned version of [ViT](https://huggingface.co/google/vit-base-patch16-224-in21k) to detect a Shiba Inu. It can detect a few other dogs too but that's not the point!

See the [HuggingFace Model](https://huggingface.co/domluna/vit-base-patch16-224-in21k-shiba-inu-detector) for more details.

## Notes

The idea was to become familiar with finetuning a model in 2022 with PyTorch since that's kind of
what's generally feasible as an AI hobbyist for most things - use a API (OpenAI, etc). or finetune a model (HuggingFace).

In that sense I started out using [PyTorch Lightning](https://www.pytorchlightning.ai/) which was quite pleasant until I wanted
to deploy the model. At that point it wasn't clear what I needed, or didn't need to do with the Lightning abstractions.

It was at this point I found out about the [Trainer API](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#trainer) within transformers which turns out does everything for you (!?). It even
sets up WanDB runs and pushes the best model to the HuggingFace hub.

Once uploaded to the hub you can use the model for inference absurdly easily:

```python

from transformers import pipeline, AutoModelForImageClassification, AutoFeatureExtractor

inf_model = AutoModelForImageClassification.from_pretrained('domluna/vit-base-patch16-224-in21k-shiba-inu-detector')
inf_feature_extractor = AutoFeatureExtractor.from_pretrained('domluna/vit-base-patch16-224-in21k-shiba-inu-detector')

pipe = pipeline("image-classification", model=inf_model, feature_extractor=inf_feature_extractor)

# returns [{'score': 0.333, 'label': 'Shiba Inu Dog'}, ..., {...}]
result = pipe(img)

def is_shiba_inu(input_img, threshold=0.5):
    scores = pipe(input_img)
    print('predicted scores', scores)

    for s in scores:
        if s['score'] >= threshold and s['label'] == 'Shiba Inu Dog':
            return True

    return False
```

To have the model as a ".pt" file for later use in inference.


```python
model = AutoModelForImageClassification.from_pretrained('domluna/vit-base-patch16-224-in21k-shiba-inu-detector', torchscript=True)

inputs = inf_feature_extractor(images=img, return_tensors="pt")

# inference on the CPU
traced_model = torch.jit.trace(model, inputs["pixel_values"].cpu())

torch.jit.save(traced_model, "traced_model.pt")

traced_model = torch.jit.load("traced_model.pt")
traced_model.eval()

traced_model(**inputs)
# output: (tensor([[-0.3375, -0.6140, -0.4331,  1.0791]], grad_fn=<AddmmBackward0>),)


# In [178]: %timeit inf_model(**inputs)
# 83.5 ms ± 580 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
# 
# In [179]: %timeit traced_model(**inputs)
# 85.5 ms ± 7.73 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
# 
# In [180]: %timeit frozen_model(**inputs)
# 82.3 ms ± 1.8 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

It doesn't look like these optimizations do very much for this model on my CPU. But I believe the main
advantage of tracing the model is you can load it in a C++ script.
