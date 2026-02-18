# Lab Summary and Results

This lab focused on fine-tuning the image encoder of a CLIP-like model, utilizing a ResNet50 backbone, for improved image-to-text retrieval and zero-shot image classification. The project processes the COCO 2014 dataset, caching text embeddings to efficiently train the image encoder to align visual and textual representations.

## Evaluation Results

Here are some visual results from the evaluation of the fine-tuned model:

### Classification Examples

![Classification Example 1](eval_results/classification_COCO_val2014_000000004795.png)
![Classification Example 2](eval_results/classification_COCO_val2014_000000148739.png)
![Classification Example 3](eval_results/classification_COCO_val2014_000000542325.png)

### Retrieval Examples

![Retrieval Example - Animal](eval_results/retrieval_animal.png)
![Retrieval Example - Car](eval_results/retrieval_car.png)
![Retrieval Example - Food](eval_results/retrieval_food.png)
![Retrieval Example - Person](eval_results/retrieval_person.png)
![Retrieval Example - Sport](eval_results/retrieval_sport.png)
