# CSCI2470 Deel Learning Project
# How Can Objects Help Video-Language Understanding?

```
This repository is created solely for code review purposes in the Deep Learning course project.
```

## Training

[scripts](./scripts) includes the training scripts of our model.
Because we employ a modality-by-modality fusion strategy, unimodal training must be executed before multimodal training.
For example, to develop a caption+box model on CLEVRER, one should first run `scripts/clevrer_cap.sh` and then run `scripts/clevrer_cap_box_resume.sh`.

## Acknowledgements

This repository uses the codebase of [Vamos](https://github.com/brown-palm/Vamos).