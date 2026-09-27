# Camera-robust recognition

Inspired by [TIACam](#references). A frozen CLIP encoder and a small head are trained while an augmentor searches for perspective, colour, blur, noise, and moire that hurt the head. The output is a class name.

Trained and tested on Freiburg Groceries: 25 classes, 988 test images.

## Models

| | |
| --- | --- |
| **M0** | Zero-shot CLIP |
| **M1** | Linear probe on frozen CLIP features |
| **M2** | Same head, fixed random augmentation |
| **M3** | Head trained against the augmentor |
| **M3-t0** | M3 with the text loss turned off |

## Headline numbers

- Clean M3 accuracy: **0.9048582995951417**
- Combined severity 1.0: M3 **0.37955465587044535**, M2 **0.3643724696356275**, M1 **0.34615384615384615**, M0 **0.18016194331983806**
- Phone recapture (25 photos): M2 **0.76**, M3 **0.68**

## Results

Severity 0.0 is the clean image. Severity 1.0 is the strongest distortion in the sweep. Every cell is copied from `results/expA_sweeps.csv` or `results/recapture_seed0.csv`.

### Experiment A — distortion sweep

| Model | Clean | Geo 1.0 | Photo 1.0 | Blur 1.0 | Noise 1.0 | Moire 1.0 | Combined 1.0 | Phone recapture (n=25) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M0 | 0.5121457489878543 | 0.46356275303643724 | 0.4868421052631579 | 0.29048582995951416 | 0.451417004048583 | 0.38360323886639675 | 0.18016194331983806 | 0.44 |
| M1 | 0.861336032388664 | 0.8006072874493927 | 0.840080971659919 | 0.5637651821862348 | 0.7834008097165992 | 0.7155870445344129 | 0.34615384615384615 | 0.6 |
| M2 | 0.7449392712550608 | 0.7165991902834008 | 0.7257085020242915 | 0.5313765182186235 | 0.7439271255060729 | 0.6811740890688259 | 0.3643724696356275 | 0.76 |
| M3 | 0.9048582995951417 | 0.8522267206477733 | 0.8714574898785425 | 0.5951417004048583 | 0.8431174089068826 | 0.7935222672064778 | 0.37955465587044535 | 0.68 |
| M3-t0 | 0.032388663967611336 | 0.032388663967611336 | 0.032388663967611336 | 0.032388663967611336 | 0.032388663967611336 | 0.032388663967611336 | 0.032388663967611336 | 0.04 |

Clean accuracy is the severity 0.0 row. It is the same number for every kind of a given model in `results/expA_sweeps.csv`. M3-t0 stays at 0.032388663967611336 for every kind and every severity in that file.

![Experiment A accuracy against distortion severity for M0, M1, M2, and M3.](plots/expA_sweeps.png)

*Top-1 accuracy on the Freiburg test set as severity goes from 0 to 1, one panel per distortion.*

### Experiment B — leave-one-out

At severity 1.0, from `results/expB_leaveout.csv`. M3-drop was trained with the other four ops and then tested on the one it never saw.

| Held out | M3 | M3-drop |
| --- | --- | --- |
| geo | 0.8522267206477733 | 0.8532388663967612 |
| photo | 0.8714574898785425 | 0.8795546558704453 |
| blur | 0.5951417004048583 | 0.5668016194331984 |
| noise | 0.8431174089068826 | 0.8481781376518218 |
| moire | 0.7935222672064778 | 0.8006072874493927 |

![Experiment B leave-one-out bars at severity 1.0.](plots/expB_leaveout.png)

*Full M3 next to the head trained without that distortion, both scored at severity 1.0.*

### Experiment C — phone recapture

![Experiment C accuracy on 25 phone recaptures.](plots/recapture_results.png)

*Accuracy of each model on the 25 photos recaptured with a phone.*

## Method

The encoder is a frozen OpenAI CLIP ViT-B/32 at 224 px. Its feature space already sits next to text, so the backbone stays fixed. The head is a residual network on those 512-d features, L2-normalized. The augmentor, following [TIACam](#references), is trained by gradient ascent on the invariance loss: it looks for a setting of perspective, colour, blur, noise, and moire that hurts this head, instead of drawing a random augmentation. Class names are encoded as text anchors and the head is pulled toward the right anchor with cross-entropy. Without that term the head collapses: M3-t0 never leaves 0.032388663967611336 on the test sweep. M2 uses the same head and the same families of distortion, but the augmentation is fixed and random, so the two runs separate the learned search from plain augmentation.

## Latency

Median time for one 224 px image through CLIP and the M3 head, 200 timed runs after 20 warmup runs (`results/latency.csv`).

| Device | Median |
| --- | --- |
| GPU | 6.337037500998122 ms |
| CPU | 146.9995760007805 ms |

## Limitations

Exp C used only 25 photos, so an 8-point gap between models is not something to make strong claims about. M3 beats M0 and M1 on the synthetic sweeps, and beats M2 on Exp A. On Exp B the full model and the held-out model are close at severity 1.0, and blur is the only held-out kind where full M3 is higher (0.5951417004048583 against 0.5668016194331984). On the phone recapture M2 is ahead of M3, 0.76 against 0.68. Those photos are a screen recapture, not a live camera, and the training images are a public grocery set.

## Reproduce

```
python prep_data.py --seed 0
python train_baselines.py --seed 0
python train_ours.py --seed 0 --epochs 15
python train_ours.py --seed 0 --lam_text 0 --epochs 15
python train_ours.py --seed 0 --epochs 15 --ops geo,photo,noise,blur
python train_ours.py --seed 0 --epochs 15 --ops moire,photo,noise,blur
python train_ours.py --seed 0 --epochs 15 --ops moire,geo,noise,blur
python train_ours.py --seed 0 --epochs 15 --ops moire,geo,photo,blur
python train_ours.py --seed 0 --epochs 15 --ops moire,geo,photo,noise
python eval_recognition.py --seed 0
python plot_results.py --seed 0
python select_recapture_images.py --seed 0
python eval_recapture.py --seed 0
```

## References

Tanvir, A. A., Dasgupta, A., and Zhong, X. (2026). TIACam: Text-Anchored Invariant Feature Learning with Auto-Augmentation for Camera-Robust Zero-Watermarking. arXiv:2602.18863.

Jund, P., Abdo, N., Eitel, A., and Burgard, W. (2016). The Freiburg Groceries Dataset. arXiv:1611.05799.

Radford, A., Kim, J. W., Hallacy, C., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. ICML. (CLIP, used here as the frozen backbone.)
