# Mixture-of-Experts-based Hierarchical Dynamic Multimodal Fusion Network for Dermatological Diagnosis


**Abstract**: In clinical computer-aided diagnosis of skin cancer, integrating skin lesion images with clinical metadata is crucial for improving diagnostic accuracy. However, most existing fusion methods rely on fixed strategies, such as weighted averaging, which do not adaptively assess the strength of feature expression. This results in an imbalance in the fusion of key discriminative information across multiple modalities and scales. To address these limitations, we propose the Mixture-of-Experts-based Hierarchical Dynamic Multimodal Fusion Network (MHDF-Net), which enables progressive and dynamic fusion of images and metadata. Specifically, we design a Local Cross-Modal Mixture-of-Experts (LC-MoE) that incorporates a Top-k Cross-Attention (TKCA) parallel branch structure, in combination with the Local Cross-Modal Gate (LC-Gate). This design enhances discriminative feature extraction across modalities and resolves the issue of dynamic complementary balance in highly similar samples. Additionally, we propose the Global Multi-scale Mixture-of-Experts (GM-MoE), which employs a multi-scale hierarchical expert architecture to adaptively integrate global contextual information of the lesion. This approach accommodates the diversity of lesion morphology by modeling spatial semantics. Extensive experiments on two publicly available skin cancer diagnosis datasets show that our model significantly outperforms existing dermatological disease classification algorithms, offering new insights for multimodal fusion in skin cancer diagnosis. The codes can be publicly available from <https://github.com/ChenHu-0413/MHDF-Net>.

## Network Architecture

![](https://cdn.modevol.com/user/cl3h8qb9c02f201m75unm19l8/images/gqto36ztim8dik6y6c1vli6v.png)

## Enviroments

* Windows/Linux both support

* python 3.8

* PyTorch 1.13.1

## Datasets

In this paper, we evaluate our work using the [PAD-UFES-20 dataset](https://data.mendeley.com/datasets/zr7vgbcyr2/1)  and the [Derm7pt dataset](https://github.com/jeremykawahara/derm7pt).&#x20;

## Run details

To train our , run: train.py

Run eval_eval.py to get the evaluation results.

All experiments are conducted using the PyTorch framework on an NVIDIA GeForce RTX 3090 GPU.     We use the Adam optimizer to update the overall network parameters, with the learning rate set to 0.001, the batch size set to 32, and the number of training epochs is 150. We adopt cross-entropy loss as the loss function. The number of training epochs was set to 150 to ensure fair comparison with existing methods and because preliminary experiments showed that the model had converged by this point, with stable loss, thereby avoiding overfitting from prolonged training.

## Acknowledgement 

Our code borrows a lot from:

> [PAD-UFES-20 dataset](https://data.mendeley.com/datasets/zr7vgbcyr2/1)

> [Derm7pt dataset](https://github.com/jeremykawahara/derm7pt)

## Contact

Should you have any question or suggestion, please contact huchen20010413@163.com.
