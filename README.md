# keras gan cat image generation

<p align="center">
  <img src="images/logo.jpg">
</p>

KERAS를 사용하여 DCGAN으로 고양이 이미지를 생성하는 프로젝트입니다.

데이터 셋은 kaggle의 [cat dataset](https://www.kaggle.com/crawford/cat-dataset)을 사용했습니다.

GAN을 공부하면서 많은 도움을 받은 [02_DCGAN_CIFAR10.ipynb](https://github.com/mafda/generative_adversarial_networks_101/blob/master/src/cifar10/02_DCGAN_CIFAR10.ipynb)를 참고하여 프로젝트를 진행했습니다. 자세한 내용은 [GAN_STUDY](GAN_STUDY.md) 문서를 참고해 주십시오.

<p align="center">
  <img src="images/result.png">
</p>

## 모델 구조

이미지의 크기는 (64,64,3) 입니다.

**generator**

![generator](images/generator.png)

**discriminator**

![discriminator](images/discriminator.png)


