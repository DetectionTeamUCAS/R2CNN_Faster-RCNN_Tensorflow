# R2CNN_Faster-RCNN_Tensorflow

## Abstract
This is a tensorflow re-implementation of [R<sup>2</sup>CNN: Rotational Region CNN for Orientation Robust Scene Text Detection](https://arxiv.org/abs/1706.09579). It should be noted that we did not re-implementate exactly as the paper and just adopted its idea.

This project is based on [Faster-RCNN](), and completed by [YangXue](https://github.com/yangxue0827) and [YangJirui](https://github.com/yangJirui).

## [DOTA](https://captain-whu.github.io/DOTA/index.html) test results      
![1](DOTA.png)

## Download Model
1、please download [resnet50_v1](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)、[resnet101_v1](http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz) pre-trained models on Imagenet, place it to data/pretrained_weights.     
2、please download [mobilenet_v2](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz) pre-trained model on Imagenet, place it to data/pretrained_weights/mobilenet.     
3、please download resnet101_v1 trained model by this project, place it to output/trained_weights.   


## Comparison
| Approaches | mAP | PL | BD | BR | GTF | SV | LV | SH | TC | BC | ST | SBF | RA | HA | SP | HC |
|------------|:---:|:--:|:--:|:--:|:---:|:--:|:--:|:--:|:--:|:--:|:--:|:---:|:--:|:--:|:--:|:--:|
|[SSD inception-v2](https://link.springer.com/chapter/10.1007%2F978-3-319-46448-0_2)|17.84|41.06|24.31|4.55|17.1|15.93|7.72|13.21|39.96|12.05|46.88|9.09|30.82|1.36|3.5|0.0|
|[YOLOv2](https://arxiv.org/abs/1612.08242)|25.492|52.75|24.24|10.6|35.5|14.36|2.41|7.37|51.79|43.98|31.35|22.3|36.68|14.61|22.55|11.89| 
|[R-FCN](http://papers.nips.cc/paper/6465-r-fcn-object-detection-via-region-based-fully-convolutional-networks)|30.84|39.57|46.13|3.03|38.46|9.1|3.66|7.45|41.97|50.43|66.98|40.34|51.28|11.14|35.59|17.45|
|[FR-H](https://ieeexplore.ieee.org/abstract/document/7485869/)|39.95|49.74|64.22|9.38|56.66|19.18|14.17|9.51|61.61|65.47|57.52|51.36|49.41|20.8|45.84|24.38|
|[FR-O](https://arxiv.org/abs/1711.10398)|54.13|79.42|**77.13**|17.7|64.05|35.3|38.02|37.16|89.41|**69.64**|59.28|50.3|52.91|47.89|47.4|46.3|
|Ours|**60.67**|**80.94**|65.75|**35.34**|**67.44**|**59.92**|**50.91**|**55.81**|**90.67**|66.92|**72.39**|**55.06**|**52.23**|**55.14**|**53.35**|**48.22**|
