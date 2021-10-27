## faster rcnn in tf2-keras

### bilibili视频讲解
[![Watch the video](https://raw.githubusercontent.com/yyccR/Pictures/master/faster_rcnn/faster-rcnn视频页.png)](https://www.bilibili.com/video/BV1Eg411G77J?share_source=copy_web)


### 测试效果 Faster-RCNN

- VOC2012

<img src="https://raw.githubusercontent.com/yyccR/Pictures/master/faster_rcnn/faster_rcnn_voc_detect_res1.png" width="350" height="230"/>  <img src="https://raw.githubusercontent.com/yyccR/Pictures/master/faster_rcnn/faster_rcnn_voc_detect_res2.png" width="350" height="230"/>

<img src="https://raw.githubusercontent.com/yyccR/Pictures/master/faster_rcnn/faster_rcnn_voc_detect_res3.png" width="350" height="230"/>  <img src="https://raw.githubusercontent.com/yyccR/Pictures/master/faster_rcnn/faster_rcnn_voc_detect_res4.png" width="350" height="230"/>

- bd100k

<img src="https://raw.githubusercontent.com/yyccR/Pictures/master/faster_rcnn/faster_rcnn_bd100_detect_res1.png" width="350" height="230"/>  <img src="https://raw.githubusercontent.com/yyccR/Pictures/master/faster_rcnn/faster_rcnn_bd100_detect_res2.png" width="350" height="230"/>

<img src="https://raw.githubusercontent.com/yyccR/Pictures/master/faster_rcnn/faster_rcnn_bd100_detect_res3.png" width="350" height="230"/>  <img src="https://raw.githubusercontent.com/yyccR/Pictures/master/faster_rcnn/faster_rcnn_bd100_detect_res4.png" width="350" height="230"/>



### 一些训练要点

- faster-rcnn样本平衡原则:

```python
faster-rcnn即使最后对正负样本做了限制, 但对于单目标小目标的样本来说还是会导致训练不平衡, 故可在最后计算loss处再平衡一次.
```

- loss取舍问题:

```python
由于rcnn系列loss来源多个, 综合起来最后loss可能过大, 可观察后调整, 开始loss在0.5-1.0较为合适, 倘若loss一直比较大, 训练容易导致奔溃loss爆炸.
```