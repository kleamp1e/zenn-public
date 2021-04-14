---
title: "けしからん画像分類器を作ってみる (9) 推論"
emoji: "👙"
type: "idea" # tech: 技術記事 / idea: アイデア
topics: ["machinelearning", "deeplearning", "computervision", "python", "keras"]
published: true
---

# 目次

* [けしからん画像分類器を作ってみる (1) 序章](202102-pornography-classifier-1)
* [けしからん画像分類器を作ってみる (2) データ収集 その1](202102-pornography-classifier-2)
* [けしからん画像分類器を作ってみる (3) データ収集 その2](202102-pornography-classifier-3)
* [けしからん画像分類器を作ってみる (4) データ収集 その3](202103-pornography-classifier-4)
* [けしからん画像分類器を作ってみる (5) データ管理 その1](202103-pornography-classifier-5)
* [けしからん画像分類器を作ってみる (6) データ管理 その2](202103-pornography-classifier-6)
* [けしからん画像分類器を作ってみる (7) 学習 その1](202104-pornography-classifier-7)
* [けしからん画像分類器を作ってみる (8) 学習 その2](202104-pornography-classifier-8)
* けしからん画像分類器を作ってみる (9) 推論（本記事）

# ついに`keshikaran.py`を手に入れた

[前回](202104-pornography-classifier-8)、「EfficientNet B0」を使った画像分類モデルを約2万枚の画像で学習し、精度85%を得ることができました。
「テストデータで精度85%！」と言われても、現実の「けしからん画像」を分類できないと意味が無いですよね。

今回は、得られた画像分類モデルを使って推論（Inference、Predict）するスクリプトを書いてみます。
そう、ついに[最初の記事](202102-pornography-classifier-1)で妄想した`keshikaran.py`を手に入れる時が来たのです！

# モデルの読み込みで少しハマった

学習スクリプト`train.py`は、学習済みのモデルを`model.save("model.h5")`みたいな感じでファイルとして書き込んでいます。
このモデルファイル`model.h5`を読み込んで推論してみましょう。

・・・いきなりモデルの読み込みで少しハマりました。

```py
import tensorflow as tf
model = tf.keras.models.load_model("model.h5")
```

みたいなコードでモデルを読み込むと、`ValueError: Unknown layer: KerasLayer`が発生しました。

```
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/local/lib/python3.8/dist-packages/tensorflow/python/keras/saving/save.py", line 206, in load_model
    return hdf5_format.load_model_from_hdf5(filepath, custom_objects,
  File "/usr/local/lib/python3.8/dist-packages/tensorflow/python/keras/saving/hdf5_format.py", line 183, in load_model_from_hdf5
    model = model_config_lib.model_from_config(model_config,
  File "/usr/local/lib/python3.8/dist-packages/tensorflow/python/keras/saving/model_config.py", line 64, in model_from_config
    return deserialize(config, custom_objects=custom_objects)
  File "/usr/local/lib/python3.8/dist-packages/tensorflow/python/keras/layers/serialization.py", line 173, in deserialize
    return generic_utils.deserialize_keras_object(
  File "/usr/local/lib/python3.8/dist-packages/tensorflow/python/keras/utils/generic_utils.py", line 354, in deserialize_keras_object
    return cls.from_config(
  File "/usr/local/lib/python3.8/dist-packages/tensorflow/python/keras/engine/sequential.py", line 492, in from_config
    layer = layer_module.deserialize(layer_config,
  File "/usr/local/lib/python3.8/dist-packages/tensorflow/python/keras/layers/serialization.py", line 173, in deserialize
    return generic_utils.deserialize_keras_object(
  File "/usr/local/lib/python3.8/dist-packages/tensorflow/python/keras/utils/generic_utils.py", line 346, in deserialize_keras_object
    (cls, cls_config) = class_and_config_for_serialized_keras_object(
  File "/usr/local/lib/python3.8/dist-packages/tensorflow/python/keras/utils/generic_utils.py", line 296, in class_and_config_for_serialized_keras_object
    raise ValueError('Unknown ' + printable_module_name + ': ' + class_name)
ValueError: Unknown layer: KerasLayer
```

少し調べてみると、`hub.KerasLayer`を使っている場合は、読み込み時にカスタムレイヤーについての情報が必要とのこと。
以下の様に`custom_objects`を指定することで、モデルを読み込むことができました。

```py
import tensorflow as tf
import tensorflow_hub as hub
model = tf.keras.models.load_model("model.h5", custom_objects={"KerasLayer": hub.KerasLayer})
```

**参考:**

* [[TF2.0] KerasLayer cannot be loaded from .h5 · Issue #26835 · tensorflow/tensorflow](https://github.com/tensorflow/tensorflow/issues/26835)

# `keshikaran.py`を実装する

モデルの読み込みが成功したので、早速`keshikaran.py`を実装してみましょう。仕様は以下の通りとします。

* 画像のファイル名をコマンドライン引数として渡す。
* 標準出力に「けしからん度合い」（要するにエロい度合い）を0〜1の実数で出力する。

実際のスクリプトは以下の通りです。短いですね。

```py:keshikaran.py
#!/usr/bin/env python3

import numpy as np
import PIL.Image
import sys
import tensorflow as tf
import tensorflow_hub as hub

image_path = sys.argv[1]

image = PIL.Image.open(image_path).convert("RGB").resize((224, 224))
image = np.array(image) / 255
image = np.expand_dims(image, 0)

model = tf.keras.models.load_model("model.h5", custom_objects={"KerasLayer": hub.KerasLayer})

predictions = model.predict(image)
print(predictions[0][0])
```

# 推論してみる

では実際の画像を入力して推論してみましょう。以下の画像は[ぱくたそ](https://www.pakutaso.com/)からお借りしました。

まずはこちらの[犬の画像](https://www.pakutaso.com/20170355090post-10833.html)から推論してみます。ちょっと羊っぽくて可愛いですね。

![](https://storage.googleapis.com/zenn-user-upload/8f6nbepxl9as9f5wvurgam5ebsjo)

```
$ python3 keshikaran.py dog.jpg
0.29267535
```

思ったよりも数値が高いですが、中間（0.5）は下回っているので「けしからんくない画像」と見なして良さそうです。

続いて、こちらの[ちょっとセクシーな女性の画像](https://www.pakutaso.com/20200943246post-18368.html)を推論してみます。

![](https://storage.googleapis.com/zenn-user-upload/1c7oyioay8ey5tyqypaj3cgrqd75)

```
$ python3 keshikaran.py woman.jpg
0.824198
```

犬の写真よりもだいぶ数値が上がりました。こちらは「けしからん画像」のようです。

さらに、より実践的（？）な画像を入力してみます。
`aHR0cHM6Ly9ibG9nLWltZ3MtMTQ1LmZjMi5jb20vcy91L20vc3Vtb21vY2hhbm5lbC92cl9ha2lyYWVsbHlfMTAyNTgtMDAyLmpwZw==`から入手した画像（リンクするのは憚られるので、URLをBase64エンコードしています）を入力してみます。

```
$ python3 keshikaran.py sexy.jpg
0.9372734
```

キター！いやー、これはけしからん。

最後にもう1枚、肌色が多めの[こちらの画像](https://www.pakutaso.com/20151207336post-6399.html)を入力してみます。

![](https://storage.googleapis.com/zenn-user-upload/f62658v3db0sqkqbavjx1dtcbp27)

```
$ python3 keshikaran.py man_and_dog.jpg
0.34251067
```

こちらは「けしからんくない画像」の様です。激しく同意。

# 今回はここまで

ついに妄想していた`keshikaran.py`を実装することができました。ここでは数枚の画像しか試していませんが、基本的には上手く動作しているようです。
正直、何に使えるかは分からないので、面白いアイデアを思いついたら教えて欲しいです。前回も書きましたが、要望があれば学習済みモデルは公開します。

途中、ラベル付けなどのネタを飛ばしてしまったので、追々、そちらも書いていければと思っています。今日はここまで！
