---
title: "EfficientNet B0のKerasモデルをONNXモデルに変換して推論する"
emoji: "🎓"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["machinelearning", "deeplearning", "computervision", "keras", "onnx"]
published: false
---

# はじめに

「[けしからん画像分類器を作ってみる](202102-pornography-classifier-1)」シリーズでは、KerasとEfficientNet B0を使って画像分類器を実装しました。
その画像分類モデルを、ONNXモデルに変換して推論してみたいと思います。

# ONNXとは？

ONNX（Open Neural Network Exchange）は、Facebook、Microsoftが主導して、機械学習フレームワークの相互運用を実現するためのプロジェクトです。詳しくは、まぁ、ググってください。

* 公式: [ONNX | Home](https://onnx.ai/)
* Wikipedia: [Open Neural Network Exchange](https://ja.wikipedia.org/wiki/Open_Neural_Network_Exchange)

# モデルを変換する方法

KerasのモデルをONNXのモデルに変換する方法は、大きく以下の2つがあります。

* tf2onnxで変換する ← オススメ！
* keras2onnxで変換する

前者が圧倒的にオススメです。勉強のために後者も試してみましたが、なかなか大変でした。

# tf2onnxで変換する


# keras2onnxで変換する
