---
title: "類似画像検索ツールを作ってみる (1) 序章"
emoji: "🔍"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["machinelearning", "deeplearning", "computervision", "python", "検索"]
published: true
---

# 目次

* 類似画像検索ツールを作ってみる (1) 序章（本記事）
* [類似画像検索ツールを作ってみる (2) 特徴化 その1](202105-similar-search-2)
* [類似画像検索ツールを作ってみる (3) 特徴化 その2](202105-similar-search-3)
* [類似画像検索ツールを作ってみる (4) 類似画像検索](202105-similar-search-4)
* [類似画像検索ツールを作ってみる (5) 類似画像検索サーバ](202105-similar-search-5)
* [類似画像検索ツールを作ってみる (6) Next.js + SVGで可視化](202106-similar-search-6)

# 背景と目的

『[けしからん画像分類器を作ってみる](202102-pornography-classifier-1)』シリーズで紹介した画像分類器を実装する過程で、多くの画像をラベル付けしました。
その際に困った事象に遭遇しました。重複した画像が思ったより多いこと多いこと。
インターネット上の「けしからん画像」には転載が多く、1ビットの相違もないファイルもあれば、拡大縮小、微妙な切り取り、マスキング、ウォーターマークの付加、再エンコードによる劣化など、「人間にとってはほとんど同じだけれど異なるビット列」の画像が大量に存在します。

機械学習のためのラベル付けするのであれば、できるだけユニークで広く分布した画像にラベル付けしたいものです。
内容がほとんど同じ画像にラベル付けしても精度の向上は期待できず、むしろ分布の偏りを生んでしまいます。

色々探してはみましたが、ちょうど良い感じの類似画像検索ツールが見当たらなかったので、画像処理、機械学習の学習も兼ねて作ってみることにしました。

それを応用して、ゆくゆくは「ラベル付けすべき画像群」（ラベル付けされた画像群と似ていない画像群）を提案するツールを作りたいと思っています。

## 参考: 既存の類似画像検索ツール

既存の類似画像検索ツールとして「Apache alike」があるみたいですが、公式サイトが無くなっている上、Apache Mahout、Apache Lucene、Apache Solr・・・と割と大がかりな感じなので、動かす前から諦めてしまいました。

# 目標

実装するにあたって、目標を設定したいと思います。

具体的には「10万枚の画像を、CPUだけを使って（GPUを使わずに）、1秒以内に検索し、類似度が高い順にソートして出力すること」を目標とします。

その他、オプショナルな要件（と言うか方針）は以下の通りです。

* 現実的な速度でインデックスを生成できること。
* インデックスの生成にはGPUを使用すること。
* 容易に画像をインデックスに追加できること。
* Dockerコンテナ内で動作すること。
* Pythonで実装すること。

# 画像の類似度

「類似画像検索」というからには、何らかの方法を用いて「画像間の距離」（類似度）を定義する必要があります。

例えば、今回検索の対象とする画像は800×1400ピクセルくらいの解像度があり、色深度はRGB各8ビットです。
そのため、800ピクセル × 1400ピクセル × 3チャネルで3,360,000個の8ビット整数が存在します。
同じ解像度、同じ色深度の画像を比較する場合、3,360,000個の整数同士のユークリッド距離を取ることもできますが、もっと大局的な、全体的な特徴を比較して欲しいですよね。

古典的な（機械学習を使わない）特徴量として「AKAZE」などがありますが、今回は機械学習の勉強も兼ねているので、機械学習をベースとした特徴量を使ってみたいと思います。
具体的には、画像分類などで使われるモデルのバックボーン部分の出力を特徴量として使います。

# 基本的な戦略

基本的には、以下の戦略（手順）で類似画像の検索を行います。

1. 画像を一定サイズ（例: 224×224ピクセル）にリサイズ（縮小）する。
    * バックボーンネットワークの入力に合わせるため。
    * この時点で、224ピクセル × 224ピクセル × 3チャネルで150,528次元。
2. 縮小した画像をバックボーンネットワークに入力して特徴量を出力する。
    * 例えば、EfficientNet B0なら特徴量は1,280次元。
3. （必要であれば）t-SNEで3次元に圧縮する。
4. （必要であれば）3次元の空間上で1次類似検索する。
5. 1,280次元の空間上で2次類似検索する。
6. ユークリッド距離でソートする。
7. 出力する。

t-SNEによる次元圧縮は、1,000次元程度の空間上における検索が実用的な速度で行えなかった場合に実施したいと思います。

例えば10万枚の画像についてインデックスを作る場合、1,280次元 × 4バイト（32ビット浮動小数点数） × 10万枚で、512MBになります。メモリに収まってしまいそうな感じですね。

ちなみに、高次元データの検索行うライブラリとして、Yahoo Japanが開発/公開している「[NGT](https://github.com/yahoojapan/NGT)」などがありますが、今回は頼らずに実装してみたいと思います。
（以前、少しだけ使ったことがありますが、インデックスへの追加、削除で難がありました・・・現在は知りませんが）

# 特徴化のためのネットワークを選定する

特徴化のためのバックボーンネットワーク（特徴化器）として、以下の3つの候補を挙げました。

* EfficientNet B0
    * https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1
* MobileNet V2
    * https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5
* MobileNet V3
    * https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5

いずれも、TensorFlow Hubで学習済みモデルが提供されており、そこそこの推論時間で、そこそこの精度が出ている、というのが選定理由です。

上記の3つのモデルについて、推論時間を調査してみました。インデックスを生成する時や検索を行う時には特徴量を得るための推論を行う必要があり、その時間が短い方が望ましいためです。
測定環境は以下の通りです。

* CPU: AMD Ryzen 7 3700X（8コア/16スレッド）
* メモリ: 64GB
* GPU: GeForce GTX 1070（メモリ8GB）

バッチサイズを1、2、4、8、16、32に設定して、それぞれ3回測定しました。結果は以下の通りです。

| モデル | バッチサイズ | 推論時間 1回目 [ms] | 2回目 [ms] | 3回目 [ms] | 平均 [ms] | 1枚あたり平均 [ms] |
|:---|---:|---:|---:|---:|---:|---:|
| EfficientNet B0 |  1 |   78.1 |   81.2 |  77.5 |   78.9 | 78.9 |
| EfficientNet B0 |  2 |  106.0 |   98.7 | 103.0 |  102.6 | 51.3 |
| EfficientNet B0 |  4 |  141.0 |  137.0 | 136.0 |  138.0 | 34.5 |
| EfficientNet B0 |  8 |  213.0 |  216.0 | 228.0 |  219.0 | 27.4 |
| EfficientNet B0 | 16 |  419.0 |  383.0 | 422.0 |  408.0 | 25.5 |
| EfficientNet B0 | 32 | 1150.0 | 1080.0 | 860.0 | 1030.0 | 32.2 |
| MobileNet V2    |  1 |   60.5 |   57.9 |  60.1 |   59.5 | 59.5 |
| MobileNet V2    |  2 |   68.5 |   70.5 |  68.4 |   69.1 | 34.6 |
| MobileNet V2    |  4 |   85.9 |   93.0 |  88.6 |   89.2 | 22.3 |
| MobileNet V2    |  8 |  147.0 |  140.0 | 145.0 |  144.0 | 18.0 |
| MobileNet V2    | 16 |  248.0 |  232.0 | 236.0 |  238.7 | 14.9 |
| MobileNet V2    | 32 |  442.0 |  441.0 | 463.0 |  448.7 | 14.0 |
| MobileNet V3    |  1 |   71.0 |   62.0 |  60.4 |   64.5 | 64.5 |
| MobileNet V3    |  2 |   86.8 |  101.0 |  94.4 |   94.1 | 47.0 |
| MobileNet V3    |  4 |  111.0 |  109.0 | 110.0 |  110.0 | 27.5 |
| MobileNet V3    |  8 |  161.0 |  152.0 | 164.0 |  159.0 | 19.9 |
| MobileNet V3    | 16 |  261.0 |  238.0 | 240.0 |  246.3 | 15.4 |
| MobileNet V3    | 32 |  417.0 |  449.0 | 444.0 |  436.7 | 13.6 |

「MobileNet V2」と「MobileNet V3」の推論時間は僅差ですが、後者の方が画像分類における精度が高く、より良い特徴を得ている可能性があるので、今回は特徴化器として「MobileNet V3」を使ってみたいと思います。

# 今回はここまで

特徴化器の選定まで終わりました。次回は実際に特徴量を取得してみたいと思います。

『[類似画像検索ツールを作ってみる (2) 特徴化](202105-similar-search-2)』に続く。

# 参考

* [類似画像検索の3つの手法と精度向上のテクニック - アイマガジン｜i Magazine｜IS magazine](https://www.imagazine.co.jp/%E9%A1%9E%E4%BC%BC%E7%94%BB%E5%83%8F%E6%A4%9C%E7%B4%A2%E3%81%AE3%E3%81%A4%E3%81%AE%E6%89%8B%E6%B3%95%E3%81%A8%E7%B2%BE%E5%BA%A6%E5%90%91%E4%B8%8A%E3%81%AE%E3%83%86%E3%82%AF%E3%83%8B%E3%83%83%E3%82%AF/)
* [類似画像検索システム「EnraEnra」](https://www.jstage.jst.go.jp/article/jjsai/29/5/29_430/_pdf/-char/ja)（PDF）