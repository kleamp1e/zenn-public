---
title: "けしからん画像分類器を作ってみる (3) データ収集 その2"
emoji: "👙"
type: "idea" # tech: 技術記事 / idea: アイデア
topics: ["machinelearning", "deeplearning", "computervision", "python", "ruby"]
published: true
---

# 目次

* [けしからん画像分類器を作ってみる (1) 序章](202102-pornography-classifier-1)
* [けしからん画像分類器を作ってみる (2) データ収集](202102-pornography-classifier-2)
* けしからん画像分類器を作ってみる (3) データ収集 その2（本記事）

# データ収集について

[前回](202102-pornography-classifier-2)、「約48万枚の静止画、約1,700本の短い動画を集めました」とさらりと書きましたが、もちろん様々なハマりどころ、紆余曲折がありました。今回は、その辺りについて書いてみたいと思います。

機械学習とは関係なく、ウェブ上から様々な情報を自動的に集めるツールを「クローラ」（Crawler、crawl=這い回る）と呼んだりします。
他にも「エージェント」（Agent）、「スパイダー」（Spider）、「コレクタ」（Collector）、「ボット」（Bot）などと呼ばれますが、本記事ではクローラと呼ぶことにします。
また、（主に人間用に作られた）ウェブページから情報を取り出すことを「ウェブスクレイピング」（Web scraping）と呼びます。詳しく調べたい時には、検索キーワードに加えると良いと思います。ぶっちゃけ、無茶苦茶泥臭い作業です。

クローラの実装に関して、工夫した点、上手くいった点などは以下の通りです。

* クローラはRubyで実装しました
* `wget`、`curl`を活用しました
* MQTT経由でクローラの状態を観察できるようにしました
* メタ情報はJSON形式のファイルとして出力しました
* URLの正規化を頑張りました
* HTMLもすべて保存しました
* データベースは使用しませんでした

# クローラはRubyで実装しました

[最初の記事](202102-pornography-classifier-1)で「Pythonで書きます」と言った早々から別のプログラミング言語が登場です。

今回、画像や動画を収集するクローラはRubyで実装しました。「[Nokogiri](https://nokogiri.org/)が便利すぎるから」と言うのが一番の理由です。
以前、Pythonでクローラを実装したこともありますが、少なくとも当時はあまり良いライブラリがなく、かなり辛いものでした。
使い慣れていることもありますが、Nokogiriであればサクサクとスクレイピングできます。切れ味の良い道具は良いですね。

# `wget`、`curl`を活用しました

ウェブページ（HTML）の取得には、基本的には標準添付ライブラリの[open-uri](https://docs.ruby-lang.org/ja/latest/library/open=2duri.html)を使用しました。

Zennっぽくたまにはコードを出しておくと、以下の様なRubyコードでGoogleのトップページのHTMLを標準出力に出力できます。（ま、実際には[Object#display](https://docs.ruby-lang.org/ja/latest/method/Object/i/display.html)なんて普段は使いませんが、思考通りの順番で書ける幸せ表現するPythonに対するアンチテーゼとして）

```rb
require "open-uri"
open("https://www.google.com/").read.display
```

ウェブページの取得は、まあこれで良いのですが、動画となると少しサイズが大きいので「進捗状況を表示したい」みたいな欲が出てきます。また、途中で失敗した場合に、最初からではなく途中から再開してほしいですよね。
そこで便利なのが`wget`や`curl`のようなツールです。餅は餅屋。自分で色々と実装することは諦めて（と言うか良い意味で手を抜いて）、まるっと任せちゃいましょう。
ちなみに`wget`を使うのがオールドタイプ、`curl`を使うのがニュータイプらしいです。（僕は環境にあるものを使うというアダプティブタイプです）

# MQTT経由でクローラの状態を観察できるようにしました

クローラの中にPub/Sub機構を組み込んで、状態を観察できるようにしました。

具体的には、パブリッシャー（Publisher）としてMQTT（[Wikipedia](https://ja.wikipedia.org/wiki/MQ_Telemetry_Transport)）サーバに接続し、収集したデータを随時出力するようにしました。
こうしておくことで、収集した画像を別のツールでリアルタイムにプレビューしたり、推論したりできて便利でした。
ちなみにパブリッシュした情報は、収集したHTML、収集した画像（をBase64エンコードしたもの）、それらのメタ情報、収集に関する統計情報などです。

MQTTの他にも[NATS](https://nats.io/)を利用したりもしました。ぶっちゃけ方法は何でも良いですが、リアルタイムに観察できるのはとても便利でした。お勧め。

# 今日はここまで

7トピック挙げましたが、3トピック書いたところで今日は力尽きました・・・。
情報量や完成度よりもスループットを重視して記事を書こうと思いますので、今日はここまで！