---
title: "画像からセクシー女優を検索するツールを実装してみた"
emoji: "👙"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["machinelearning", "deeplearning", "computervision", "python", "検索"]
published: true
---

# はじめに

画像に含まれているセクシー女優（と言うか、端的にはAV女優）を検索するOSS（オープンソースソフトウェア）を実装してみました。どうぞご査収ください。

https://github.com/kleamp1e/pornstar-recognizer

# なぜ作ったの？

画像からセクシー女優を検索するサイトは、探せばいくつか見つかりますが、以下のような不満がありました。

* UIがイマイチ
    * 特に1枚の画像に複数の人物が含まれている場合に、サクサクと検索できない
* 動画から簡単に検索できない
    * 自分でフレームを切り出し、画像に変換する必要があって面倒
* インターネットに画像を送信する必要がある
    * プライベートな写真から「似ている人」を探す場合は送信し辛いですよね？
* APIが公開されていない
    * まあ、そりゃそうですね

そんなわけで、勉強も兼ねて自分で実装してみました。

# どんな技術を使ったの？

いわゆる「顔検出」、「顔認証」という技術を使いました。

Zennを眺めていたら、[Yuya Kato](https://zenn.dev/yuyakato)さんがシリーズものの記事を公開してくれており、とても参考になりました。ありがたや。

* [InsightFaceとFastAPIで顔検出サーバを作ってみた](https://zenn.dev/yuyakato/articles/6a1d8177901381)
* [InsightFaceの顔検出結果をNext.jsで可視化してみた](https://zenn.dev/yuyakato/articles/e96b9d8ec289cc)
* [InsightFaceで顔認証（特徴量抽出、比較）してみた](https://zenn.dev/yuyakato/articles/d35b185d36a33b)
* [InsightFaceをGPUで動かしてみた](https://zenn.dev/yuyakato/articles/c780a08c8385e7)

また、ちょっと調べただけでも、同じようなテーマの記事が見つかりました。みんな考えることは一緒ですね。

* [ディープラーニングで「顔が似ているAV女優を教えてくれるbot」を構築 - Qiita](https://qiita.com/tmnck/items/af82deb04d432f1f4f6e)
* [chainerによるディープラーニングでAV女優の類似画像検索サービスをつくったノウハウを公開する - Qiita](https://qiita.com/xolmon/items/0b82f4861cf93fd28e33)
* [Facenetを使った類似AV女優検索 - Qiita](https://qiita.com/zeze/items/1cec8c75833c853b5074)

# どこからデータを集めたの？

日本でAVと言えば「[FANZA](https://www.dmm.co.jp/top/)」（リンク先は成人向けなので注意）ですよね。
都合が良いことに「[AV女優一覧](https://www.dmm.co.jp/digital/videoa/-/actress/recommend/)」（リンク先は成人向けなので注意）のページもあるので、顔の画像はこちらから拝借しました。

今回、OSSとして公開したツールに含まれているのは、作品数順上位1,000人、それぞれ5枚の5,000枚分のデータ（顔特徴量のみ、画像は含まず）です。

# どうやって使うの？

まだフロントエンド（ウェブアプリなど）は作成しておらず、現状ではバックエンド（APIサーバ）のみです。
そのため、`curl`などのツールを使ってAPIを呼び出す必要があります。

画像からの検索は、「顔検出」と「顔識別」の2段階に分かれています。

* 顔検出: `POST /detect`に顔を含む画像を送信します。
    * 画像に含まれる複数の顔が検出されます。
    * それぞれの顔情報には、バウンディングボックスや顔特徴量などが含まれています。
* 顔識別: `POST /recognize`に、顔特徴量を含むJSONを送信します。
    * 送信するJSONデータは`{"embedding": "xxx"}`みたいな感じです。
    * セクシー女優の名前、類似度、FANZAの動画一覧ページへのURLなどが出力されます。

具体例を以下に示します。

```sh
# 検索対象の画像を取得します。
wget https://pics.dmm.co.jp/mono/actjpgs/hatano_yui.jpg

# 顔認識します。
curl -X POST \
  --header "Content-Type: multipart/form-data" \
  --form "file=@hatano_yui.jpg;type=image/jpeg" \
  http://localhost:8001/detect \
  > hatano_yui.json

# 顔認識した結果から顔特徴量を抽出します。
jq "{embedding: .response.faces[0].embedding}" \
  < hatano_yui.json \
  > hatano_yui_embedding.json

# 顔特徴量からAV女優を検索します。
curl -X POST \
  --header "Content-Type: application/json" \
  --data-binary @hatano_yui_embedding.json \
  http://localhost:8001/recognize
```

ビルド方法、起動方法などは[README.md](https://github.com/kleamp1e/pornstar-recognizer/blob/main/README.md)を参照してください。

# 最後に

既存のPythonライブラリを使うことで、簡単に顔検出、顔認証することができました。
ぶっちゃけ、難しかった（と言うか面倒だった）のは、顔画像の収集と前処理です。

さすがにAPIサーバだけだと使いづらいので、次はフロントエンドを作ってみようと思います。
現時点でもAPIがあるので、動画からフレームを切り出して自動的に検索・・・なども簡単に行えそうです。夢が広がりますね♪

今のところ1,000人分のデータしか含まれていませんが、要望があれば拡充したいと思います。
