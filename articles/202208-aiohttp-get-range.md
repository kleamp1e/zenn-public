---
title: "aiohttpを使って範囲（Range）リクエストを送信する"
emoji: "📏"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["python", "asyncio", "aiohttp"]
published: true
---

# はじめに

HTTP経由で大きなファイルの一部を取得したい場合、HTTP範囲（Range）リクエストを使うことで実現できます。
HTTP範囲リクエストについては、MDNが詳しいです。

https://developer.mozilla.org/ja/docs/Web/HTTP/Range_requests

今回は、Pythonの非同期HTTPライブラリ`aiohttp`を使って、HTTP範囲リクエストを送信してみました。

# インストール

`aiohttp`は`pip`コマンドでインストールできます。

```sh
pip install aiohttp
```

https://pypi.org/project/aiohttp/

# 実装例

ひとまずコードをどうぞ。

```py
import re

import aiohttp
import asyncio

def make_http_range_header(first_byte_position: int, last_byte_position: int):
    return {
        "Range": "bytes={:d}-{:d}".format(first_byte_position, last_byte_position),
    }

def parse_http_content_range_header(value: str):
    m = re.match(r"^bytes (\d+)-(\d+)/(\d+)$", value)
    return {
        "first_byte_position": int(m.group(1)),
        "last_byte_position": int(m.group(2)),
        "content_length": int(m.group(3)),
    }

async def get_range(session: aiohttp.ClientSession, url: str, first_byte_position: int, last_byte_position: int):
    headers = make_http_range_header(first_byte_position, last_byte_position)
    async with session.get(url, headers=headers) as response:
        body = await response.read()
        return {
            "status": response.status,
            "content_type": response.headers["content-type"],
            "content_length": response.headers["content-length"],
            "content_range": response.headers["Content-Range"],
            "body": body,
        }

async def main():
    async with aiohttp.ClientSession() as session:
        url = "https://lh3.googleusercontent.com/a-/AOh14GiZIqfnPF3cOcl6Q7kUGYKq82YRwUhpPdkYQQdf=s96-c"
        result = await get_range(session, url, 0, 63)
        print("result:", result)
        print("content_range:", parse_http_content_range_header(result["content_range"]))
        assert result["status"] == 206

asyncio.run(main())
```

実行結果は以下の通りです。なお、上記のURLは、私のZennのアイコンです。

```
result: {'status': 206, 'content_type': 'image/png', 'content_length': '64', 'content_range': 'bytes 0-63/19637', 'body': b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00`\x00\x00\x00`\x08\x02\x00\x00\x00m\xfa\xe0o\x00\x00\x00\x03sBIT\x08\x08\x08\xdb\xe1O\xe0\x00\x00\x00\x06bKGD\x00\xff\x00\xff\x00\xff\xa0\xbd'}
content_range: {'first_byte_position': 0, 'last_byte_position': 63, 'content_length': 19637}
```

以下、簡単な解説です。

* `aiohttp`には`Range`ヘッダを生成する機能はなさそうなので、自分で生成しています。
* `Range`ヘッダには「開始バイト位置」と「終了バイト位置」を指定します。サイズではないので注意。
* `Range`ヘッダを適切に処理できるHTTPサーバの場合、HTTPステータスとして`206 Partial Content`が返ります。
* レスポンスには`Content-Range`ヘッダが含まれており、「開始バイト位置」、「終了バイト位置」、「総バイト数」が含まれています。こちらも自分でパースしています。
* ファイルサイズを超えて要求してもエラーにはならず、存在するバイト位置までが返ってくるようです。
* `Accept-Ranges`ヘッダの確認は省略しています。本当はチェックした方が良いかと思います。

# おわりに

割と雑な実装ではありますが、`aiohttp`と`Range`ヘッダを使い、HTTP範囲リクエストを送信することができました。
何らかの参考になれば幸いです。
