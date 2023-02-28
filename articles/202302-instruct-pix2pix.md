
# はじめに

[InstructPix2Pix](https://github.com/timothybrooks/instruct-pix2pix)をローカルのDocker環境で動かしてみた際のメモです。
予めDocker、NVIDIAドライバ、NVIDIA Container Runtimeのセットアップが必要です。

# 環境

試した環境は以下の通りです。

```text
$ grep PRETTY_NAME /etc/os-release
PRETTY_NAME="Ubuntu 20.04.5 LTS"

$ uname -a
Linux gpu 5.4.0-139-generic #156-Ubuntu SMP Fri Jan 20 17:27:18 UTC 2023 x86_64 x86_64 x86_64 GNU/Linux

$ nvidia-smi | head -n 4
Tue Feb 28 23:05:19 2023
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.60.11    Driver Version: 525.60.11    CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+

$ nvidia-smi -a | grep "Product Name"
    Product Name                          : NVIDIA GeForce RTX 4080

$ docker version
Client:
 Version:           20.10.12
 API version:       1.41
 Go version:        go1.16.2
 Git commit:        20.10.12-0ubuntu2~20.04.1
 Built:             Wed Apr  6 02:14:38 2022
 OS/Arch:           linux/amd64
 Context:           default
 Experimental:      true

Server:
 Engine:
  Version:          20.10.12
  API version:      1.41 (minimum version 1.12)
  Go version:       go1.16.2
  Git commit:       20.10.12-0ubuntu2~20.04.1
  Built:            Thu Feb 10 15:03:35 2022
  OS/Arch:          linux/amd64
  Experimental:     false
 containerd:
  Version:          1.5.9-0ubuntu1~20.04.6
  GitCommit:
 runc:
  Version:          1.1.0-0ubuntu1~20.04.2
  GitCommit:
 docker-init:
  Version:          0.19.0
  GitCommit:

$ docker-compose version
Docker Compose version v2.1.1
```

# Dockerfile

`Dockerfile`の内容は以下の通りです。`requirements.txt`の内容は、リポジトリを直接参照してください。

```Dockerfile
FROM nvcr.io/nvidia/pytorch:22.11-py3
WORKDIR /root/
COPY requirements.txt ./
RUN python3 -m pip install --requirement requirements.txt
```

https://github.com/kleamp1e/202302-instruct-pix2pix

# コードの取得 & ビルド

`git clone`し、Dockerイメージをビルドします。

```sh
git clone https://github.com/kleamp1e/202302-instruct-pix2pix.git
cd 202302-instruct-pix2pix

docker-compose build
```

# 実行

Dockerコンテナを起動し、`example.py`を実行します。
初回はモデルのダウンロードが行われるため、それなりの時間が掛かります。

```sh
docker-compose run --rm shell

# 以下、コンテナ内
python3 example.py
```

なお、`example.py`の内容は以下のページに記載されているものをフォーマットしたものです。

https://huggingface.co/timbrooks/instruct-pix2pix
