# 日本語版 Pheme 音声合成モデル

このリポジトリは、音声合成モデル Pheme (https://github.com/PolyAI-LDN/pheme) を日本語向けにカスタマイズしたものです。

Pheme の詳細については、元論文 [Pheme: Efficient and Conversational Speech Generation](https://arxiv.org/pdf/2401.02839.pdf) をご覧ください。

**本モデルで生成した音声を公開する場合には使用した声のクレジットを記載してください。**
現在使用可能な声は、[黄琴海月さん](https://kikyohiroto1227.wixsite.com/kikoto-utau/kurage) と [ルナイトネイルさん](https://runaitoneiru.fanbox.cc/posts/3786422) です。

本モデルは、下記の禁止事項に該当する行為を除き、自由にご利用いただけます。

**禁止事項:**

・犯罪目的、差別目的、誹謗中傷目的、その他社会通念上不適切な目的で利用する行為

# 環境セットアップ

conda 環境の作成:

``` 
conda create --name pheme3 python=3.10
conda activate pheme3

pip3 install torch torchvision torchaudio
pip3 install -r requirements.txt --no-deps
pip3 install mecab-python3 unidic-lite jaconv
```
事前学習済みの T2S および S2A モデル、辞書、話者特徴量のダウンロード:

``` bash
git clone https://huggingface.co/offtoung/pheme-ja
mv pheme-ja/* .
```

事前学習済みの SpeechTokenizer のダウンロード

``` bash
st_dir="ckpt/speechtokenizer/"
mkdir -p ${st_dir}
cd ${st_dir}
wget "https://huggingface.co/fnlp/SpeechTokenizer/resolve/main/speechtokenizer_hubert_avg/SpeechTokenizer.pt"
wget "https://huggingface.co/fnlp/SpeechTokenizer/resolve/main/speechtokenizer_hubert_avg/config.json" 
cd ../../
```


# 音声の生成

次のようなコマンドを実行することで、音声が生成されます。
この例では、生成音声は output というディレクトリに格納されます。

```
python transformer_infer.py \
--text こんにちは、お元気ですか？ \
--outputdir output \
--voice runaitoneiru \
```

現在、話者として指定可能なのは、
- ルナイトネイルさん (runaitoneiru)
- 黄琴海月さん ひそひそスタイル (kikoto_kurage_hisohiso)
の 2 名です。

# 学習の実行方法
準備中

## Citation

本リポジトリのコードやモデルを利用した発表などを行う場合は、元論文を引用してください。

```Tex
@misc{budzianowski2024pheme,
      title={Pheme: Efficient and Conversational Speech Generation}, 
      author={Paweł Budzianowski and Taras Sereda and Tomasz Cichy and Ivan Vulić},
      year={2024},
      eprint={2401.02839},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```
