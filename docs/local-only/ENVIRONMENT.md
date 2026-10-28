# ローカル実行環境の準備

既存の`conceptgraph-local`が動く場合は再構築せず、[RUNBOOK.md](RUNBOOK.md)から再開する。
以下は別PCや環境を作り直す場合の手順。ホストのDriver・CUDA・system Python・既存Conda環境を上書きしない。

## 検証済み構成

2026-10-25の実測値。ホストnvcc13.0とPyTorch CUDA runtime11.8は別物で、CUDA拡張をホストnvccで再ビルドしない。

|項目|実測|
|---|---|
|OS|Ubuntu 22.04.5 LTS / native Linux|
|GPU|NVIDIA RTX 6000 Ada Generation / capability 8.9 / 49140 MiB|
|Driver|580.196.07 (`nvidia-smi` の CUDA 表示は 13.0)|
|ホスト Toolkit|nvcc 13.0.88|
|専用環境|新規 `conceptgraph-local` / Python 3.10.22|
|PyTorch / torchvision|2.0.1+cu118 / 0.15.2+cu118|
|PyTorch CUDA runtime|11.8 (`torch.version.cuda`)|
|PyTorch3D|0.7.4 / 公式 py310_cu118_pyt201 wheel|
|Ultralytics / supervision|8.2.0 / 0.20.0|
|OpenCLIP / timm|2.24.0 / 0.9.16|
|Open3D / FAISS|0.18.0 / faiss-cpu 1.7.4|
|Rerun / Hydra|0.15.1 / 1.3.2|
|NumPy / OpenCV|1.26.4 / 4.9.0.80|
|OpenAI SDK / httpx|1.40.8 / 0.27.2 (M2でHTTPクライアントとしてのみ利用)|
|Ollama|0.30.10|
|既存モデル|qwen3.5:9b `56671c2ab938`; qwen3.5:35b `72aac7c28be8`|

## 専用環境

`conda env list`で同名が存在しないことを確認する。別名にする場合はenvironment.ymlのnameをコピー先で変更し、
RUNBOOKのactivateと`CONCEPTGRAPH_CONDA_ENV`を合わせる。システムコマンドとして`ollama`・`strace`・`bwrap`が必要。
GUIはデスクトップ接続、既存の2Dグラフ用`draw_relation_graph.py`はGraphvizの`dot`も使う。

```bash
conda env list
conda env create -f docs/local-only/environment.yml
conda activate conceptgraph-local
export PYTHONPATH=
export PYTHONNOUSERSITE=1
python -m pip install numpy==1.26.4 torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r docs/local-only/requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu118
python -m pip install -e .
python -m pip check
python scripts/local_only/m0_smoke.py
python -m pytest -q tests/local_only
```

m0_smokeの出力先は`outputs/local-only/m0/`。import・GPU算術・点群生成・CPU/GPU IoUを検査する。
既存のROS用PYTHONPATHやユーザーsiteの混入を避けるため、上の2つのexportを省略しない。
依存の固定値は`requirements-lock.txt`、直接依存の意図は`requirements.txt`。

## データとモデル

Gitに含まれない入力を、実行経路に合わせて用意する。

| 入力 | 標準配置・準備方法 |
|---|---|
| Replica room0 RGB-Dとpose | `outputs/local-only/assets/data/Replica/room0/`。初期取得は下のprepare_assets |
| YOLO-World／SAM／CLIP | `outputs/local-only/assets/weights/`と`clip_checkpoint.txt`。prepare_assetsが不足分を取得 |
| YOLO用CLIP ViT-B/32 | prepare_assetsがユーザーcacheへ準備。重みフォルダだけのコピーでは不足する場合がある |
| Qwen | 既存Ollamaモデル配置の`manifests/`と`blobs/`。9B/35Bの必要なタグがあることを確認 |
| 後処理用の完成マップ | RUNBOOKの`CG_MAP`。73物体マップと観測元RGBはローカル成果物で、cloneだけでは得られない |
| 公開main参照ソース | RUNBOOKの`CG_REFERENCE`。保存済みsnapshotを使う。なければ[公開mainのソース](https://github.com/concept-graphs/concept-graphs/blob/main/conceptgraph/scenegraph/build_scenegraph_cfslam.py)を初期準備で保存し、`CG_REFERENCE`をそのパスへ変更 |

```bash
# 初期セットアップだけネットワークを利用する。既存ファイルは再利用。
python scripts/local_only/prepare_assets.py --frames 100
# 全区間stride10の200枚には全2000組が必要。必要なときだけ取得範囲を拡大。
# python scripts/local_only/prepare_assets.py --frames 2000 --data-only
```

prepare_assetsは`clip_checkpoint.txt`へ実ファイルの絶対パスを記録する。別PCへコピーした場合は実配置に合わせて直すか、
`CONCEPTGRAPH_CLIP_CHECKPOINT`を指定する。元の取得先と入力hashは`outputs/local-only/assets/manifest.json`。
Qwenが既にある場合は再取得しない。新PCで未導入なら初期準備段階でモデルを取得してから、通常実行へ進む。

既存後処理の参照ソースhashは`7d748815a92ab606bcdde2dc0ee817836867042574797ffc41472215b0896841`（SHA256）。
取得URL・他の参照ソースhashは`outputs/local-only/paper_scenegraph_room0/attempt1/source_refs/sources.json`に保存済み。
最新mainは同じ内容とは限らない。既存snapshotを使えない試行は参照ソースの違いを区別する。

保存マップ・後処理manifest・cacheには絶対パスが含まれる。**別PCでファイルをコピーするだけの再開には対応していない**。
同じ配置を復元するか、移行先でmappingから作り直して新しい後処理rootをprepareする。
通常実行は暗黙のダウンロードやクラウド推論を許可せず、接続はローカルのみ。起動手順はRUNBOOKを参照。
