# 完全ローカル環境・M0 記録

検証日: 2026-10-25 (JST)。開始 commit: `5e83edd`。
作業ブランチ: `codex/local-only-conceptgraphs`。開始時の変更なし。

## 実測結果

**M0: PASS**。`outputs/local-only/m0/smoke.json` と `smoke.log` に証拠を保存。
`pip-check.txt`: `No broken requirements found.`

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

GPU 256×256 行列積の有限性と同期を確認 (checksum `4194197.5`)。
Open3D 点群生成に成功。PyTorch3D の実使用 API `box3d_overlap` をCPU/GPUの両方で
実行し同一立方体のIoU=1を確認。さらに実装の `compute_3d_iou_accurate_batch`
によるOpen3D頂点の並べ替え・拡張を通したGPU演算も合格。
対象 mapping モジュールと必須依存をすべて import 済み。

## M1前の技術選択

原実装のPyTorch3Dを維持する。公式配布の既存バイナリが取得でき、GPUで実測合格したため、
互換バージョン探索の段階で解決。Toolkit追加・ビルド・幾何アルゴリズム置換は不要。
`CUDA_HOME` は実測で未設定。ホストnvcc 13.0でcu118拡張をビルドしない。
CUDA runtime 11.8はPyTorch wheelが供給し、ホストDriver上で動作する。

参照: [PyTorch3D公式インストール手順](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)、
[公式wheel](https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt201/pytorch3d-0.7.4-cp310-cp310-linux_x86_64.whl)。

## 再現手順 (リポジトリルート)

既存同名環境を削除・上書きしないこと。`conda env list` に同名があれば、
`conceptgraph-local-YYYYMMDD` 等の新規名へ置き換える。

```bash
conda env list
conda env create -f docs/local-only/environment.yml
conda activate conceptgraph-local
# 継承されるROSパッケージとユーザーsiteをこのシェル内で隔離する。
export PYTHONPATH=
export PYTHONNOUSERSITE=1
python -c 'import os,sys; from pathlib import Path; assert Path(os.environ["CONDA_PREFIX"]).name.startswith("conceptgraph-local"); assert sys.prefix == os.environ["CONDA_PREFIX"]'
python -m pip install numpy==1.26.4 torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r docs/local-only/requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu118
python -m pip install -e .
python -m pip check
python scripts/local_only/m0_smoke.py
```

直接依存の意図は `requirements.txt`、推移依存とCLIPのcommitを含む実測固定値は
`requirements-lock.txt`。再現用Conda設定に個人の絶対パスを含めない。
元README全体の環境やセンサー・シミュレータ用依存は導入しない。

## 問題・警告・保護

- 初回pipがシェルのROS `PYTHONPATH`上の依存を「導入済み」と誤認した。
  import失敗を環境層と特定し、`PYTHONPATH=` / `PYTHONNOUSERSITE=1`付きで
  同一専用環境への依存導入を再実行した。解消後に上記スモークとpip checkを再実施。
  初回失敗も `pip-dependencies.log`、修正は `pip-isolated.log` に保持。
- 診断はセットアップ段階で実施した。モデル重み・Replicaの初期ダウンロードと、
  通常実行のネットワーク監査はM1以降で区別する。
- 既存Ollamaサービスの `/api/status` は `cloud.disabled=false`。
  このサービスのままではM2の本番推論を始めない。恒久設定を変更せず、
  クラウド無効を実測できる一時プロセス等で解決する必要がある。
- Driver / system CUDA / system Python / 既存Conda環境 / bashrc / Ollamaサービスは変更しない。
  `outputs/local-only/m0/protected-before.json` に保護ファイルのSHA256を保存。
- インストールコマンドのログ: `conda-create.log`, `pip-torch.log`,
  `pip-dependencies.log`, `pip-isolated.log`, `pip-editable.log`。
- 全Condaパッケージ・pip freeze・開始時GPU/モデル一覧も同じGit追跡外ディレクトリに保存。

M1/M2/M3はこのM0合格時点では未実行。各結果は `RUNBOOK.md` に追記する。
