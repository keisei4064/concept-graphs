# Codex 実装指示書：ConceptGraphs の完全ローカル実行

作成日: 2026-10-24  
対象リポジトリ: https://github.com/keisei4064/concept-graphs  
作業ブランチ: `codex/local-only-conceptgraphs`（`ali-dev`起点）  
主対象実装: `conceptgraph/slam/rerun_realtime_mapping.py`

## 0. ゴール・優先順位

**主目的：既存の ConceptGraphs (`ali-dev`) を、クラウド推論を一切使用せず、研究室PC上で再現実行する。** まず Replica の posed RGB-D データで object-centric 3D mapping を再現し、次いで Ollama による画像キャプション・物体間関係推定を組み込み、Scene Graph の表示と保存まで確認する。

優先順位は **原実装への忠実さ ＞ 動く最小構成 ＞ 安全・再現性 ＞ 性能・高速化**。新しいアルゴリズムの開発、物体対応付けや融合の改善、RDF/WorldGraphへの接続は今回の対象外。

- **M0**: 環境診断・依存関係インストール、GPU算術と必要パッケージの確認
- **M1**: VLM無しで Replica `room0` の検出→逆投影→Association/Fusion→物体地図の保存・Rerun可視化
- **M2**: Ollama 上の Qwen3.5 9B / 35B を使用して、キャプションと関係推定を独立に検証
- **M3**: 完全ローカルの一連の処理で Scene Graph を生成・保存・可視化

**M1の合格を確認してからM2/M3へ進むこと。** 失敗の際は「環境 / 検出 / 幾何 / 対応付け / VLM / シリアライズ」のどの層かを特定する。失敗した工程を成功扱いせず、再現手順と未解決事項を記録する。

## 1. ハードウェア・ソフトウェアの既知条件

研究室PC（ユーザーが実行結果を提供済み）:

- Ubuntu 22.04.5 LTS (native Linux)
- NVIDIA RTX 6000 Ada Generation, 48GB (nvidia-smi 表示総量 49140 MiB)
- Driver 580.196.07; `nvidia-smi` 表示 CUDA 13.0
- ホストの `nvcc` は CUDA Toolkit 13.0 (V13.0.88)
- Python 3.10.12, Conda 25.11.0
- 調査時VRAMは約8040MiB使用、約40GiB空き（変動する）。GPUはデスクトップ用途にも使用中。
- **Ollama に `qwen3.5:9b` と `qwen3.5:35b` の両方をダウンロード済み**（`ollama pull` がそれぞれ success）。推論実行、Vision入力、GPU全面ロードは**未検証**。不要な再ダウンロードをしないこと。

注意: `qwen3.5:35b` は 35B-A3B MoE 系列の Ollama タグ。初回は9Bで接続とパーサ検証し、次に35Bを比較する。量子化方式の強制・独自指定はしない（既存のデフォルトモデルを使う）。

### 最優先の安全制約

1. **ホストの NVIDIA Driver / システム CUDA Toolkit / system Python / 既存Conda環境を絶対に変更しない。**
2. `sudo apt install`, `apt upgrade`, `sudo pip`, `conda update --all`, グローバル `pip install`, 既存 `~/.bashrc` の書換え、Ollamaサービスの恒久的変更は禁止。必要なら理由と代替案を報告する。
3. 専用の新規Conda環境 `conceptgraph-local` を使う。既存同名環境があれば**削除・上書き禁止**で、新規名に切り替え報告する。
4. Conda内のCUDAランタイムとホスト上のCUDA 13.0を混同しない。`torch.version.cuda`、`nvcc --version`、`CUDA_HOME`、GPU capability を明示検証する。古いCUDA拡張をビルドする場合、Toolkitのバージョン整合を検証する。ホストの `/usr/local/cuda` を変更しない。
5. 原則ユーザー確認なしの破壊的操作をしない。既存のファイル、データセット、モデル、GPUで実行中の他プロセスを消去・停止しない。
6. **クラウドへの推論リクエスト厳禁。** OpenAI・Anthropic・Ollama Cloud・Hugging Face hosted inference など、外部推論APIを使わない。依存物や重み、Replicaデータセットをダウンロードするネットワーク通信は初期セットアップに限り許可。通常実行時の通信はローカルループバックだけに制限する設計とし、実測で確認する。キーの設定を要求・保存しない。
7. 設定変更や追加コードはこの作業ブランチだけにコミット。`ali-dev` / `main` を触らない。大規模リファクタリングや他ブランチへのPR/マージはしない。
8. 各修正は原処理を保つ小さな差分にする。テストを通すためだけの空出力・例外握りつぶしは禁止。

## 2. ブランチ / 作業場所

本リポジトリをcloneしていなければ:
```bash
git clone https://github.com/keisei4064/concept-graphs.git
cd concept-graphs
git fetch origin
git switch codex/local-only-conceptgraphs
git status --short
```
既にcloneしてある場合は既存の未コミット変更を保存・確認してから当該ブランチへ移ること。既存変更を勝手に破棄しない。ブランチは既に作成済みで、この指示書が含まれている。

実験データ・重み・ログ・Conda環境は原則Git追跡外（既存データは再利用）。設定の機密値や個人固有の絶対パスはコミットしない。パスは環境変数、gitignore済みのローカルoverride、またはCLI overrideで指定できるようにする。

## 3. リポジトリ固有の確認ポイント（事前調査済み）

**注意: 論文の処理と `ali-dev` の実装は完全に同一ではない。** 本タスクは `ali-dev` を再現するものであり、論文版の後処理型MSTエッジ推定パイプラインへの全面置換はしない。

対象の実体：
- `conceptgraph/slam/rerun_realtime_mapping.py`：入力、YOLO-World + Ultralytics SAM、OpenCLIP、3D mapping、VLM・Rerun・保存の統合
- `conceptgraph/utils/vlm.py`：OpenAI client生成、画像キャプション、画像関係推定、キャプション統合。既定 `gpt_model = "gpt-4o-2024-05-13"`
- `conceptgraph/utils/general_utils.py`：`make_vlm_edges_and_captions`、検出結果保存、その他
- `conceptgraph/slam/utils.py`：mask/depth→3D点群、物体と検出結果のデータ構造、filter
- `conceptgraph/slam/mapping.py`：幾何・視覚的類似度、マッチング・融合
- `conceptgraph/hydra_configs/rerun_realtime_mapping.yaml`：現状 defaults は **`record3d`**、`make_edges: true`。Replica用には上書きが必要
- `conceptgraph/hydra_configs/base_paths.yaml`：開発者固有の絶対パスが入っている。個人パスで上書きコミットしない
- `conceptgraph/hydra_configs/replica.yaml`：`room0`入力
- `conceptgraph/hydra_configs/early_exit.json`：中断後保存を試みる機構

既知のVLM無効化バグ候補（実行とテストで検証して修正）:
- `make_vlm_edges_and_captions(..., make_edges_flag=False)` で `captions` が未定義のまま返る経路
- `rerun_realtime_mapping.py` は `make_edges=False` でも `get_openai_client()` を呼ぶ
- 終了時に `consolidate_captions()` が無条件に呼ばれる
- `filter_gobs` および `make_detection_list_from_pcd_and_gobs` は `captions` を参照するため、VLM無効でも **物体数・indexと整合する空キャプション**が必要。単純に `[]` を返せば済むとは限らない。検出ラベルごとの `{"id": "...", "name": "...", "caption": null}` 相当を整合した順序で用意しテストすること。
- VLM無効時でも `gobs["edges"]` は空リスト、`captions` は各検出に対応したスキーマに合わせる。
- `process_edges` がVLMなしで破綻しないことを確認する。

`make_edges=True` の `ali-dev` は主として**フレーム画像上の注釈付き物体からVLMで関係・キャプションを取得し、対応付け結果と結び付ける**仕組み。論文のMSTによる3Dペア候補生成と混同しないこと。

## 4. M0 — 実環境を診断し、専用環境で依存を解決

1. `git status`, `nvidia-smi`, `nvcc --version`, `python3 --version`, `conda --version`, `ollama --version`, `ollama list` を確認。
2. PyTorchとPyTorch3D、Ultralytics、OpenCLIP、Open3D、FAISS、Rerun、Hydra等の実コードがimportする依存を洗い出す。READMEは古い。セットアップ・動作状況から必要最小セットを確定。
3. Conda `conceptgraph-local` でPython 3.10。READMEにある PyTorch 2.0.1 + CUDA 11.8 は**出発点**であって絶対必須ではない。RTX 6000 Ada (SM 8.9) と NVIDIA Driver 580 を考慮し互換性を実測する。
4. PyTorch3Dのビルド済みwheel/conda packageが利用できない、あるいはCUDAと衝突する場合は、次の順で検討: (a)互換バージョン探索、(b)隔離環境内のToolkitでのビルド、(c)依存箇所の最小置換。**M1開始前に技術的選択を文書化する**。無関係なシステム変更はしない。
5. `python -m pip install -e .` は**専用Conda環境の有効化を確認してから**実施。必須importと少量GPU行列積、Open3D点群生成、PyTorch3Dの実使用APIをスモークテスト。
6. すべてのバージョン、ドライバ、設定値、警告を `docs/local-only/ENVIRONMENT.md` に記録。再現用 `environment.yml` または明確なsetup手順を用意（グローバル絶対パスは含めない）。

**M0合格**: 既存環境に非破壊で必要ライブラリをimportでき、GPU上の演算が完走。失敗時は原因・代替案とログを残す。

## 5. M1 — VLMなし Object Mapping

### 入力
Nice-SLAM公開のReplica RGB-D sequences の `room0`。RGB、depth、カメラ内部行列、姿勢が取得できること。既存ローカルデータの有無を確認して再利用し、ない場合のみダウンロード案内/取得。処理量を抑えて最初はごく少数のフレーム（stride/endまたは明示的なフレーム数）から始める。対象のRGB-D整列・深度単位・カメラ内部行列・pose座標系を検査する。

### 実装
- Hydraの新規実験用設定（例: `conceptgraph/hydra_configs/rerun_local_replica.yaml`）を追加する。元の `rerun_realtime_mapping.yaml` の破壊的な編集は避け、`replica`、`make_edges=false`、`use_rerun=true`、`save_rerun` の挙動確認、入力フレーム範囲、ユニークな `exp_suffix` を指定。
- クラウドVLMクライアントを生成せずに実行できるようにする。空キャプションと関係のスキーマが検出数・物体更新処理と一致することをテスト。
- YOLO-World、SAM、OpenCLIP、mask→3D、Association/Fusion、denoise/mergeの原実装を保つ。モデル重みは初期セットアップでローカルキャッシュへダウンロード（存在すれば再利用）。実行中の暗黙ダウンロードについては記録し、最終テストではローカルキャッシュ利用を確認。
- 3D Object Mapの可視化と保存を確認。Rerunのタイムライン、物体ID、点群、Bounding Boxを確認。GUI不可時はRerunファイルやOpen3D等の代替検証を行い、その区別を明記。
- 結果 `pkl.gz` を再ロードして、非空の物体リスト、点群形状・位置・特徴ベクトル、対象フレーム数・関連検出数を確認。物体ID対応付けのログが確認できるようにする。

**M1合格**: 外部VLM/APIキー不要で少数フレームを完走し、非空の物体地図を保存・再ロード・可視化できる。100フレーム程度への拡大は短時間のスモークテスト後に行う。

## 6. M2 — Ollama VLM 単体

既にローカルに存在する `qwen3.5:9b`, `qwen3.5:35b` を使う。モデルは明示的な利用許諾・再ダウンロードなしで使用する。最初に `ollama list`, `ollama ps`, `nvidia-smi` で実際の可用性を調べる。

### 設計
- `conceptgraph/utils/vlm.py` を大改造せず、OpenAI互換の **Ollamaローカルendpoint**（デフォルト `http://127.0.0.1:11434/v1`）とモデル名を設定可能にする小さなAdapter/Factoryを導入。
- OpenAI SDKをHTTPクライアントとして再利用してもよいが、ローカルendpoint固定であることを検証し、意図しない `api.openai.com` フォールバックを作らない。APIキーはダミーで足りる構成を使い、実鍵を読み取ったり送信したりしない。
- **Fail closed**: Local onlyモード中、endpointが loopback 以外だったら起動時エラー。Ollamaに接続できなければ明確なエラーを返し、外部APIへのフォールバックは厳禁。
- 画像は既存と同様、ローカル画像をOpenAI互換vision messagesの `image_url: data:image/jpeg;base64,...` で送信。Qwen3.5実行時に互換性を実測。
- GPT-4o用の関係推定・キャプション・キャプション統合という**3つのタスクを保つ**。各タスクのプロンプト、出力スキーマ、モデル別実際のレスポンスを確認。
- 既存の `extract_list_of_tuples`, `vlm_extract_object_captions` は自由形式の文字列parseに依存する。可能ならモデル対応のJSON schema/structured outputsを導入。ただし最小変更で実装し、出力を検証した後に既存が期待する形へ変換する。
- 関係端点の番号・ラベル、物体IDが有効なものだけ採用し、存在しないID、自己ループ、不正なRelation、parseエラーを**記録して明確に扱う**。単に空のグラフへ落とすサイレントな例外握りつぶし禁止。
- モデル応答のtemperature、プロンプト、出力JSON、所要時間、入力画像、モデル名を後から比較できる形で保存（大量ログ・base64画像本文を標準出力に垂れ流さない）。
- **9B→35B** の順で、保存した同一の注釈画像に対して単体テストを行う。推論がGPUで実施されていることを `ollama ps` / `nvidia-smi` で確認し、VRAMに余裕がなければ3D処理とVLM処理を段階実行。

Ollamaのクラウド機能オフ設定は、既存のサービス設定を勝手に変更せず、既存サービスがローカルのみで動作することを確認する。オフライン保証を満たせなければ**本番テストを開始せず**、サービス管理方法に応じて非破壊の設定提案を記す。環境変数 `OLLAMA_NO_CLOUD=1` は、プロセス側に設定しなければ効かないことに注意（呼び出し側のシェルだけではサービス側に伝わらない場合がある）。

**M2合格**: 画像入力で有効なCaption / Relation / 統合Captionを得られ、9Bと35Bの結果・遅延を保存し、ローカルendpointのみを使ったことを示せる。

## 7. M3 — End-to-end Scene Graph

- Phase 1で動作確認したReplica入力と、Phase 2のVLM Adapterを統合し、`make_edges=true` で実行。
- 当面は `ali-dev` の既存設計（2Dフレーム関係→3D物体対応→map_edges）を維持し、**元の論文のMSTアルゴリズムと混同・混成しない**。
- 生成されたノード数/エッジ数、エッジの両端IDが存在すること、キャプションが元の検出対象と整合すること、関係ラベル・方向をチェック。
- 検出とマッピングのキャッシュを再利用してモデルを切り替えられるようにする。保存済み検出結果にVLMフィールドがなくても、検出の再実行を必要としない実験設計が可能か調査し、最小変更で対応可能なら実装。高コストなら制約を明示し無理に書き換えない。
- Rerunまたは既存可視化で、3Dマップ上の物体・関係エッジを確認。可視化できることと、関係が意味的に正しいことを別の判定項目にする。
- 異なるモデルで同じ入力を使う比較は小規模に留め、恣意的な精度主張はしない。人手確認した数件で物理的に誤った関係や未観測関係の幻覚を例示可能なら記録。
- 実験条件・実行コマンド・設定スナップショット・成果物パスと再現手順を `docs/local-only/RUNBOOK.md` にまとめる。

**M3合格**: 追加のクラウドAPI無しでReplicaから非空のObject Mapと関係付きScene Graphを作り、保存・再ロード・可視化できる。少なくとも1つのVLMモデルで完走。可能なら9B/35Bを同一入力で比較。

## 8. 必須テスト・検証観点

テスト用コードは小さく保ち、以下を必ず実施する。

1. **無VLMモード**: `make_edges=false` でOpenAI APIキー無し、Ollama停止中でもObject Mappingを完了できる。キャプション要素の数・形式が後続コードの契約を満たす。
2. **完全ローカル制約**: Local-only モードで誤設定された外部URLを拒否し、接続失敗時に外部へフォールバックしない。外部APIにHTTPリクエストを送っていないことを実測できるログまたはネットワーク監査で確認（初回セットアップ時の配布元DLは区別）。
3. **マップの整合性**: 点群座標・入力深度スケール・内部行列・姿勢が妥当。保存物をロードして、非空・有限値・ID一意性・参照整合性を点検。
4. **VLMインターフェース**: 画像付きの実際のOllama応答と、破損JSON・不正ID・API停止を模擬した単体テスト。例外を検出・記録できる。
5. **Rerun**: 物体/関係が表示される。GUIが無い状況でも保存と構造検証を代替として実施し、その限界を明記。
6. **再現性**: 使用した commit、Conda packages、モデルタグ・digestが分かればdigest、閾値、stride、推論設定、コマンドと出力場所を記録。
7. **安全性**: 既存Conda/env/host Driver/CUDAが変更されていないこと。インストールコマンドと変更ファイル一覧を報告。

ネットワーク監査は単なる `OPENAI_API_KEY` 未設定だけをもって合格にしないこと。実際に endpoint、通信先、モデル重みDL、ライブラリによる暗黙の外部リクエストを確認する。

## 9. 作業の進め方 / 報告ルール

- 最初に `git status` と現在環境を観察し、作業ブランチへ移動。元の変更は保持。
- 各 M0/M1/M2/M3 が完了するたびに、**実施内容・変更ファイル・実行コマンド・成功証拠・課題**を簡潔にまとめたコミットを作る。
- 長大な学習・検出ログのターミナル全文をコンテキストへ垂れ流さない。ログはファイルへ保存し、通常は終了コード・tail・エラー抽出・小さなサマリーだけ表示する。
- 時間のかかる処理やモデルロード時は保存先・PID・リソースを記録。失敗時にむやみに全工程をやり直さず、キャッシュを活かし障害範囲を切り分ける。
- 少数フレームで合格→フレーム増加→35Bの順。初回から大規模データ処理をしない。
- 実機センサー、BEHAVIOR/OmniGibson、RDF、WorldGraphとの接続は今回実装しない。
- 最終報告では完了したM0-M3、失敗した項目、必要な手動操作、代表的出力・実験パス、コミットSHAを明示。実行できなかった工程を「動作確認済み」としない。

## 10. 期待する納品物

- 本ブランチ上でコミットされた、再現可能なローカルConceptGraphs環境・必要最小限の修正。
- `docs/local-only/ENVIRONMENT.md`: 依存、バージョン、GPU環境、制約と再現法
- `docs/local-only/RUNBOOK.md`: M1 / M2 / M3 のコマンド、成果物、トラブル対応
- ローカルReplica用Hydra構成（パスをハードコードしない）
- Ollama Adapterまたは既存VLMモジュールへの局所的な変更
- VLMなし／VLMあり／パーサ異常系のスモーク・単体テスト
- 実験ログとモデル比較の簡潔なレポート（重い生成物はGitにコミットしない）

## 11. 開始前の補足

ユーザーから追加確認を待たずに、上記の範囲は調査・実装・検証してよい。ただし root権限の変更、システム環境改変、既存環境の破壊、秘密情報の利用、他ブランチ変更、外部API推論、従量課金が発生する操作は行わない。安全な方法で完遂できない場合は停止点と理由、必要なユーザー操作を報告する。

**開始してください。まずM0の環境診断と依存関係の実装状況の調査から着手してください。**
