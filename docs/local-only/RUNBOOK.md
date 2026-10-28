# ローカル実行：再開と新規試行

| やりたいこと | 読む箇所 | 必要なもの |
|---|---|---|
| 保存結果を見る | [1](#1-保存結果を見る推論不要) | 保存済みRRD、GUI |
| 同じマップでCaption・MST・関係推定を試す | [2](#2-保存マップから後処理を試す) | 73物体マップ、元RGB、参照ソース、Qwen35B |
| RGB-Dからmappingを作り直す | [3](#3-rgb-dからmappingを試す) | Replica、YOLO/SAM/CLIP重み。VLM有効時はQwen |
| 別PC・環境消失から準備する | [ENVIRONMENT.md](ENVIRONMENT.md) | 環境・データの準備 |

方式は2系統ある。**後処理**は完成済みマップの観測画像→Caption→要約→MST→関係推定。
**リアルタイム版**はフレーム画像から関係を求め、mappingの対応先へ結び付ける。
過去の件数・意味的な誤り・論文との差は[RESULTS.md](RESULTS.md)。原論文全体の忠実再現とは呼ばない。

## 0. 共通準備

各端末でリポジトリルートへ移動して実行する。以降はこのシェルの変数を使う。

```bash
conda activate conceptgraph-local
export PYTHONPATH=
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
CG_SAVED=outputs/local-only/paper_scenegraph_room0/attempt1
CG_DATA=outputs/local-only/assets/data/Replica
CG_MAP="$CG_DATA/room0/exps/room0_reference_graph_attempt1/pcd_room0_reference_graph_attempt1.pkl.gz"
CG_REFERENCE="$CG_SAVED/source_refs/official_main_scenegraph.py"
command -v python conda bwrap strace rerun ollama
```

保存先やデータが違う場合は上の変数を変更する。表示にはデスクトップ接続が必要。
既存環境が使えるか不明なら`python -m pytest -q tests/local_only`を実行する。
データやモデルが欠けている場合は[環境準備](ENVIRONMENT.md#データとモデル)へ進む。

## 1. 保存結果を見る（推論不要）

```bash
# 最終グラフ：RGB・切り出し・2Dグラフ・3D・説明
bash scripts/local_only/view_paper_scenegraph.sh "$CG_SAVED" full_main
# MST前→MST後→LLM入力→判定→最終グラフ
bash scripts/local_only/view_paper_scenegraph.sh "$CG_SAVED" full_main candidate_stages.rrd
```

どちらか一つを選ぶ。`candidate_step`は0〜4が全体段階、5〜25が個別候補、26が案内。
実入出力は`$CG_SAVED/full_main/evidence.md`、`relations.jsonl`、`object_summaries.jsonl`。
RRDだけなら元画像・Qwenは不要。静的グラフの表示時刻は関係の成立時刻ではない。

RRDがなくても推論結果と準備データがあれば、次で検証・再生成できる（レポートと表示ファイルを更新）。

```bash
python scripts/local_only/run_paper_scenegraph.py verify --root "$CG_SAVED" --run full_main
python scripts/local_only/run_paper_scenegraph.py render --root "$CG_SAVED" --run full_main
python scripts/local_only/run_paper_scenegraph.py render-candidates --root "$CG_SAVED" --run full_main
```

`verify`は`status: PASS`を確認する。`full_main`が最終版で、`full`はBBox修正前の中間結果。
GUIとrenderは専用設定・bwrapで通信を隔離する。bwrapなしでは停止する。

## 2. 保存マップから後処理を試す

この経路はYOLO・SAM・CLIP・3D融合を再実行しない。元マップに保存されたRGBパスが現在も読める必要がある。
マップ・元RGB・参照ソースを確認し、新しい試行先を作る。

```bash
test -f "$CG_MAP" && test -f "$CG_REFERENCE" && test -f "$CG_DATA/room0/results/frame000000.jpg"
mkdir -p outputs/local-only/trials
CG_TRIAL=$(mktemp -d outputs/local-only/trials/paper-XXXXXX)
printf '今回の試行先: %s\n' "$CG_TRIAL"
python scripts/local_only/run_paper_scenegraph.py prepare \
  --root "$CG_TRIAL" --map "$CG_MAP" --reference "$CG_REFERENCE"
```

`prepared: true`と`$CG_TRIAL/contact_sheet.jpg`を確認する。
prepareは全物体から最大10観測を準備し、下のpilotだけ推論対象を5物体・最大3観測へ絞る。
現在の73物体マップでは全体384画像。別のマップでは`--indices`の物体indexを選び直す。

### 一時Ollamaを起動（別端末）

同じ共通準備を行い、**既存モデルがあるディレクトリ**を指定する。このPCで確認済みの配置は次のとおり。

```bash
export CONCEPTGRAPH_OLLAMA_MODELS=/usr/share/ollama/.ollama/models
# 別配置なら上を書き換える。manifests/ と blobs/ が必要。
test -d "$CONCEPTGRAPH_OLLAMA_MODELS/manifests"
test -d "$CONCEPTGRAPH_OLLAMA_MODELS/blobs"
mkdir -p outputs/local-only/trials
CG_SERVER=$(mktemp -d outputs/local-only/trials/ollama-XXXXXX)
bash scripts/local_only/serve_ollama.sh "$CG_SERVER"
```

起動した端末はそのままにする。既存11434サービスは使わず、空いている11435で起動する。
終了はこの端末でCtrl-C。共有モデル・通常サービス・他のGPUプロセスを削除／停止しない。

### 小さく試す→全体へ進む（元の端末）

```bash
curl --fail --noproxy '*' http://127.0.0.1:11435/api/status
OLLAMA_HOST=127.0.0.1:11435 ollama list
```

`cloud.disabled=true`とローカルの`qwen3.5:35b`を確認してから実行する。

```bash
python scripts/local_only/run_paper_scenegraph.py infer --root "$CG_TRIAL" \
  --run pilot --indices 2 7 12 13 19 --max-observations 3 --model qwen3.5:35b
python scripts/local_only/run_paper_scenegraph.py verify --root "$CG_TRIAL" --run pilot
python scripts/local_only/run_paper_scenegraph.py render --root "$CG_TRIAL" --run pilot
bash scripts/local_only/view_paper_scenegraph.sh "$CG_TRIAL" pilot
```

`verify`のPASSと画像・Caption・関係方向を確認する。API／構文の成功は意味的な正解を保証しない。
同じ元の端末で全対象へ進むなら次を実行する。pilotで取得済みの画像Captionはcacheで再利用する。

```bash
python scripts/local_only/run_paper_scenegraph.py infer --root "$CG_TRIAL" \
  --run full --max-observations 10 --model qwen3.5:35b
python scripts/local_only/run_paper_scenegraph.py verify --root "$CG_TRIAL" --run full
python scripts/local_only/run_paper_scenegraph.py render --root "$CG_TRIAL" --run full
python scripts/local_only/run_paper_scenegraph.py render-candidates --root "$CG_TRIAL" --run full
bash scripts/local_only/view_paper_scenegraph.sh "$CG_TRIAL" full
```

進捗は`$CG_TRIAL/<run>/state.json`、完了集計は`report.json`、最終グラフは`scene_graph.json`。
根拠は`evidence.md`、3タスクの実応答は`captions.jsonl`・`object_summaries.jsonl`・`relations.jsonl`、cacheは`$CG_TRIAL/vlm_cache/`。
同じroot・run・設定でinferを再実行すると有効cacheを再利用する。設定やモデルを替えるときはrun名を変える。
新規推論だけで比較したい場合は新しい`CG_TRIAL`を作る。CLI既定のroot/runは既存結果を指すため、省略しない。

## 3. RGB-Dからmappingを試す

モデル重みと`outputs/local-only/assets/clip_checkpoint.txt`が必要。以下の3枚検証はQwen不要。

```bash
mkdir -p outputs/local-only/trials
CG_MAPPING=$(mktemp -d outputs/local-only/trials/map-XXXXXX)
CG_EXP=$(basename "$CG_MAPPING")
python scripts/local_only/inspect_replica.py "$CG_DATA" --end 3 --output "$CG_MAPPING/input.json"
bash scripts/local_only/run_mapping.sh "$CG_MAPPING" \
  exp_suffix="$CG_EXP" end=3 stride=1 save_objects_all_frames=false
python scripts/local_only/validate_map.py \
  "$CG_DATA/room0/exps/$CG_EXP/pcd_$CG_EXP.pkl.gz" --no-vlm
```

`inspect_replica`と`validate_map`のPASS、`$CG_MAPPING/network-audit.json`を確認する。
変更可能な項目は`end`（終端、含まない）・`stride`・`exp_suffix`。別試行では上のmktempからやり直す。
独自のデータ配置は`CONCEPTGRAPH_DATA_ROOT`（Replicaの親）、重みは`CONCEPTGRAPH_MODEL_ROOT`、
CLIPは`CONCEPTGRAPH_CLIP_CHECKPOINT`、Conda環境名は`CONCEPTGRAPH_CONDA_ENV`で指定する。
上の`CG_DATA`は入力検査用なので、独自配置では`CONCEPTGRAPH_DATA_ROOT`も合わせる。

VLM付きの3枚試行は、別端末の一時Ollamaを起動してから**新しい**試行先・実験名で次を実行する。

```bash
CG_GRAPH=$(mktemp -d outputs/local-only/trials/graph-XXXXXX)
CG_EXP=$(basename "$CG_GRAPH")
bash scripts/local_only/run_mapping.sh "$CG_GRAPH" \
  --config-name=rerun_local_scene_graph exp_suffix="$CG_EXP" end=3 stride=1 \
  vlm.base_url=http://127.0.0.1:11435/v1 vlm.model=qwen3.5:35b
python scripts/local_only/validate_map.py \
  "$CG_DATA/room0/exps/$CG_EXP/pcd_$CG_EXP.pkl.gz" --require-edges
```

地図・元RRD・`associations.jsonl`は`$CG_DATA/room0/exps/$CG_EXP/`。ログは`$CG_GRAPH/mapping.log`。
関係0件では`--require-edges`が失敗する。保存不整合と、推論が関係を返さなかったケースをログで分ける。
100枚へ拡大する場合は新しい実験で`end=100 stride=1`、全区間200枚なら`end=-1 stride=10`を明示する。
後者には全2000組のRGB-Dが必要。旧2000枚ジョブや`run_full_room0.py`の通常モードは自動で再開しない。

## 困ったとき

| 症状 | 確認するもの |
|---|---|
| import失敗／コマンドなし | 共通準備、ENVIRONMENTの依存、選択中Conda環境 |
| 元画像が見つからない | mapの`color_path`と実データ。別PCへのコピーだけでは絶対パスは直らない |
| 11435が使用中 | 既存の所有プロセスを確認。勝手に停止せず、再利用するならcloud無効を確認 |
| settings changed | 設定変更は新run、入力／prepare条件変更は新root |
| infer中断 | `state.json`と`vlm_cache/failed_*.json`。同じコマンドでcache再開 |
| Caption欠落・関係の逆向き | 実応答と理由文。形式検証のPASSとは別に確認 |
| mappingが止まった | `mapping.log`、`network.jsonl`、`network.strace`。融合は先頭から再計算 |
| GUIが開かない／重い | デスクトップ接続、bwrap、ソフトウェア描画。まず`diagram.png`や`evidence.md`で確認 |

通常のリアルタイムVLM方針は`upstream`で、不正項目を記録・除外し欠落Captionをnull補完する。自動再生成0回。
`completion.json`の`vlm`と`rejected_*.jsonl`・`missing_captions.jsonl`を確認し、`completed_with_vlm_issues`を全件正常と扱わない。
