# 完全ローカル実行手順・実験記録

リポジトリルートから実行する。専用環境の再現は [ENVIRONMENT.md](ENVIRONMENT.md)。
指定ブランチ以外では作業しない。`outputs/` はGit追跡外で、データ・重み・生ログを置く。
セットアップの配布元ダウンロードと、通常推論の外部通信禁止は別の段階として扱う。

## M1: VLMなしのReplica

初期セットアップ (ネットワーク利用可、既存ファイルは再利用):

```bash
conda activate conceptgraph-local
export PYTHONPATH=
export PYTHONNOUSERSITE=1
export YOLO_CONFIG_DIR="$PWD/outputs/local-only/yolo-config"
python scripts/local_only/prepare_assets.py --frames 100
python scripts/local_only/inspect_replica.py outputs/local-only/assets/data/Replica --end 3
python -m pytest -q tests/local_only
```

ReplicaはNICE-SLAM公開のposed RGB-Dをそのまま使用する。HTTP Rangeで`room0`の
先頭100組と`traj.txt`を取り出し、12.4GBの全シーンアーカイブのダウンロードを避ける。
既存データを使用する場合はセットアップのデータ取得を省略し、下の環境変数で指定する。
セットアップ中のディレクトリを入力検査・mappingが同時に読まないこと。

```bash
# 通常実行: クライアント側はループバック以外への接続/DNSを拒否。
# 3フレームの合格を確認してから100フレームへ拡大する。
bash scripts/local_only/run_mapping.sh outputs/local-only/m1/smoke \
  exp_suffix=m1_smoke end=3 stride=1
python scripts/local_only/validate_map.py \
  outputs/local-only/assets/data/Replica/room0/exps/m1_smoke/pcd_m1_smoke.pkl.gz
RERUN_ANALYTICS_ENABLED=false rerun print \
  outputs/local-only/assets/data/Replica/room0/exps/m1_smoke/rerun_m1_smoke.rrd \
  > outputs/local-only/m1/smoke/rerun-structure.txt

# smokeが合格した後のみ
bash scripts/local_only/run_mapping.sh outputs/local-only/m1/100frames \
  exp_suffix=m1_100frames end=100 stride=1
```

独自の既存データ・モデル位置は `CONCEPTGRAPH_DATA_ROOT` (その下に`Replica/room0`)、
`CONCEPTGRAPH_MODEL_ROOT`、`CONCEPTGRAPH_CLIP_CHECKPOINT` で上書きできる。
Conda環境を新規別名にした場合は `CONCEPTGRAPH_CONDA_ENV` を指定する。
既存の成功結果を保持するには `exp_suffix` とログディレクトリを新しい名前にする。

主な成果物:

- `.../room0/exps/<exp_suffix>/pcd_<exp_suffix>.pkl.gz`: 元実装形式の点群・特徴・物体ID・関係。
- `.../rerun_<exp_suffix>.rrd`: viewerを起動せず保存するRerun記録。
- `.../config_params*.json`: 解決済み設定、`run_metadata.json`: フレーム数と検出数。
- `.../associations.jsonl`: 検出番号→物体UUIDと新規/融合の対応。
- `.../local_detections_<start>_<end>_<stride>/detections/`: 検出・SAM mask・OpenCLIP特徴キャッシュ。
- `outputs/local-only/m1/<run>/`: 実行ログ、PID、開始時刻、commit、override、GPU前後状態。

`network.jsonl`はPythonの接続・名前解決の許可/拒否を記録する。
`network.strace`はネイティブライブラリと子プロセスを含む実際のネットワークsyscallを記録し、
`network-audit.json`が外部通信先を検査する。Ultralyticsのimport時の到達性確認などが
拒否ログに現れても外部へのsyscallは許可しない。初期ダウンロードはこの通常実行に含めない。
PythonガードはOSネットワーク名前空間の代替ではないため、ネイティブ通信の実測監査も必須。

可視化は `rerun <file.rrd>` で開く。自動検証では保存と`rerun print`による構造検査を行い、
GUIの目視確認と区別して記録する。Rerunは0.15.1で再生する。

## 失敗の切り分け

- 環境: ROSの`PYTHONPATH`混入を避け、M0 smoke / `pip check`を再確認。
- 入力・幾何: RGB/depth組数、`traj.txt`、6553.5のdepth scale、内部行列、c2wを検査。
- 検出: 重みの存在を確認。通常実行での暗黙ダウンロードは許可しない。
- 対応付け: `associations.jsonl`と元実装の閾値を確認。アルゴリズム調整で誤魔化さない。
- シリアライズ: `validate_map.py`で非空・有限値・UUID・caption数を検査。
- VLM: M1合格までは実行しない。M2で接続と異常系を独立検証してからM3へ進む。

## 実行結果

M0: 合格、commit `b5d041d`。詳細はENVIRONMENT.md。
M1 少数実走: 3/3フレーム完走、17物体、61関連検出、特徴1024次元。
`m1_smoke` のmap保存・再ロード、有限点群/特徴、UUID一意性、caption整合性、
Rerunのframeタイムライン/点群/Bounding Box/uuidの構造検査に合格。
キャプションはすべてnull、関係は空。APIキーは実行スクリプト内で未設定にする。
`m1/smoke/network-audit.json`: PASS、インターネット/ループバック宛てconnect/send先ともゼロ。
GUIの目視確認は未実施で、Rerun保存・構造検査を代替とする。
全100フレームのRGB-D/c2w/内部行列/深度スケール/投影往復検査も合格。
**M1: 合格**。100/100フレームのmappingも完走し、25物体・1980関連検出を保存・再ロード。
全100フレームが保存物の関連検出に含まれる。特徴・点群・ID・null captions・空edgesの検査に合格。
`m1_100frames` のRerun記録は597,807,630 bytesで、`rerun print`が正常読込。
`m1/100frames/network-audit.json`もPASS、通信先なし。mappingループは約213秒。
最終denoise/filter/mergeと明示的Rerun closeを含めて実行した。
単体テストは7件合格 (`outputs/local-only/m1/unit.log`)。
実験入力と4つの重みファイルのSHA256は `outputs/local-only/assets/manifest.json` に保存。
M2/M3: M1合格を確認済み、これから実装・検証する。

観測した問題と対応 (失敗を合格として扱わない):

- 環境層: M0初回のROS依存混入を隔離し、import/GPU smokeを再実施して合格。
- 入力層: 初回の取得中データ検査はRGB/depth数不一致で失敗。
  取得完了後の3フレーム/100フレーム検査に合格。データそのものは変更していない。
- 設定層: モジュール入口ではHydraのconfig packageに`__init__.py`が必要だった。
  追加後、解決済みReplica構成を検証。
- 実装層: `MapObjectList(device=...)` はlistの未対応引数だったため対象scriptだけ修正。
  VLM無効時のmerge戻り値のunpackも型契約どおりに修正。幾何/Association/Fusionの計算は維持。
- シリアライズ層: Rerun 0.15.1にグローバル`flush()`は存在しない。
  初回3フレームの記録はプロセス終了時に保存され構造検査に合格したが、
  明示的なファイルcloseは`disconnect()`へ修正し、100フレーム実走で検証する。
- supervision 0.20のColorPalette/BoxAnnotator非推奨警告は記録。
  現バージョンでは機能し、無関係なAPI更新は行わない。
