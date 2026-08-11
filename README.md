# attic-shogi

将棋エンジン [YaneuraOu](https://github.com/yaneurao/YaneuraOu) の Rust
移植です。後述の相違点を除いて移植しています。

移植元の上流コードは YaneuraOu のコミット
`76d58ef2e4cf64116784f41fd4816425ab6817ee` (V9.70) の状態を基準にしています。

本 README はバージョン番号の更新に合わせて更新しています。それ以外のタイミングでは、
上記の移植元コミットより後の変更を本リポジトリが含んでいる場合があります。

> 名称について
> 公開リポジトリ名は `attic-shogi`、ビルドされるバイナリ名は `attic`、USI の
> `id name` は `Attic 9.70` です。

## ライセンスと帰属

本ソフトウェアは GNU General Public License v3（GPLv3） で配布されます。全文は
[`LICENSE`](LICENSE) を参照してください。

本ソフトウェアは GPLv3 で公開されている YaneuraOu の派生物です。移植元の著作権は
yaneurao 氏および YaneuraOu の各コントリビューターに帰属します。上流のソースコードは
以下で公開されています。

- YaneuraOu: <https://github.com/yaneurao/YaneuraOu>

## 対応環境

- Linux x86_64 （Ubuntu 24.04 相当を想定）。他の OS は対象外です。WSL2 上の
  Ubuntu でも同じ手順で動作します。
- 評価関数（NNUE）の計算に SIMD 実装を使うかは、ビルド時に決まります。実行時の
  CPU 判定は行いません。選択の基準はビルドが有効化している CPU 機能で、AVX-512 の
  F + BW が有効なら特徴量変換器と要素単位の演算が SIMD 実装になり、さらに VNNI も
  有効ならレイヤースタックをまとめて計算する処理も SIMD 実装になります。いずれも有効で
  なければスカラ実装です。本リポジトリは
  `-C target-cpu=native` でビルドする（[ビルドと実行](#ビルドと実行)参照）ため、
  実際に選ばれる実装はビルドしたマシンの CPU で決まります。
- どちらの実装が選ばれても、評価値はビット単位で一致します（近似ではなく同じ値を返すことを、SIMD とスカラの等価性テストで検証しています）。
- 生成されたバイナリはビルドしたマシンの CPU 向けです（[ビルドと実行](#ビルドと実行)参照）。

## ビルドと実行

ビルドには `rustup`（Rust のインストーラ兼ツールチェイン管理ツール）が必要です。
未導入の場合は公式の手順（<https://rustup.rs/>）でインストールしてください。また、
リンクに C コンパイラを使うため、Ubuntu では `build-essential` パッケージも
入れておいてください（`sudo apt install build-essential`）。

ビルド時に使用する Rust ツールチェインのバージョンを [`rust-toolchain.toml`](rust-toolchain.toml) に明記しています。
そのため、後述する `cargo build --release` の実行時に、`rustup` が該当バージョンを自動的にダウンロードします。

クローンしたディレクトリに移動してから、リリースビルドを行います。

```bash
cargo build --release
```

- ビルドされる実行ファイルは `target/release/attic` です。
- [`.cargo/config.toml`](.cargo/config.toml) が全プロファイルに
  `-C target-cpu=native` を適用します。生成されるバイナリはビルドしたマシンの
  CPU に最適化されるため、実行するマシン上でビルドしてください。

エンジンとのやり取りは、標準入出力上の USI プロトコルで行います。引数なしで
起動すると、USI コマンドの入力待ちになります。

```bash
./target/release/attic
```

正しくビルドできたかは、`usi` コマンドへの応答で確認できます。

```bash
printf 'usi\nquit\n' | ./target/release/attic
```

`id name` / `id author` と各 `option` 行、最後に `usiok` が出力されれば成功です。

### Windows の USI GUI から WSL2 経由で使う

USI は標準入力／標準出力でやり取りするだけなので、`wsl.exe` を介してエンジンを起動する
Windows 側の `.bat` を用意すれば、それをエンジンの実行ファイルとして登録できます
（`.bat` を登録できるかどうかは GUI 次第です）。

```bat
@echo off
set REPO=/home/<user>/<clone-dir>
wsl.exe --cd %REPO% ^
  -- ./target/release/attic
```

- `--cd` によりリポジトリのルートが作業ディレクトリになるため、`EvalDir` の既定値
  （`eval`）はリポジトリの `eval/` を指します。別の方法として、GUI のエンジン設定で
  `EvalDir` に WSL 側の絶対パスを指定してもかまいません。
- WSL のディストリビューションを複数インストールしている場合は、`-d <distribution>`
  で使用するものを選べます。

この手順は本リポジトリでは動作確認していません。

### どちらの NNUE 実装が選ばれたかを確認する

評価関数の SIMD 実装はビルド時に固定される（[対応環境](#対応環境)参照）ので、選択結果は
できあがったバイナリを逆アセンブルすれば確認できます。VNNI の内積命令 `vpdpbusd` の
出現数を数えます。

```bash
objdump -d target/release/attic | grep -c vpdpbusd
```

- 0 以外なら AVX-512（F + BW、VNNI）実装が選ばれています。
- 0 ならスカラ実装です。選ばれなかった側の実装は、どこからも呼ばれないため
  リリースビルドのデッドコード除去で実行ファイルから取り除かれます。

実測例（AMD EPYC 9B45 / Zen 5 上の `cargo build --release`）: 既定の
`-C target-cpu=native` ビルドで 29、`target-cpu` を AVX-512 を持たない
`x86-64-v2` に上書きしたビルドで 0 でした。

なお `vpdpbusd` は VNNI 実装の有無を示す命令です。F + BW だけが有効で VNNI がない
CPU 向けのビルドでは、特徴量変換器などが SIMD 実装でも計数は 0 になります。その場合は
ZMM レジスタの使用有無で判別できます。

```bash
objdump -d target/release/attic | grep -c '%zmm'
```

上の 2 つのビルドでは、それぞれ 2382 と 0 でした。

## 評価関数ファイル（`nn.bin`）

本エンジンが読み込める評価関数は、YaneuraOu を
`YANEURAOU_EDITION=YANEURAOU_ENGINE_SFNN1536` でビルドした場合と同じ、
SFNNwoP1536（SFNN-1536）ネットワーク構成のものだけです。評価関数ファイルは本リポジトリに含まれていないため、
この形式の `nn.bin` を別途用意してください。入手方法の一例として、YaneuraOu
プロジェクトを支援する方法があります
（<https://yaneuraou.yaneu.com/support-the-project/>）。

用意した `nn.bin` は、作業ディレクトリの `eval/nn.bin` に置いてください。`EvalDir` の
既定値（`eval`）のまま読み込めます。

### ヘッダ検証（[`crates/attic-eval/src/loader.rs`](crates/attic-eval/src/loader.rs)）

読み込み時の検証は移植元に準じます。

- バージョン番号が一致しない場合は読み込みを中止します（別のシリアライズ
  形式のファイルとみなします）。
- ファイル全体のハッシュおよび各セクション（特徴量変換器・各レイヤースタック）
  のハッシュが一致しない場合は、`info string` で警告を出したうえで読み込みを
  続行します。
- アーキテクチャ文字列は読み取りますが、比較には使いません。

### 読み込みのタイミング

- `EvalDir`（既定値 `eval`）が評価関数ファイルの置かれたディレクトリを指定します。実際に
  読み込むファイルは `<EvalDir>/nn.bin` です。
- 読み込みは `isready` の時点で行われます。成功すると `readyok` を返します。失敗した
  場合は `info string eval load failed: …` を出力し、`readyok` は返しません。

### オプション上書きファイル

エンジンは `isready` のたびに、次の 2 つのファイルをこの順で読みます（存在しない
場合は何もしません）。

1. カレントディレクトリの `engine_options.txt`
2. `<EvalDir>/eval_options.txt`

これらのファイルの各行はオプションを上書きし、そのオプションの設定を固定（FIXED）します
（固定後は `setoption` による変更を無視します）。記法は次の 3 通りです。

- `FV_SCALE 24` のような `<名前> <値>`
- `FV_SCALE=24` のような `<名前>=<値>`
- `option name FV_SCALE type spin default 24 …` のような完全形（`default` の値を採用）

評価関数ファイルが推奨する `FV_SCALE` は、この上書きファイルを通じて適用します。

## USI の使い方

### オプション一覧

`usi` コマンドに対して出力されるオプションは次のとおりです（宣言順）。

| オプション名 | 型 | 既定値 | 意味 |
| --- | --- | --- | --- |
| `USI_Hash` | spin | `1024` | 置換表サイズ [MB]（1〜33554432） |
| `Threads` | spin | `4` | 探索スレッド数（1〜、上限はコア数に応じて動的） |
| `MultiPV` | spin | `1` | 候補手（読み筋）を出力する本数（1〜600） |
| `EvalDir` | string | `eval` | `nn.bin` を置くディレクトリ |
| `FV_SCALE` | spin | `16` | NNUE の出力を割って評価値にする除数（1〜128） |
| `USI_OwnBook` | check | `true` | エンジン側で定跡を使う |
| `NarrowBook` | check | `false` | 定跡の採用手を絞り込む |
| `BookMoves` | spin | `16` | 定跡を適用する手数（0〜10000） |
| `BookIgnoreRate` | spin | `0` | 定跡を無視する確率 [%]（0〜100） |
| `BookFile` | combo | `no_book` | 使用する定跡ファイル（既定は定跡なし） |
| `BookDir` | string | `book` | 定跡ファイルを置くディレクトリ |
| `BookEvalDiff` | spin | `30` | 定跡採用手の評価値の許容差（0〜99999） |
| `BookEvalBlackLimit` | spin | `0` | 先手番で定跡を採用する評価値の下限 |
| `BookEvalWhiteLimit` | spin | `-140` | 後手番で定跡を採用する評価値の下限 |
| `BookDepthLimit` | spin | `16` | 定跡として採用する最小の深さ（0〜99999） |
| `BookOnTheFly` | check | `false` | 定跡を全読み込みせず逐次参照する |
| `ConsiderBookMoveCount` | check | `false` | 定跡手の採用回数を考慮する |
| `BookPvMoves` | spin | `8` | 定跡から出力する PV の手数（1〜246） |
| `IgnoreBookPly` | check | `false` | 定跡照合時に手数を無視する |
| `FlippedBook` | check | `true` | 左右反転した局面も定跡照合する |
| `EnteringKingRule` | combo | `CSARule27` | 入玉宣言勝ちのルール |
| `DepthLimit` | spin | `0` | 探索深さの上限（0 = 無制限） |
| `NodesLimit` | spin | `0` | 探索ノード数の上限（0 = 無制限） |
| `MaxMovesToDraw` | spin | `0` | 引き分けとする手数（0 = 無制限） |
| `PvInterval` | spin | `300` | PV 出力の最小間隔 [ms]（0 = 抑制しない） |
| `ConsiderationMode` | check | `false` | 検討モード |
| `OutputFailLHPV` | check | `true` | fail-high/low 時にも PV を出力する |
| `DrawValueBlack` | spin | `-2` | 先手から見た引き分けの評価値 |
| `DrawValueWhite` | spin | `-2` | 後手から見た引き分けの評価値 |
| `ResignValue` | spin | `99999` | 投了する評価値のしきい値 |
| `GenerateAllLegalMoves` | check | `false` | 不成なども含む全合法手を生成する |
| `NetworkDelay` | spin | `120` | 平均通信遅延 [ms] |
| `NetworkDelay2` | spin | `1120` | 最悪時（時間切れ回避）の通信遅延 [ms] |
| `MinimumThinkingTime` | spin | `2000` | 最小思考時間 [ms] |
| `SlowMover` | spin | `100` | 思考時間の倍率 [%] |
| `RoundUpToFullSecond` | check | `true` | 秒単位に切り上げて時間を使う（秒読み用） |
| `NumaPolicy` | string | `auto` | NUMA ノードへの割り当て方針 |
| `USI_Ponder` | check | `false` | 先読み（ponder）を有効化する |
| `Stochastic_Ponder` | check | `false` | 確率的 ponder を有効化する |

### 対応コマンド

| コマンド | 説明 |
| --- | --- |
| `usi` | エンジン情報とオプション一覧を出力し `usiok` を返す |
| `isready` | 定跡と評価関数を読み込み、成功すれば `readyok` を返す |
| `setoption name <名前> value <値>` | オプションを設定する |
| `usinewgame` | 新規対局の開始（出力なし） |
| `position [startpos \| sfen <SFEN>] [moves <手> …]` | 局面を設定する |
| `go […]` | 探索を開始し `bestmove` を返す |
| `stop` | 探索を停止する |
| `ponderhit` | 先読みが的中したことを通知する |
| `gameover` | 対局終了 |
| `quit` | 終了する |
| `bench [ttSizeMB] [threads] [limit] [default\|current\|<fenFile>] [limitType]` | 固定条件での NPS 計測。引数はすべて省略可で、左から順に既定値（`ttSizeMB=1024`, `threads=1`, `limit=15000`, ソース `default`, `limitType=movetime`）で埋められる |

コマンドライン用のサブコマンドとして、perft（指し手生成の数え上げ）も利用できます
（[`crates/attic/src/main.rs`](crates/attic/src/main.rs)）。

```bash
attic perft startpos <depth>
attic perft sfen <SFEN> <depth>
attic perft sfen <SFEN> moves <m1> [<m2> …] <depth>
```

### 未対応のコマンド

移植元にはデバッグ・補助用のコマンドがいくつかありますが、本エンジンでは実装して
いません。たとえば `d`、`eval`、`moves`、`flip`、`getoption`、`compiler`、
`unittest`、`test` などです。認識できないコマンドを受け取った場合は
`info string unknown command: <入力行>` を出力して読み飛ばします。

## 移植元との相違点

移植元との主な違いは次のとおりです（いずれも本リポジトリのコードで確認できます）。

- 対応環境が Linux x86_64 のみです（[対応環境](#対応環境)参照）。
- 定跡は `.ybb`（YaneuraOu のバイナリ定跡）形式のみ読み込みます。これに合わせて
  `BookFile` オプションの選択肢も、実際に読み込める `.ybb` の名前だけを提示します
  （移植元は `.db` / `book.bin` の名前を並べます）。
- 未対応の補助コマンドがあります（[未対応のコマンド](#未対応のコマンド)参照）。
- 探索スレッドごとの履歴テーブルの確保方法が異なります。移植元は履歴テーブル群を
  `Worker` 構造体に内蔵して 1 回の large page 確保で持ちますが、本エンジンはテーブル
  単位の large page 確保に分けています（移植元と同じ一括確保の形は、実測で NPS が
  約 1〜1.5% 低下したため採用していません。探索結果・出力には影響しません）。
- 上記のような意図的な仕様差は、該当するコード箇所にコメントで明記しています
  （例: `BookFile` の既定値を `no_book` にする分岐。
  [`crates/attic-protocol/src/options.rs`](crates/attic-protocol/src/options.rs)）。

## 移植元との性能比較

参考値として、本移植（V9.70 時点）と移植元 YaneuraOu V9.70（TOURNAMENT ビルド、
AVX-512 VNNI）を、同一マシン（AMD EPYC 9B45 = Zen 5、Linux）・同一評価関数で
比較した結果です。

- NPS（1 スレッド、固定深さの `bench`）は移植元を 1 割ほど下回ります。
- レーティングは、互角局面から 100 局（持ち時間 2 秒+1 手 1 秒加算、1 スレッド）の
  対局で、その速度差に見合う程度（10 前後）下回る結果でした。局数が少ないため
  誤差は大きめです。
- なお固定深さの探索では、ノード数が移植元と完全に一致します
  （つまり、探索の中身は同じで、上記の差は主に速度によるものです）。
