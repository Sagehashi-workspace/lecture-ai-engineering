# day1 演習変更点
## 1 google/gemma-3-12b-itモデルの追加・ユーザ側がモデルを切り替えられるように変更
入力と出力の調整は細かく行えていませんが、とりあえずgemmaモデル2つを自由に切り替えられるようにしました。もう少し時間があればモデルごとに出力履歴をまとめるなどをしてみても良かったと思います。

## 2 正確性フィードバックを0.1刻みに変更
部分的に正確の程度を細かく指定できるようにしました。この変更に伴い、正確性スコアの方にも変更が入り、単純にスコアの数の分布を表示するようにしました。

## 3 軽微なUIの修正
