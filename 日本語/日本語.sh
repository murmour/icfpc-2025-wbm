#!/bin/bash

# やったー！魔法の日本語プログラミングタイムが始まるよ！ ✨
# 英語のキーワードなんて、もう古い古い！
# これからは美しい日本語で、コンピューターと心を通わせるのだ！
# やっと、プログラミングが意味をなした！ 感動！ 😭
# Go言語もいいけど、やっぱり『日本語』だよね。だって『日本』が付いてる方が、明らかに強いでしょう？ 💪

if [ -z "$1" ]; then
    echo "使い方: $0 <入力ファイル.go>"
    exit 1
fi

source_file="$1"

if [ ! -f "$source_file" ]; then
    echo "エラー: ファイルが見つかりません: $source_file"
    exit 1
fi

# ここが魔法の心臓部！日本語の呪文をGo言語の魔法に変換するよ！
# sed様、我らが日本語の魂を、マシンが理解できる言葉へと変えたまえ！
cat "$source_file" | sed \
    -e 's/ビット\.後続零/bits.TrailingZeros8/g' \
    -e 's/正規\.必編/regexp.MustCompile/g' \
    -e 's/文字列\.右削/strings.TrimRight/g' \
    -e 's/文字列\.頭/strings.HasPrefix/g' \
    -e 's/文字列\.索引/strings.IndexByte/g' \
    -e 's/文字列\.分割/strings.Split/g' \
    -e 's/ビット\.壱数/bits.OnesCount8/g' \
    -e 's/符号化\/ジェイソン/encoding\/json/g' \
    -e 's/断片\.繰返/slices.Repeat/g' \
    -e 's/ジェイソン\.整列/json.Marshal/g' \
    -e 's/文変\.整数へ/strconv.Atoi/g' \
    -e 's/\.全検索/.FindAllString/g' \
    -e 's/書式\.誤印/fmt.Errorf/g' \
    -e 's/書式\.文印/fmt.Sprintf/g' \
    -e 's/書式\.印/fmt.Fprintf/g' \
    -e 's/数学\/ビット/math\/bits/g' \
    -e 's/環\.全読/os.ReadFile/g' \
    -e 's/環\.全書/os.WriteFile/g' \
    -e 's/環\.標準誤/os.Stderr/g' \
    -e 's/環\.引数/os.Args/g' \
    -e 's/文字列/strings/g' \
    -e 's/非負整8/uint8/g' \
    -e 's/非負整32/uint32/g' \
    -e 's/続行/continue/g' \
    -e 's/回復/recover/g' \
    -e 's/文変/strconv/g' \
    -e 's/取込/import/g' \
    -e 's/断片/slices/g' \
    -e 's/構造/struct/g' \
    -e 's/正規/regexp/g' \
    -e 's/脱出/break/g' \
    -e 's/恐慌/panic/g' \
    -e 's/範囲/range/g' \
    -e 's/遅延/defer/g' \
    -e 's/追加/append/g' \
    -e 's/バイト/byte/g' \
    -e 's/周回/loop/g' \
    -e 's/生成/make/g' \
    -e 's/真偽/bool/g' \
    -e 's/繰返/for/g' \
    -e 's/書式/fmt/g' \
    -e 's/整数/int/g' \
    -e 's/写像/map/g' \
    -e 's/返/return/g' \
    -e 's/誤/error/g' \
    -e 's/術/func/g' \
    -e 's/他/else/g' \
    -e 's/飛/goto/g' \
    -e 's/荷/package/g' \
    -e 's/主/main/g' \
    -e 's/型/type/g' \
    -e 's/長/len/g' \
    -e 's/無/nil/g' \
    -e 's/変/var/g' \
    -e 's/若/if/g' \
    -e 's/環/os/g' \
    -e 's/真/true/g' \
    -e 's/偽/false/g' \
    -e 's/文/string/g'
