SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
while true; do
    read -p " path:$SCRIPT_DIR/log内のファイルを削除しますか？(y/n): " yn
    case "$yn" in
        [Yy]* ) 
			rm -rf SCRIPT_DIR/log/*
            echo "削除しました"
            break
            ;;
        [Nn]* ) 
            echo "処理を中止しました。"
            exit 0
            ;;
        * ) 
            echo "y または n で入力してください。"
            ;;
    esac
done