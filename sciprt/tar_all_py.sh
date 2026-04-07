#!/bin/bash

# 取得當前目錄的名稱 (例如：my_mnist)
DIR_NAME=$(basename "$(realpath .)")

# 退到上層目錄，這樣打包時抓到的路徑就會是 "my_mnist/*.py"
cd ..

# 使用 tar 打包 (推薦在 Linux 環境使用)
# 將生成的壓縮檔直接存回原本的目錄中
tar -czvf "${DIR_NAME}/${DIR_NAME}.tar.gz" "${DIR_NAME}/"*.py

# 如果您更喜歡 zip 格式，可以註解掉上面的 tar，改用下面這行：
# zip "${DIR_NAME}/${DIR_NAME}.zip" "${DIR_NAME}/"*.py

# 切回原本的目錄
cd "${DIR_NAME}"

echo "✅ 打包完成！已在當前目錄生成 ${DIR_NAME}.tar.gz"
