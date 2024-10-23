# 使用 ultralytics/ultralytics:latest-python 作為基礎映像
FROM ultralytics/ultralytics:latest-python

# 更新 pip 以避免相依性問題
RUN pip install --upgrade pip

# 安裝指定版本的 cvat-sdk 和 timm
RUN pip install cvat-sdk==2.7.6 timm==1.0.9 scikit-learn flask comet_ml
RUN pip install -U ultralytics==8.3.2

# 下載 nuctl 並移動到 /usr/local/bin
RUN curl -L https://github.com/nuclio/nuclio/releases/download/1.11.24/nuctl-1.11.24-linux-amd64 -o /usr/local/bin/nuctl

# 賦予執行權限
RUN chmod +x /usr/local/bin/nuctl

# 設置 PATH 環境變量
ENV PATH="/usr/local/bin:${PATH}"

# 預設執行 bash（可根據需要更改為其他執行指令）
CMD ["bash"]
