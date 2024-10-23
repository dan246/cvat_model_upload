from flask import Flask, render_template, request, redirect, url_for, flash, Response, stream_with_context
import os
import requests
import yaml
import shutil
import subprocess
import stat
from ultralytics import YOLO
import json

app = Flask(__name__)
app.secret_key = 'your_secret_key'

BASE_DIR = '/app'
SAMPLE_TEMPLATE_DIR = os.path.join(BASE_DIR, 'sample')
NUCTL_PATH = os.path.join(BASE_DIR, 'nuctl-1.11.24-linux-amd64')

# 自訂表示器以使用區塊字串格式
class LiteralString(str):
    pass

def literal_str_representer(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')

# 將表示器新增至 SafeDumper
yaml.SafeDumper.add_representer(LiteralString, literal_str_representer)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        model_url = request.form.get('model_url')
        sample_name = request.form.get('sample_name')

        if not model_url or not sample_name:
            flash('請提供模型 URL 和範例名稱。')
            return redirect(url_for('index'))

        sample_dir = os.path.join(BASE_DIR, sample_name)
        if os.path.exists(sample_dir):
            flash('範例名稱已存在，請選擇其他名稱。')
            return redirect(url_for('index'))

        os.makedirs(sample_dir)

        try:
            # 複製 deploy_gpu.sh 和 function-gpu.yaml
            shutil.copy(os.path.join(SAMPLE_TEMPLATE_DIR, 'deploy_gpu.sh'), sample_dir)
            shutil.copy(os.path.join(SAMPLE_TEMPLATE_DIR, 'function-gpu.yaml'), sample_dir)
            
            # 根據模型 URL 是否包含 "seg" 來決定複製哪個 main 檔案
            if 'seg' in model_url.lower():
                shutil.copy(os.path.join(SAMPLE_TEMPLATE_DIR, 'main_seg.py'), os.path.join(sample_dir, 'main.py'))
            else:
                shutil.copy(os.path.join(SAMPLE_TEMPLATE_DIR, 'main.py'), sample_dir)
        except Exception as e:
            flash(f'複製範本檔案失敗：{str(e)}')
            return redirect(url_for('index'))

        MODEL_PATH = os.path.join(sample_dir, 'best.pt')
        FUNCTION_YAML_PATH = os.path.join(sample_dir, 'function-gpu.yaml')

        try:
            r = requests.get(model_url)
            r.raise_for_status()
            with open(MODEL_PATH, 'wb') as f:
                f.write(r.content)
        except Exception as e:
            flash(f'下載模型失敗：{str(e)}')
            return redirect(url_for('index'))

        try:
            model = YOLO(MODEL_PATH)
            labels = model.names
        except Exception as e:
            flash(f'載入模型失敗：{str(e)}')
            return redirect(url_for('index'))

        label_list = [{'id': int(idx) + 1, 'name': name} for idx, name in labels.items()]
        # 將 label_list 轉換為 JSON 字串
        label_json = json.dumps(label_list, ensure_ascii=False, indent=4)

        try:
            with open(FUNCTION_YAML_PATH, 'r') as f:
                yaml_content = yaml.safe_load(f)
        except Exception as e:
            flash(f'讀取 YAML 範本失敗：{str(e)}')
            return redirect(url_for('index'))

        try:
            yaml_content['metadata']['name'] = sample_name
            yaml_content['metadata']['annotations']['name'] = sample_name
            yaml_content['metadata']['annotations']['framework'] = 'pytorch'
            yaml_content['metadata']['annotations']['type'] = 'detector'
            # 使用 LiteralString 確保 'spec' 欄位以區塊字串格式寫入
            yaml_content['metadata']['annotations']['spec'] = LiteralString(label_json)
            yaml_content['spec']['description'] = sample_name
            yaml_content['spec']['build']['image'] = sample_name

            directives = yaml_content['spec']['build']['directives']
            for directive_list in directives.values():
                if isinstance(directive_list, list):
                    for directive in directive_list:
                        if 'wget' in directive.get('value', '') and 'best.pt' in directive.get('value', ''):
                            directive['value'] = f'wget {model_url}'
        except Exception as e:
            flash(f'更新 YAML 內容失敗：{str(e)}')
            return redirect(url_for('index'))

        try:
            with open(FUNCTION_YAML_PATH, 'w') as f:
                yaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False, Dumper=yaml.SafeDumper)
        except Exception as e:
            flash(f'寫入更新後的 YAML 檔案失敗：{str(e)}')
            return redirect(url_for('index'))

        flash('模型已更新，準備部署。')
        return redirect(url_for('deploy', sample_name=sample_name))

    return render_template('index.html')

@app.route('/deploy/<sample_name>')
def deploy(sample_name):
    sample_dir = os.path.join(BASE_DIR, sample_name)
    FUNCTION_YAML_PATH = os.path.join(sample_dir, 'function-gpu.yaml')

    if not os.path.exists(sample_dir):
        flash('範例不存在。')
        return redirect(url_for('index'))

    try:
        with open(FUNCTION_YAML_PATH, 'r') as f:
            yaml_content = f.read()
    except Exception as e:
        flash(f'讀取 YAML 檔案失敗：{str(e)}')
        return redirect(url_for('index'))

    return render_template('deploy.html', sample_name=sample_name, yaml_content=yaml_content)


@app.route('/deploy_logs/<sample_name>')
def deploy_logs(sample_name):
    sample_dir = os.path.join(BASE_DIR, sample_name)
    deploy_script = os.path.join(sample_dir, 'deploy_gpu.sh')

    def generate():
        try:
            # 確保部署腳本是可執行的
            os.chmod(deploy_script, os.stat(deploy_script).st_mode | stat.S_IEXEC)
            process = subprocess.Popen(
                ['bash', deploy_script],
                cwd=sample_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            # 逐行傳輸輸出
            while True:
                output = process.stdout.readline()
                if output:
                    yield f"data: {output.strip()}\n\n"
                elif process.poll() is not None:
                    break

            # 檢查是否有剩餘輸出
            remaining = process.stdout.read()
            if remaining:
                yield f"data: {remaining.strip()}\n\n"

            return_code = process.poll()
            if return_code == 0:
                yield "data: 部署成功！\n\n"
                yield "event: done\ndata: 部署成功！\n\n"
            else:
                yield f"data: 部署失敗，返回碼：{return_code}\n\n"
                yield "event: done\ndata: 部署失敗\n\n"
        except Exception as e:
            yield f"data: 執行部署腳本失敗：{str(e)}\n\n"
            yield "event: done\ndata: 部署失敗\n\n"

    return Response(generate(), mimetype='text/event-stream')

@app.route('/view_yaml/<sample_name>')
def view_yaml(sample_name):
    sample_dir = os.path.join(BASE_DIR, sample_name)
    FUNCTION_YAML_PATH = os.path.join(sample_dir, 'function-gpu.yaml')

    if not os.path.exists(FUNCTION_YAML_PATH):
        flash('YAML 檔案不存在。')
        return redirect(url_for('index'))

    try:
        with open(FUNCTION_YAML_PATH, 'r') as f:
            yaml_content = f.read()
    except Exception as e:
        flash(f'讀取 YAML 檔案失敗：{str(e)}')
        return redirect(url_for('index'))

    return render_template('view_yaml.html', yaml_content=yaml_content, sample_name=sample_name)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=15037)
