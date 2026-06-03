# Inha iClass 作业批量下载工具

自动登录 [learn.inha.ac.kr](https://learn.inha.ac.kr) 的课程,按**周次/章节**分文件夹下载所有作业(`mod/assign`)的说明内容与附件。

## 为什么要在本地运行?

Inha 的 iClass 登录走的是校方 **SSO(exsignon,`idp.inha.ac.kr:8443`)**,认证端点在**非标准端口 8443**。云端/受限网络环境通常只放行标准 443 端口,连不到 8443,因此**必须在你自己的电脑上运行**(校园网或普通家庭网络都能访问 8443)。

## 安装(只需一次)

```bash
pip install playwright beautifulsoup4
playwright install chromium
```

## 使用

```bash
# 推荐:用环境变量传账号,避免密码留在命令历史
export ICLASS_USER=22232331
export ICLASS_PASS='你的密码'
python download_assignments.py --course 69973
```

或直接运行,按提示交互式输入账号密码:

```bash
python download_assignments.py --course 69973
```

> `--course` 后面的数字就是课程链接 `course/view.php?id=69973` 里的 `69973`。

### 遇到验证码 / 二次验证 / 登录框找不到

加 `--headful` 参数会弹出浏览器窗口,你可以手动协助登录:

```bash
python download_assignments.py --course 69973 --headful
```

## 输出结构

```
iclass_downloads/
└── <课程名>/
    ├── 00_<第1周章节名>/
    │   └── <作业名>/
    │       ├── description.html   # 作业说明(网页原样)
    │       ├── description.txt    # 作业说明(纯文本)
    │       ├── info.json          # 元数据(链接、截止时间等)
    │       └── <附件文件...>
    ├── 01_<第2周章节名>/
    │   └── ...
    └── ...
```

文件夹名前缀的数字(`00_`、`01_`…)对应课程页上章节的先后顺序,方便按周次排序。

## 安全提示

- 脚本**不会**把你的密码写进任何文件;优先用环境变量或交互式输入。
- 不要把含密码的命令或文件提交到 git。
