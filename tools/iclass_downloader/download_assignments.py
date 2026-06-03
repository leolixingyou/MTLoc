#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inha iClass (learn.inha.ac.kr) 作业批量下载工具
================================================

功能:
  1. 通过校方 SSO (exsignon, idp.inha.ac.kr:8443) 登录;
  2. 打开指定课程页, 解析 Moodle 的「周次 / 章节 (sections)」结构;
  3. 找出每个章节下的所有作业 (mod/assign) 链接;
  4. 抓取每个作业的说明文字 (intro) 与全部附件;
  5. 按周次 / 章节分别建立文件夹保存。

为什么用 Playwright:
  Inha 的登录走的是基于 JavaScript 跳转的 SSO, 认证端点在非标准端口 8443。
  用真实浏览器内核 (Playwright + Chromium) 驱动登录最稳定, 登录后再用浏览器
  自带的 cookie 上下文去抓取页面和下载文件。

--------------------------------------------------------------------
安装 (只需做一次):
    pip install playwright beautifulsoup4
    playwright install chromium

运行:
    # 方式 A: 用环境变量传账号 (推荐, 不会把密码留在命令历史里)
    export ICLASS_USER=22232331
    export ICLASS_PASS='你的密码'
    python download_assignments.py --course 69973

    # 方式 B: 直接交互式输入 (运行后按提示输入)
    python download_assignments.py --course 69973

    # 调试 / 遇到验证码时, 显示浏览器窗口手动协助:
    python download_assignments.py --course 69973 --headful

输出:
    ./iclass_downloads/<课程名>/<周次或章节名>/<作业名>/
        description.html   # 作业说明 (网页原样)
        description.txt    # 作业说明 (纯文本)
        info.json          # 作业元数据 (链接, 截止时间等)
        <附件...>          # 作业附带的文件
--------------------------------------------------------------------
"""

import argparse
import json
import os
import re
import sys
import getpass
from pathlib import Path
from urllib.parse import urljoin, urlparse, parse_qs

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
except ImportError:
    sys.exit("缺少 playwright, 请先运行:\n    pip install playwright beautifulsoup4\n    playwright install chromium")

try:
    from bs4 import BeautifulSoup
except ImportError:
    sys.exit("缺少 beautifulsoup4, 请先运行: pip install beautifulsoup4")


BASE = "https://learn.inha.ac.kr"


# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------
def safe_name(name: str, maxlen: int = 120) -> str:
    """把任意字符串变成安全的文件 / 文件夹名。"""
    name = (name or "").strip()
    name = re.sub(r"[\\/:*?\"<>|\r\n\t]+", "_", name)
    name = re.sub(r"\s+", " ", name).strip(" .")
    if not name:
        name = "untitled"
    return name[:maxlen]


def log(msg: str):
    print(f"[iclass] {msg}", flush=True)


# ----------------------------------------------------------------------
# 登录
# ----------------------------------------------------------------------
def do_login(page, username: str, password: str, headful: bool):
    """驱动浏览器完成 SSO 登录。"""
    log("打开登录页 ...")
    page.goto(f"{BASE}/login.php", wait_until="domcontentloaded", timeout=60000)

    # 等待跳转到 SSO (idp.inha.ac.kr) 或出现输入框
    try:
        page.wait_for_load_state("networkidle", timeout=30000)
    except PWTimeout:
        pass

    log(f"当前页面: {page.url}")

    # 常见的用户名 / 密码输入框选择器 (尽量覆盖 exsignon 与 Moodle 两种登录页)
    user_selectors = [
        "input#userid", "input#username", "input#user_id", "input[name='userid']",
        "input[name='username']", "input[name='user_id']", "input[name='login_id']",
        "input[name='id']", "input[type='text']:visible", "input[type='email']:visible",
    ]
    pass_selectors = [
        "input#userpwd", "input#password", "input#user_password",
        "input[name='userpwd']", "input[name='password']", "input[name='user_password']",
        "input[name='login_pwd']", "input[type='password']:visible",
    ]

    def first_match(selectors):
        for sel in selectors:
            try:
                loc = page.locator(sel).first
                if loc.count() > 0 and loc.is_visible():
                    return loc
            except Exception:
                continue
        return None

    # SSO 页面可能稍慢, 多等一会儿
    user_box = None
    for _ in range(20):
        user_box = first_match(user_selectors)
        if user_box:
            break
        page.wait_for_timeout(500)

    if not user_box:
        if headful:
            log("没有自动找到登录框。请在弹出的浏览器里手动登录, 登录完成后回到终端按回车继续 ...")
            input()
            return
        raise RuntimeError(
            "找不到登录输入框, 登录页结构可能变了。\n"
            "请改用  --headful  参数运行, 手动在浏览器里登录。"
        )

    pass_box = first_match(pass_selectors)
    log("填写账号密码 ...")
    user_box.fill(username)
    if pass_box:
        pass_box.fill(password)

    # 提交: 优先回车, 否则点登录按钮
    submitted = False
    for sel in ["button[type='submit']", "input[type='submit']",
                "a#loginButton", "button#loginButton", "#btnLogin", ".btn-login"]:
        try:
            btn = page.locator(sel).first
            if btn.count() > 0 and btn.is_visible():
                btn.click()
                submitted = True
                break
        except Exception:
            continue
    if not submitted:
        (pass_box or user_box).press("Enter")

    # 等待跳回 learn.inha.ac.kr
    try:
        page.wait_for_url(re.compile(r"learn\.inha\.ac\.kr"), timeout=60000)
        page.wait_for_load_state("networkidle", timeout=30000)
    except PWTimeout:
        pass

    log(f"登录后页面: {page.url}")
    # 校验是否已登录 (登出链接 / 用户菜单)
    html = page.content()
    if "login" in page.url.lower() and "logout" not in html.lower():
        if headful:
            log("似乎仍未登录 (可能有验证码 / 二次验证)。请在浏览器里完成, 然后回终端按回车 ...")
            input()
        else:
            raise RuntimeError("登录失败, 请检查账号密码; 若有验证码请用 --headful 运行。")
    log("登录成功。")


# ----------------------------------------------------------------------
# 解析课程页, 提取「章节 -> 作业链接」
# ----------------------------------------------------------------------
def parse_course(page, course_id: str):
    url = f"{BASE}/course/view.php?id={course_id}"
    log(f"打开课程页: {url}")
    page.goto(url, wait_until="domcontentloaded", timeout=60000)
    try:
        page.wait_for_load_state("networkidle", timeout=20000)
    except PWTimeout:
        pass

    soup = BeautifulSoup(page.content(), "html.parser")

    course_title = "course"
    h1 = soup.find(["h1"])
    if h1 and h1.get_text(strip=True):
        course_title = h1.get_text(strip=True)
    else:
        if soup.title:
            course_title = soup.title.get_text(strip=True)

    sections = []  # [{name, assignments:[{name,url}]}]

    # Moodle 课程页: 每个章节是 li.section / div.section
    sec_nodes = soup.select("li.section, div.section, .course-section, [data-region='course-section']")
    if not sec_nodes:
        sec_nodes = soup.select("ul.topics > li, ul.weeks > li")

    for idx, sec in enumerate(sec_nodes):
        # 章节标题
        sec_name = None
        for sel in [".sectionname", ".section-title", "h3.sectionname", "h3", ".content > h3"]:
            t = sec.select_one(sel)
            if t and t.get_text(strip=True):
                sec_name = t.get_text(strip=True)
                break
        if not sec_name:
            attr = sec.get("aria-label") or sec.get("data-sectionname")
            sec_name = attr.strip() if attr else f"Section_{idx}"

        # 该章节内的所有作业链接 (mod/assign)
        assigns = []
        for a in sec.select("a[href*='/mod/assign/view.php']"):
            href = urljoin(BASE, a.get("href"))
            # 作业名: 链接里的 instancename, 否则用链接文字
            nm = a.select_one(".instancename")
            name = nm.get_text(strip=True) if nm else a.get_text(strip=True)
            name = re.sub(r"(과제|Assignment|작업)\s*$", "", name).strip() or name
            assigns.append({"name": name, "url": href})

        # 去重
        seen, uniq = set(), []
        for x in assigns:
            if x["url"] not in seen:
                seen.add(x["url"])
                uniq.append(x)

        if uniq:
            sections.append({"index": idx, "name": sec_name, "assignments": uniq})

    # 兜底: 如果按章节没解析出来, 就全页面收集所有作业链接放到一个组里
    if not sections:
        log("按章节未解析到作业, 改为全页面收集所有作业链接 ...")
        assigns, seen = [], set()
        for a in soup.select("a[href*='/mod/assign/view.php']"):
            href = urljoin(BASE, a.get("href"))
            if href in seen:
                continue
            seen.add(href)
            nm = a.select_one(".instancename")
            name = nm.get_text(strip=True) if nm else a.get_text(strip=True)
            assigns.append({"name": name, "url": href})
        if assigns:
            sections.append({"index": 0, "name": "all_assignments", "assignments": assigns})

    total = sum(len(s["assignments"]) for s in sections)
    log(f"课程《{course_title}》共解析到 {len(sections)} 个章节, {total} 个作业。")
    return course_title, sections


# ----------------------------------------------------------------------
# 抓取单个作业内容 + 附件
# ----------------------------------------------------------------------
def fetch_assignment(context, page, assign, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"  -> 作业: {assign['name']}")
    page.goto(assign["url"], wait_until="domcontentloaded", timeout=60000)
    try:
        page.wait_for_load_state("networkidle", timeout=15000)
    except PWTimeout:
        pass

    soup = BeautifulSoup(page.content(), "html.parser")

    # 作业说明区 (Moodle: .activity-description / #intro / .box.generalbox)
    intro = None
    for sel in ["#intro", ".activity-description", ".box.generalbox.boxaligncenter",
                "[data-region='activity-information']", ".no-overflow"]:
        node = soup.select_one(sel)
        if node and node.get_text(strip=True):
            intro = node
            break
    if intro is None:
        intro = soup.select_one("div[role='main']") or soup.body

    # 保存说明
    (out_dir / "description.html").write_text(str(intro), encoding="utf-8")
    (out_dir / "description.txt").write_text(intro.get_text("\n", strip=True), encoding="utf-8")

    # 截止时间等表格信息
    meta = {"name": assign["name"], "url": assign["url"], "fields": {}}
    for row in soup.select("table.generaltable tr, .submissionsummarytable tr"):
        cells = row.find_all(["th", "td"])
        if len(cells) >= 2:
            k = cells[0].get_text(" ", strip=True)
            v = cells[1].get_text(" ", strip=True)
            if k:
                meta["fields"][k] = v

    # 收集附件链接 (pluginfile / resource / 下载链接)
    file_links = []
    for a in intro.select("a[href]") + soup.select(
        "a[href*='pluginfile.php'], .fileuploadsubmission a[href], "
        ".assignsubmission_file a[href], a[href*='/mod/resource/view.php']"
    ):
        href = a.get("href")
        if not href:
            continue
        absu = urljoin(assign["url"], href)
        # 只要看起来是文件 (pluginfile) 或下载
        if "pluginfile.php" in absu or "forcedownload" in absu or "/mod/resource/" in absu:
            fname = a.get_text(strip=True) or os.path.basename(urlparse(absu).path)
            file_links.append((absu, fname))

    # 去重下载
    seen = set()
    downloaded = []
    for absu, fname in file_links:
        if absu in seen:
            continue
        seen.add(absu)
        try:
            resp = context.request.get(absu, timeout=60000)
            if not resp.ok:
                log(f"     [跳过] {fname} (HTTP {resp.status})")
                continue
            # 文件名: 优先 Content-Disposition
            disp = resp.headers.get("content-disposition", "")
            m = re.search(r"filename\*?=(?:UTF-8'')?\"?([^\";]+)", disp)
            real = m.group(1) if m else (os.path.basename(urlparse(absu).path) or fname)
            real = safe_name(real)
            if not real or real in ("view.php", "index.php"):
                real = safe_name(fname) or "file"
            target = out_dir / real
            i = 1
            while target.exists():
                target = out_dir / f"{Path(real).stem}_{i}{Path(real).suffix}"
                i += 1
            target.write_bytes(resp.body())
            downloaded.append(target.name)
            log(f"     [下载] {target.name}")
        except Exception as e:
            log(f"     [失败] {fname}: {e}")

    meta["downloaded_files"] = downloaded
    (out_dir / "info.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Inha iClass 作业批量下载")
    ap.add_argument("--course", required=True, help="课程 id (course/view.php?id= 后面的数字)")
    ap.add_argument("--out", default="iclass_downloads", help="输出根目录")
    ap.add_argument("--headful", action="store_true", help="显示浏览器窗口 (调试 / 验证码时用)")
    args = ap.parse_args()

    username = os.environ.get("ICLASS_USER") or input("学号 (ICLASS_USER): ").strip()
    password = os.environ.get("ICLASS_PASS") or getpass.getpass("密码 (ICLASS_PASS): ")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not args.headful)
        context = browser.new_context(accept_downloads=True,
                                       user_agent="Mozilla/5.0 (X11; Linux x86_64) "
                                                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                                                  "Chrome/120.0 Safari/537.36")
        page = context.new_page()

        do_login(page, username, password, args.headful)
        course_title, sections = parse_course(page, args.course)

        course_dir = out_root / safe_name(course_title)
        course_dir.mkdir(parents=True, exist_ok=True)

        for s in sections:
            # 文件夹名: 用「序号_章节名」保证按周次排序
            folder = course_dir / safe_name(f"{s['index']:02d}_{s['name']}")
            log(f"章节 [{s['index']}] {s['name']}  ({len(s['assignments'])} 个作业)")
            for assign in s["assignments"]:
                a_dir = folder / safe_name(assign["name"])
                try:
                    fetch_assignment(context, page, assign, a_dir)
                except Exception as e:
                    log(f"  [作业失败] {assign['name']}: {e}")

        browser.close()

    log(f"全部完成! 文件保存在: {course_dir.resolve()}")


if __name__ == "__main__":
    main()
