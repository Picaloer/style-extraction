# style-extraction

一个 [Claude Code](https://claude.com/claude-code) skill：给它几段游戏实机视频或截图（动画、电影、插画也行），它会**测量**画面和声音，写出一份能复现的风格定义文档，输出自包含的 **HTML** 和分页 **PDF**。

A Claude Code skill that distills the visual and music style of a game (or film, animation) from gameplay videos or screenshots into a measured style document (HTML + PDF), laid out as an "Amber Diorama" art book.

## 它做什么

- **画面**：三层调色（暗部 / 中间调 / 亮部的面积和主色）、Lab 色相角、暗角、景深清晰带、像素尺度、光晕半径、色差和颗粒。
- **声音**：每秒响度、速度、调性、频段分工、打击成分。
- **文档**：定义、色彩、七层拆解（体块、表面、角色、光、空气、镜头与后期、界面）、动势、声音、守则、复现配方。
- **提示词**：中英文生图提示词和配乐提示词。
- 每条规律都带数值、出处（哪段素材的哪一秒）和证据标签：`实测` / `目测` / `起调` / `模型识别` / `资料`。

## 目录

```
SKILL.md                    skill 本体：完整流程和写作要求
tools/
  stylex.py                 全部命令：probe / sample / overlay / measure / profile / pick / light / post / pixel / audio / crops / build / pdf / shots / facts
  amber.css                 琥珀微缩画册样式（原版，不要改）
  amber_ext.css             扩展样式和打印规则
  example.content.json      content.json 示例
  example.crops.json        crops.json 示例
```

## 安装

把整个仓库放进 Claude Code 的 skills 目录，目录名用 `style-extraction`：

```bash
git clone https://github.com/Picaloer/style-extraction ~/.claude/skills/style-extraction
```

SKILL.md 第 0 步默认从 `/mnt/project-files/...` 或一个私有页面取工具。用本仓库时，直接用 `tools/` 里的文件：

```bash
W=<工作目录>
mkdir -p $W/tools
cp ~/.claude/skills/style-extraction/tools/* $W/tools/
python3 $W/tools/stylex.py -h | sed -n 4p
```

## 依赖

- 必需：ffmpeg 和 ffprobe，Python 3 + numpy + Pillow
- 导出 PDF 和截图：无头 Chromium。依次尝试 Python Playwright、Node Playwright、本机 Chrome / Chromium / Edge；也可以用环境变量 `CHROME_PATH` 指定
- 可选：`pip install fonttools brotli pypdf`，再装 poppler（提供 `pdftoppm`、`pdftotext`）。PDF 会小一半左右，还能做空白页和缩略图检查

Windows 上注意：路径总长超过 260 字符时浏览器会打不开文件，导出 PDF 前把 HTML 复制到短路径。

## 用法

对 Claude 说：

> 提炼这段视频的美术风格 `D:\videos\game.mp4`

或者直接调用 `/style-extraction`。流程是：盘点素材 → 抽帧看片 → 去水印 → 测画面 → 测声音 → 出配图和标本 → 写 `content.json` → `build` 出 HTML → `pdf` 出 PDF → `shots` 检查。详见 [SKILL.md](SKILL.md)。

## 局限

- 视频经过压缩，色值、像素尺寸和声音数值都是近似值。
- 脚本听不出乐器；只有分类模型或资料能支撑时才写乐器名。
- 素材只有一个场景时，规律只对这个场景成立。
