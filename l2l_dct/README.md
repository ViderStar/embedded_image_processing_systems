# L2L DCT — система сжатия изображений

Семестровый проект по дисциплине **«Встраиваемые системы обработки изображений для задач мультимедиа»**.

Реализация L2L (lossless-to-lossy) кодера на основе блочной лестничной структурной параметризации **ДКП–ОДКП** с арифметикой с фиксированной запятой. Теория: Петровский А.А. и др., *Системы обработки медиаданных*, БГУИР, 2018, гл. 4–6 ([Petrovskii_018.pdf](../Petrovskii_018.pdf)).

## Структура

```
l2l_dct/
├── python/          # основная реализация (Python)
├── matlab/          # сверка с Fixed-Point Designer
├── tests/           # unit/integration тесты
├── run_phase*.py    # демонстрационные скрипты
├── docs/            # архитектура и FPGA
├── report.tex       # отчёт
└── output/          # PNG, метрики (генерируется)
fpga/                # HLS + Pynq overlay
```

## Быстрый старт

```bash
cd l2l_dct
pip install -r requirements.txt

python3 run_phase1_dct.py        # ДКП/ОДКП (Лофлер, fixed-point)
python3 run_phase2_chessboard.py # эффект «шахматной доски» ± SIB
python3 run_phase3_l2l_lossless.py
python3 run_phase4_l2l_lossy.py

python3 tests/test_loeffler.py
python3 tests/test_sib_chessboard.py
```

## Ключевые результаты

| Режим | PSNR (типично) | Описание |
|-------|----------------|----------|
| Без SIB | ~63 dB | видимая «шахматная доска» (артефакт блочности) |
| С SIB | ∞ | bit-exact восстановление (lossless) |
| Lossy | настраивается | квантование + RLE/zlib |

## FPGA (Pynq)

```bash
cd fpga/scripts
python3 export_golden.py
vitis_hls -f build_hls.tcl   # C-simulation + synth
```

На плате Pynq: `fpga/pynq/l2l_overlay.ipynb`.

## Отчёт

```bash
cd l2l_dct
pdflatex report.tex
pdflatex report.tex
```
