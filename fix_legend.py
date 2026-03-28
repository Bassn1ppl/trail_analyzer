import re

# Read the file
with open('RacePlanOptimized_PySide6.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove legend_items and dummy series creation
pattern = r'        legend_items = \[\s*\("Steep \(>±20%\)", QColor\(186, 33, 33, 150\)\),\s*\("Moderate \(±10-20%\)", QColor\(235, 96, 37, 145\)\),\s*\("Slight \(±5-10%\)", QColor\(245, 166, 35, 140\)\),\s*\("Flat \(±0-5%\)", QColor\(82, 179, 88, 135\)\),\s*\]\s*for label, color in legend_items:\s*dummy = QLineSeries\(\)\s*dummy\.setName\(label\)\s*pen = QPen\(color\)\s*pen\.setWidth\(6\)\s*dummy\.setPen\(pen\)\s*chart\.addSeries\(dummy\)'
content = re.sub(pattern, '', content, flags=re.DOTALL)

# Hide legend
content = content.replace('chart.legend().setVisible(True)', 'chart.legend().setVisible(False)')

# Remove duplicate data_series = []
lines = content.split('\n')
new_lines = []
skip_next = False
for i, line in enumerate(lines):
    if skip_next:
        skip_next = False
        continue
    if line.strip() == 'data_series = []' and i > 0 and 'data_series = []' in '\n'.join(lines[max(0, i-5):i]):
        skip_next = False
        continue
    new_lines.append(line)

content = '\n'.join(new_lines)

# Write back
with open('RacePlanOptimized_PySide6.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Legend removed and hidden from chart")
